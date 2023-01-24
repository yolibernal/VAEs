include("generate.jl")
include("visualize.jl")

using Flux
using Flux: @functor
using Flux.Data: DataLoader
using Flux.Losses: logitbinarycrossentropy
using Parameters: @with_kw
using MLDatasets
using BSON
using DrWatson: struct2dict
using Distributions: MvNormal, Normal, logpdf
using Statistics: mean, std
using Plots

using TensorBoardLogger, Logging, Random

tb_logger = TBLogger("runs/run", min_level=Logging.Info)

@with_kw mutable struct Args
    η = 1e-3                # learning rate
    β = (0.90, 0.999)       # decay of momentums
    λ = 0.0f0               # regularization paramater
    batch_size = 128        # batch size
    sample_size = 10        # sampling size for output
    epochs = 15             # number of epochs
    seed = 0                # random seed
    input_dim = 28^2        # image size
    u_dim = 10
    latent_dim = 2          # latent dimension
    hidden_dim = 500        # hidden dimension

    ivae = true             # use IVAE or VAE
    loss = "elbo"           # loss function to use: "elbo" or "vae"
    save_path = "output"    # results path
end

function get_dataloader(args::Args, test_data::Bool=false)
    if test_data == true
        X, y = MLDatasets.MNIST.testdata()
    else
        X, y = MLDatasets.MNIST.traindata()
    end

    X = reshape(Float32.(X), args.input_dim, :)
    y = Flux.onehotbatch(y, 0:9)
    # Use class labels as u
    u = y

    return DataLoader((X, y, u), batchsize=args.batch_size, shuffle=!test_data)
end

struct Encoder
    encoder_features
    encoder_μ
    encoder_logσ
end
@functor Encoder

Encoder(input_dim::Int, out_dim::Int, hidden_dim::Int) = Encoder(
    # encoder_features
    Dense(input_dim, hidden_dim, relu),
    # encoder_μ
    Dense(hidden_dim, out_dim),
    # encoder_logσ
    Dense(hidden_dim, out_dim),
)

function (encoder::Encoder)(Xu)
    h = encoder.encoder_features(Xu)
    encoder.encoder_μ(h), encoder.encoder_logσ(h)
end

function (encoder::Encoder)(X, u)
    Xu = vcat(X, u)
    h = encoder.encoder_features(Xu)
    encoder.encoder_μ(h), encoder.encoder_logσ(h)
end


# Decoder

Decoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) = Chain(
    Dense(latent_dim, hidden_dim, relu),
    Dense(hidden_dim, input_dim)
)

function vae_loss(X, X̂, μ, logσ)
    batch_size = size(X)[end]
    # Closed-form solution for Normal + Standard Normal: https://jamesmccaffrey.wordpress.com/2021/02/03/the-kullback-leibler-divergence-for-two-gaussian-distributions/
    # The @. macro makes sure that all operates are elementwise (https://github.com/alecokas/flux-vae/blob/master/conv-vae/main.jl#L81)
    kl_q_p = -0.5f0 * sum(@. (1.0f0 + 2 * logσ - μ^2 - exp(2 * logσ))) / batch_size

    # TODO: what is c? do we need it?
    # reconstruction_loss = (1 / 2c) * sum(norm(X - X̂)) / batch_size
    reconstruction_loss = logitbinarycrossentropy(X̂, X, agg=sum) / batch_size

    loss = reconstruction_loss + kl_q_p
    return loss
end

function elbo_loss(X̂, μ, prior_μ, logσ, prior_logσ, X, z)
    batch_size = size(X)[end]

    # decoder_dist
    # log_px_z = logpdf(MvNormal(reshape(X̂, :), decoder_σ), reshape(X, :)) / batch_size
    # encoder_dist
    log_qz_xu = logpdf(MvNormal(reshape(μ, :), reshape(exp.(logσ), :)), reshape(z, :)) / batch_size
    # prior_dist
    log_pz_u = logpdf(MvNormal(reshape(prior_μ, :), reshape(exp.(prior_logσ), :)), reshape(z, :)) / batch_size

    reconstruction_loss = logitbinarycrossentropy(X̂, X, agg=sum) / batch_size

    kl_q_p = log_qz_xu - log_pz_u

    loss = reconstruction_loss + kl_q_p
    return loss
end

function evaluate(args::Args, encoder, decoder, prior_encoder)
    dataloader = get_dataloader(args, true)

    loss = 0.0f0
    for (X, y, u) in dataloader
        prior_μ, prior_logσ = prior_encoder(u)

        μ, logσ = encoder(X, u)
        z = μ + randn(Float32, size(logσ)) .* exp.(logσ)
        X̂ = decoder(z)

        # TODO: refactor to not duplicate training
        if args.loss == "vae"
            loss = vae_loss(X, X̂, μ, logσ)
        elseif args.loss == "elbo"
            loss = elbo_loss(X̂, μ, prior_μ, logσ, prior_logσ, X, z)
        else
            error("Unknown loss: $(args.loss)")
        end
    end
    return loss
end

function train(; kws...)
    local loss

    args = Args(; kws...)
    dataloader = get_dataloader(args, false)
    opt = AdamW(args.η, args.β, args.λ)

    if args.ivae
        prior_encoder = Encoder(args.u_dim, args.latent_dim, args.hidden_dim)
        encoder = Encoder(args.input_dim + args.u_dim, args.latent_dim, args.hidden_dim)
    else
        prior_encoder = (u) -> (zeros(args.latent_dim, size(u)[end]), ones(args.latent_dim, size(u)[end]))
        encoder = Encoder(args.input_dim, args.latent_dim, args.hidden_dim)
    end

    decoder = Decoder(args.input_dim, args.latent_dim, args.hidden_dim)

    # TODO: generalize for other distributions
    # prior_dist = MvNormal
    # decoder_dist = MvNormal
    # encoder_dist = MvNormal

    # decoder_σ = 0.01f0

    parameters = Flux.params(encoder.encoder_features, encoder.encoder_μ, encoder.encoder_logσ, decoder)

    @info "Starting training, $(args.epochs) epochs..."
    for epoch in 1:args.epochs
        @info "Epoch $(epoch)..."
        for (X, _, u) in dataloader
            grad = Flux.gradient(parameters) do
                prior_μ, prior_logσ = prior_encoder(u)

                if args.ivae
                    μ, logσ = encoder(X, u)
                else
                    μ, logσ = encoder(X)
                end

                # encoder_dist
                z = μ + randn(Float32, size(logσ)) .* exp.(logσ)

                X̂ = decoder(z)

                if args.loss == "vae"
                    loss = vae_loss(X, X̂, μ, logσ)
                elseif args.loss == "elbo"
                    loss = elbo_loss(X̂, μ, prior_μ, logσ, prior_logσ, X, z)
                else
                    error("Unknown loss: $(args.loss)")
                end

                return loss
            end
            with_logger(tb_logger) do
                @info "Loss" loss = loss
            end
            Flux.Optimise.update!(opt, parameters, grad)
        end
        X̂ = generate_digits(prior_encoder, decoder, struct2dict(args))
        image = convert_to_image(X̂, struct2dict(args)[:sample_size])
        eval_loss = evaluate(args, encoder, decoder, prior_encoder)

        if args.latent_dim == 2
            y = 0:9
            u = Flux.onehotbatch(y, 0:9)
            prior_μ, prior_logσ = prior_encoder(u)
            priors_plot = visualize_priors_2d(prior_μ, prior_logσ, y)
            with_logger(tb_logger) do
                @info "Priors" priors = priors_plot log_step_increment = 0
            end
        end

        with_logger(tb_logger) do
            @info "Loss" eval_loss = eval_loss log_step_increment = 0
            @info "Digits" digits = image log_step_increment = 0
            @info "Histogram" histogram = histogram(reshape(X̂, :)) log_step_increment = 0
        end
    end

    # Save model
    model_path = joinpath(args.save_path, "ivae.bson")
    let encoder = encoder, decoder = decoder, args = struct2dict(args)
        BSON.@save model_path encoder decoder prior_encoder args
        @info "Model saved: $(model_path)"
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end


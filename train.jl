using Flux
using VAEs
using Parameters: @with_kw
using MLDatasets
using BSON
using DrWatson: struct2dict
using Logging
using TensorBoardLogger
using ArgParse
using Flux.Data: DataLoader
using Plots

args_settings = ArgParseSettings(autofix_names=true)
@add_arg_table! args_settings begin
    "--data-dir"
    help = "directory to store data"
    required = true

    "--visualize"
    help = "generate plots during training"
    action = :store_true
end

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

    ivae = true             # use iVAE or VAE
    save_path = "output"    # results path

    visualize               # visualize results
    data_dir
end

tb_logger = TBLogger("runs/run", min_level=Logging.Info)

function get_dataloader(args::Args, test_data::Bool=false)
    if test_data == true
        dataset = MNIST(:test, dir=args.data_dir)
    else
        dataset = MNIST(:train, dir=args.data_dir)
    end
    X, y = dataset.features, dataset.targets

    X = reshape(Float32.(X), args.input_dim, :)
    y = Flux.onehotbatch(y, 0:9)
    # Use class labels as u
    u = y

    return DataLoader((X, y, u) |> gpu, batchsize=args.batch_size, shuffle=!test_data)
end

function evaluate(args::Args, vae)
    dataloader = get_dataloader(args, true)

    loss = 0.0f0
    for (X, y, u) in dataloader
        if args.ivae
            X̂, μ, prior_μ, logσ, prior_logσ, z = vae(X, u)
        else
            X̂, μ, prior_μ, logσ, prior_logσ, z = vae(X)
        end

        loss = elbo_loss(X̂, μ, prior_μ, logσ, prior_logσ, X, z)
    end
    return loss
end

function train(; kws...)
    local loss
    local prior_μ_history

    prior_μ_history = []

    @show args = Args(; kws...)

    dataloader = get_dataloader(args, false)
    opt = AdamW(args.η, args.β, args.λ)

    decoder = Decoder(args.input_dim, args.latent_dim, args.hidden_dim)
    if args.ivae
        prior_encoder = Encoder(args.u_dim, args.latent_dim, args.hidden_dim)
        encoder = Encoder(args.input_dim + args.u_dim, args.latent_dim, args.hidden_dim)
        vae = iVAE(prior_encoder, encoder, decoder)
    else
        encoder = Encoder(args.input_dim, args.latent_dim, args.hidden_dim)
        vae = iVAE(encoder, decoder)
    end

    vae = vae |> gpu

    # TODO: generalize for other distributions
    # prior_dist = MvNormal
    # decoder_dist = MvNormal
    # encoder_dist = MvNormal

    # decoder_σ = 0.01f0

    parameters = Flux.params(vae)

    @info "Starting training, $(args.epochs) epochs..."
    for epoch in 1:args.epochs
        @info "Epoch $(epoch)..."
        for (X, _, u) in dataloader
            grad = Flux.gradient(parameters) do
                if args.ivae
                    X̂, μ, prior_μ, logσ, prior_logσ, z = vae(X, u)
                else
                    X̂, μ, prior_μ, logσ, prior_logσ, z = vae(X)
                end

                loss = elbo_loss(X̂, μ, prior_μ, logσ, prior_logσ, X, z)

                return loss
            end
            with_logger(tb_logger) do
                @info "Loss" loss = loss
            end
            Flux.Optimise.update!(opt, parameters, grad)
        end
        eval_loss = evaluate(args, vae)
        with_logger(tb_logger) do
            @info "Loss" eval_loss = eval_loss log_step_increment = 0
        end


        if args.visualize
            samples_per_class = 5
            X̂ = generate_digits(struct2dict(args), prior_encoder, decoder, samples_per_class=samples_per_class) |> cpu
            digits_plot = convert_to_image(X̂, num_columns=struct2dict(args)[:sample_size], num_rows=samples_per_class)

            latent_space_plot = visualize_latent_space(struct2dict(args), get_dataloader(args, true), encoder)

            with_logger(tb_logger) do
                @info "Digits" digits = digits_plot log_step_increment = 0
                @info "Digits" histogram = histogram(reshape(X̂, :)) log_step_increment = 0
                @info "Latent space" latent_space = latent_space_plot log_step_increment = 0
            end

            if args.latent_dim == 2
                y = 0:9
                u = Flux.onehotbatch(y, 0:9)
                prior_μ, prior_logσ = vae.prior_encoder(u)
                push!(prior_μ_history, prior_μ)

                priors_plot = visualize_priors_2d(prior_μ |> cpu, prior_logσ |> cpu, y |> cpu)
                # prior_history_plot = visualize_prior_mean_history_2d(prior_μ_history, prior_logσ)
                prior_history_plot = visualize_prior_mean_history_2d(prior_μ_history |> cpu)

                with_logger(tb_logger) do
                    @info "Latent space" priors = priors_plot log_step_increment = 0
                    @info "Latent space" prior_history = prior_history_plot log_step_increment = 0
                end
            end
        end

    end

    # Save model
    if !isdir(args.save_path)
        mkpath(args.save_path)
    end

    model_path = joinpath(args.save_path, "ivae.bson")
    let encoder = encoder |> cpu, decoder = decoder |> cpu, prior_encoder = prior_encoder |> cpu, args = struct2dict(args) |> cpu
        BSON.@save model_path encoder decoder prior_encoder args
        @info "Model saved: $(model_path)"
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    parsed_args = parse_args(args_settings, as_symbols=true)
    train(; parsed_args...)
end

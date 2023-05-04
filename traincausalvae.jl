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
using LinearAlgebra
using Images
using Flux.Losses: logitbinarycrossentropy
using CUDA
using Statistics

# https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988/2
ENV["GKSwstype"] = 100

args_settings = ArgParseSettings(autofix_names=true)
@add_arg_table! args_settings begin
    "--data-dir"
    help = "directory to store data"
    required = true

    "--visualize"
    help = "generate plots during training"
    action = :store_true
end

concepts = ["pendulum angle", "light angle", "shadow length", "shadow position"]

@with_kw mutable struct Args
    batch_size::Int = 32
    adam_η = 1e-3                # learning rate
    adam_β = (0.90, 0.999)       # decay of momentums
    adam_λ = 0.0f0               # regularization paramater

    pretrain_η = 10.0f0
    pretrain_γ = 0.25f0

    n_dim::Int = 4
    latent_dim::Int = 4
    input_dim::Int = 96 * 96 * 4
    hidden_dim::Int = 512
    α::Float32 = 1
    β::Float32 = 1
    γ::Float32 = 1

    epochs = 100
    epochs_pretrain = 15
    save_path = "output"    # results path

    visualize               # visualize results
    data_dir = "data/pendulum/"
end

tb_logger = TBLogger("runs/causalvae/run", min_level=Logging.Info)

function logpdf_normal(x, μ, logσ)
    return (@. -logσ - 0.5 * log(2π) - 0.5 * ((x - μ) / exp.(logσ))^2)
end

c = 100 # "arbitrary postive number"

# TODO: find different DAG constraint
H(A) = (n_dim = size(A)[1]; sum(diag((I + (c / n_dim) * A .* A)^n_dim)) - n_dim)
# H(A) = (n_dim = size(A)[1]; sum(diag((I + (c / n_dim) * abs.(A))^n_dim)) - n_dim)
# TODO: add l1 loss for A to make it sparse

function load_dataset(data_dir, test_data::Bool=false)
    if test_data
        dataset_dir = joinpath(data_dir, "test")
    else
        dataset_dir = joinpath(data_dir, "train")
    end
    imgpaths = readdir(dataset_dir, join=true)

    X = channelview.(load.(imgpaths))
    X = reshape.(X, :)
    X = reduce(hcat, X)

    imgnames = [f[1] for f in splitext.(basename.(imgpaths))]
    imgnames = chop.(imgnames, head=2, tail=0)
    u = split.(imgnames, "_")
    u = parse.(Float32, hcat(u...))

    u = (u .- mean(u, dims=2)) ./ std(u, dims=2)

    X, u
end

function get_dataloader(args::Args, test_data::Bool=false)
    if test_data == true
        dataset = load_dataset(args.data_dir, true)
    else
        dataset = load_dataset(args.data_dir, false)
    end
    X, u = dataset

    return DataLoader((X, u) |> gpu, batchsize=args.batch_size, shuffle=!test_data)
end

"Element-wise KL divergence between two normal distributions"
function kl_normal(μ1, σ1, μ2, σ2)
    return sum((@. 0.5 * (2 * log(σ2) - 2 * log(σ1) + (σ1^2 + (μ1 - μ2)^2) / σ2^2 - 1)))
end

function train(; kws...)
    local loss
    local loss_pretrain

    local NELBO
    local l_u
    local l_m
    local H_A

    local reconstruction_loss
    local kl_qϵ_pϵ
    local kl_qz_pz

    @show args = Args(; kws...)

    dataloader = get_dataloader(args, false)
    opt = AdamW(args.adam_η, args.adam_β, args.adam_λ)

    prior_encoder = Encoder(args.n_dim, args.latent_dim, args.hidden_dim) |> gpu
    encoder = Encoder(args.input_dim + args.n_dim, args.latent_dim, args.hidden_dim) |> gpu
    decoder = Decoder(args.input_dim, args.latent_dim, args.hidden_dim) |> gpu

    # A = UpperTriangular(randn(Float32, args.n_dim, args.n_dim))
    A = fill(0.5f0, (4, 4))
    A[diagind(A)] .= 0.0f0
    @info "Causal graph after initialization" A = A

    A = A |> gpu

    I_A = Matrix{Float32}(I, size(A)...) |> gpu

    parameters = Flux.params(A, prior_encoder, encoder, decoder)

    @info "Starting pre-training, $(args.epochs_pretrain) epochs..."
    # Initialize
    λ = 0.0f0
    c = 1.0f0
    H_A = H(A)
    H_A_prev = H_A

    # Pretrain causal graph using Augemented Lagrangian method
    for epoch in 1:args.epochs_pretrain
        @info "Epoch (pretrain) $(epoch)..."
        for (X, u) in dataloader
            grad = Flux.gradient(parameters) do
                batch_size = size(X)[end]

                A_ = A - Diagonal(A) # Force diagonal 0

                H_A = H(A_)

                l_u = norm(u - transpose(A_) * u)^2 / batch_size
                loss_pretrain = l_u + λ * H_A + (c / 2) * H_A^2

                # Update multiplier and penalty parameter
                λ = λ + c * H_A
                if abs(H_A) < args.pretrain_γ * abs(H_A_prev)
                    c = args.pretrain_η * c
                end
                H_A_prev = H_A
                return loss_pretrain
            end
            with_logger(tb_logger) do
                @info "Pretrain" loss_pretrain = loss_pretrain
            end
            Flux.Optimise.update!(opt, parameters, grad)
        end
    end

    @info "Causal graph after pretrain" A = A
    if args.visualize
        causal_graph_heatmap_plot = plot_causal_graph_heatmap(A |> cpu, concepts=concepts)
        causal_graph_plot = plot_causal_graph(A |> cpu, concepts=concepts)
        with_logger(tb_logger) do
            @info "Pretrain" heatmap = causal_graph_heatmap_plot causal_graph = causal_graph_plot log_step_increment = 0
        end
    end

    # TODO: force diagonal 0, e.g. substract diagonal or make diagonal not learnable
    @info "Starting training, $(args.epochs) epochs..."
    for epoch in 1:args.epochs
        @info "Epoch $(epoch)..."
        for (X, u) in dataloader
            grad = Flux.gradient(parameters) do
                A_ = A - Diagonal(A) # Force diagonal 0

                batch_size = size(X)[end]

                # TODO: use z ∈ R^{n × k}? with matrix Gaussian instead of multivariate Gaussian (see: Appendix C)

                # Generate "noise"
                μ_ϵ, logσ_ϵ = encoder(X, u)
                # min_σ_ϵ = 0.1f0
                # σ_ϵ = exp.(logσ_ϵ) .+ min_σ_ϵ
                σ_ϵ = 1.0f0

                # ϵ = μ_ϵ + (randn(Float32, size(μ_ϵ)) |> gpu) .* σ_ϵ

                # Causal Layer
                # Solve for μ_z = (I_A - Aᵀ)⁻¹ μ_ϵ
                C = I_A - transpose(A_)
                μ_z = C \ μ_ϵ
                σ_z = 1.0f0

                # Masking Layer
                # z_masked = transpose(A) * z + ϵ # Without mild non-linear function: zᵢ = Aᵀᵢz + ϵᵢ
                μ_z_masked = transpose(A_) * μ_z
                # With mild non-linear function: z_i = g_i(Ai .* z, η_i) + ϵ_i
                # TODO: also mask u?

                # TODO: apply g() to z and use attention?

                # ELBO
                # prior_μ, prior_logσ = prior_encoder(u)
                # min_prior_σ = 0.1f0
                # prior_σ = exp.(prior_logσ) .+ min_prior_σ
                prior_μ, prior_logσ = u, 0.0f0
                prior_σ = 1.0f0

                z_masked = μ_z_masked + (randn(Float32, size(μ_z_masked)) |> gpu) .* σ_z
                X̂ = decoder(z_masked)

                reconstruction_loss = logitbinarycrossentropy(sigmoid(X̂), X, agg=mean) / batch_size
                # kl_qϵ_pϵ = sum(logpdf_normal(ϵ, μ_ϵ, log.(σ_ϵ)) - logpdf_normal(ϵ, 0, 0)) / batch_size
                kl_qϵ_pϵ = kl_normal(μ_ϵ, σ_ϵ, 0, 1) / batch_size
                # kl_qz_pz = sum(logpdf_normal(z, u, z_logσ) - logpdf_normal(z, prior_μ_ϵ, log.(prior_σ))) / batch_size
                # kl_qz_pz = sum(logpdf_normal(z_masked, z, prior_logσ) - logpdf_normal(z, prior_μ, prior_logσ)) / batch_size
                kl_qz_pz = kl_normal(μ_z, σ_z, prior_μ, prior_σ) / batch_size

                ELBO = -reconstruction_loss - kl_qϵ_pϵ - kl_qz_pz
                NELBO = -ELBO

                # Learning Causal Structure
                l_u = norm(u - transpose(A_) * u)^2 / batch_size
                l_m = norm(z_masked - transpose(A_) * z_masked)^2 / batch_size     # (Without mild non-linear function)

                # Constrain A to be DAG
                H_A = H(A_)

                # Loss with regularization
                loss = NELBO + args.α * H_A + args.β * l_u + args.γ * l_m
                # @info "Loss" loss = loss
                return loss
            end
            with_logger(tb_logger) do
                @info "Loss" loss = loss
                @info "Loss" NELBO = NELBO l_u = l_u l_m = l_m H_A = H_A log_step_increment = 0

                @info "ELBO" ELBO = -NELBO reconstruction_loss = reconstruction_loss kl_qϵ_pϵ = kl_qϵ_pϵ kl_qz_pz = kl_qz_pz log_step_increment = 0
            end
            Flux.Optimise.update!(opt, parameters, grad)
        end
        # @info "Loss" loss = loss
        # with_logger(tb_logger) do
        #     @info "Loss" eval_loss = eval_loss log_step_increment = 0
        # end
        if args.visualize
            causal_graph_heatmap_plot = plot_causal_graph_heatmap(A |> cpu, concepts=concepts)
            causal_graph_plot = plot_causal_graph(A |> cpu, concepts=concepts)
            with_logger(tb_logger) do
                @info "Causal Graph" heatmap = causal_graph_heatmap_plot causal_graph = causal_graph_plot log_step_increment = 0
            end
        end
    end

    @info "Causal graph after train" A = A

    # Save model
    if !isdir(args.save_path)
        mkpath(args.save_path)
    end

    model_path = joinpath(args.save_path, "causalvae.bson")
    let encoder = encoder |> cpu, decoder = decoder |> cpu, prior_encoder = prior_encoder |> cpu, A = A |> cpu, args = struct2dict(args) |> cpu
        BSON.@save model_path encoder decoder prior_encoder A args
        @info "Model saved: $(model_path)"
    end
end

if isinteractive() || abspath(PROGRAM_FILE) == @__FILE__
    parsed_args = parse_args(args_settings, as_symbols=true)
    train(; parsed_args...)
end

using Flux
using Flux: @functor
using Flux.Losses: logitbinarycrossentropy
using Statistics
using LinearAlgebra

# Encoder
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

# iVAE
struct iVAE
    prior_encoder
    encoder
    decoder
end
@functor iVAE

function (vae::iVAE)(X, u)
    prior_μ, prior_logσ = vae.prior_encoder(u)
    μ, logσ = vae.encoder(X, u)

    # encoder_dist
    z = μ + (randn(Float32, size(logσ)) |> gpu) .* exp.(logσ)
    X̂ = vae.decoder(z)

    return X̂, μ, prior_μ, logσ, prior_logσ, z
end

function (vae::iVAE)(X)
    prior_μ, prior_logσ = vae.prior_encoder(u)
    μ, logσ = vae.encoder(X)

    # encoder_dist
    z = μ + (randn(Float32, size(logσ)) |> gpu) .* exp.(logσ)
    X̂ = vae.decoder(z)

    return X̂, μ, prior_μ, logσ, prior_logσ, z
end

function logpdf_normal(x, μ, logσ)
    return (@. -logσ - 0.5 * log(2π) - 0.5 * ((x - μ) / exp.(logσ))^2)
end

function elbo_loss(X̂, μ, prior_μ, logσ, prior_logσ, X, z)
    batch_size = size(X)[end]

    # decoder_dist
    # log_px_z = logpdf(MvNormal(reshape(X̂, :), decoder_σ), reshape(X, :)) / batch_size
    # encoder_dist
    # log_qz_xu = logpdf(MvNormal(reshape(μ, :), reshape(exp.(logσ), :)), reshape(z, :)) / batch_size
    log_qz_xu = sum(logpdf_normal(z, μ, logσ)) / batch_size
    # prior_dist
    # log_pz_u = logpdf(MvNormal(reshape(prior_μ, :), reshape(exp.(prior_logσ), :)), reshape(z, :)) / batch_size
    log_pz_u = sum(logpdf_normal(z, prior_μ, prior_logσ)) / batch_size

    reconstruction_loss = logitbinarycrossentropy(X̂, X, agg=sum) / batch_size

    kl_q_p = log_qz_xu - log_pz_u

    loss = reconstruction_loss + kl_q_p
    return loss
end

# VAE
iVAE(encoder, decoder) = iVAE((u) -> (zeros(args.latent_dim, size(u)[end]), ones(args.latent_dim, size(u)[end])), encoder, decoder)

# CausalVAE
struct CausalVAE
    prior_encoder
    encoder
    decoder
    A
end
@functor CausalVAE

function (vae::CausalVAE)(X, u, A_)
    I_A = Matrix{Float32}(I, size(A_)...) |> gpu

    # TODO: use z ∈ R^{n × k}? with matrix Gaussian instead of multivariate Gaussian (see: Appendix C)

    # Generate "noise"
    μ_ϵ, logσ_ϵ = vae.encoder(X, u)
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
    X̂ = vae.decoder(z_masked)

    return X̂, μ_ϵ, σ_ϵ, μ_z, σ_z, prior_μ, prior_σ, u, z_masked
end

"Element-wise KL divergence between two normal distributions"
function kl_normal(μ1, σ1, μ2, σ2)
    return sum((@. 0.5 * (2 * log(σ2) - 2 * log(σ1) + (σ1^2 + (μ1 - μ2)^2) / σ2^2 - 1)))
end

# c = 100 # "arbitrary postive number"
c = 2 # "arbitrary postive number"

# TODO: find different DAG constraint
H(A) = (n_dim = size(A)[1]; sum(diag((I + (c / n_dim) * (A .* A))^n_dim)) - n_dim)
# H(A) = (n_dim = size(A)[1]; sum(diag((I + (c / n_dim) * abs.(A))^n_dim)) - n_dim)

function elbo_loss_causal(X̂, X, μ_ϵ, σ_ϵ, μ_z, σ_z, prior_μ, prior_σ, u, A_, z_masked, args)
    batch_size = size(X)[end]

    reconstruction_loss = logitbinarycrossentropy(X̂, X, agg=mean) / batch_size
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
    # without u reconstruction
    # loss = NELBO + args.α * H_A + args.γ * l_m
    # @info "Loss" loss = loss
    return loss, NELBO, reconstruction_loss, kl_qϵ_pϵ, kl_qz_pz, l_u, l_m, H_A
end

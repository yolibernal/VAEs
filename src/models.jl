using Flux
using Flux: @functor
using Flux.Losses: logitbinarycrossentropy

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

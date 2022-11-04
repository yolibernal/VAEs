using Flux
using Flux: @functor
using Flux.Data: DataLoader
using Flux.Losses: logitbinarycrossentropy
using Parameters: @with_kw
using MLDatasets
using BSON
using DrWatson: struct2dict

@with_kw mutable struct Args
  η = 1e-3                # learning rate
  β = (0.90, 0.999)       # decay of momentums
  # λ = 0.01f0            # regularization paramater
  λ = 0.0f0               # regularization paramater
  batch_size = 128        # batch size
  sample_size = 10        # sampling size for output
  epochs = 20             # number of epochs
  seed = 0                # random seed
  input_dim = 28^2        # image size
  latent_dim = 2          # latent dimension
  hidden_dim = 500        # hidden dimension
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

  return DataLoader((X, y), batchsize=args.batch_size, shuffle=true)
end

# Encoder

struct Encoder
  encoder_features
  encoder_μ
  encoder_logσ
end
@functor Encoder

Encoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) = Encoder(
  # encoder_features
  Dense(input_dim, hidden_dim, relu),
  # encoder_μ
  Dense(hidden_dim, latent_dim),
  # encoder_logσ
  Dense(hidden_dim, latent_dim),
)

function (encoder::Encoder)(X)
  h = encoder.encoder_features(X)
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
  kl_q_p = -0.5f0 * sum(@. (1.0f0 + logσ - μ^2 - exp(logσ))) / batch_size

  # TODO: what is c? do we need it?
  # reconstruction_loss = (1 / 2c) * sum(norm(X - X̂)) / batch_size
  reconstruction_loss = logitbinarycrossentropy(X̂, X, agg=sum) / batch_size

  loss = kl_q_p + reconstruction_loss
  return loss
end

function train(; kws...)
  args = Args(; kws...)
  dataloader = get_dataloader(args, false)
  opt = AdamW(args.η, args.β, args.λ)

  encoder = Encoder(args.input_dim, args.latent_dim, args.hidden_dim)
  decoder = Decoder(args.input_dim, args.latent_dim, args.hidden_dim)

  # parameters = Flux.params(encoder, decoder)
  parameters = Flux.params(encoder.encoder_features, encoder.encoder_μ, encoder.encoder_logσ, decoder)

  @info "Starting training, $(args.epochs) epochs..."
  for epoch in 1:args.epochs
    @info "Epoch $(epoch)..."
    for (X, _) in dataloader
      grad = Flux.gradient(parameters) do
        μ, logσ = encoder(X)
        z = μ + randn(Float32, size(logσ)) .* exp.(logσ)
        X̂ = decoder(z)
        loss = vae_loss(X, X̂, μ, logσ)
      end
      Flux.Optimise.update!(opt, parameters, grad)
    end
  end

  # Save model
  model_path = joinpath(args.save_path, "model.bson")
  let encoder = encoder, decoder = decoder, args = struct2dict(args)
    BSON.@save model_path encoder decoder args
    @info "Model saved: $(model_path)"
  end
end

if abspath(PROGRAM_FILE) == @__FILE__
  train()
end

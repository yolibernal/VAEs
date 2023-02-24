using Flux
using BSON: @load
using DrWatson: struct2dict
using Flux

function generate_digits(args, prior_encoder, decoder; samples_per_class=1)
  y = repeat(0:9, samples_per_class)
  u = Flux.onehotbatch(y, 0:9)

  prior_μ, prior_logσ = prior_encoder(u)

  z = prior_μ .+ (randn(Float32, args[:latent_dim], args[:sample_size] * samples_per_class) |> gpu) .* exp.(prior_logσ)

  X̂ = decoder(z)
  X̂ = sigmoid.(X̂)
end

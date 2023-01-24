using Flux
using BSON: @load
using DrWatson: struct2dict
using Flux: chunk
using Images
using Plots

function convert_to_image(x, y_size)
  Gray.(permutedims(vcat(reshape.(chunk(x, y_size), 28, :)...), (2, 1)))
end

function generate_digits(prior_encoder, decoder, args)
  y = 0:9
  u = Flux.onehotbatch(y, 0:9)

  prior_μ, prior_logσ = prior_encoder(u)

  z = prior_μ .+ randn(args[:latent_dim], args[:sample_size]) .* exp.(prior_logσ)

  X̂ = decoder(z)
  X̂ = sigmoid.(X̂)
end

if abspath(PROGRAM_FILE) == @__FILE__
  include("vae.jl")
  @load "output/ivae.bson" encoder decoder prior_encoder args
  X̂ = generate_digits(prior_encoder, decoder, args)
  image = convert_to_image(X̂, args[:sample_size])
end

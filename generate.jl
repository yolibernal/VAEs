using Flux
using BSON: @load
using DrWatson: struct2dict
using Flux
using Images
using Plots

function convert_to_image(X; num_rows, num_columns)
  img_list = reshape.(chunk(X, num_columns * num_rows), 28, :)

  img_rows = map(chunk(img_list, num_rows)) do imgs
    permutedims(vcat(imgs...), (2, 1))
  end
  img_grid = Gray.(vcat(img_rows...))
  return img_grid
end

function generate_digits(args, prior_encoder, decoder; samples_per_class=1)
  y = repeat(0:9, samples_per_class)
  u = Flux.onehotbatch(y, 0:9)

  prior_μ, prior_logσ = prior_encoder(u)

  z = prior_μ .+ (randn(args[:latent_dim], args[:sample_size] * samples_per_class) |> gpu) .* exp.(prior_logσ)

  X̂ = decoder(z)
  X̂ = sigmoid.(X̂)
end

if abspath(PROGRAM_FILE) == @__FILE__
  include("vae.jl")
  @load "output/ivae.bson" encoder decoder prior_encoder args
  samples_per_class = 5
  num_classes = 10

  X̂ = generate_digits(args, prior_encoder, decoder, samples_per_class=samples_per_class)
  image = convert_to_image(X̂, num_rows=samples_per_class, num_columns=num_classes)
end

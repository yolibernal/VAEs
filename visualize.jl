using StatsPlots
using BSON: @load
using LinearAlgebra

function visualize_priors_2d(prior_μ, prior_logσ, labels=nothing)
    if size(prior_μ, 1) != 2
        error("Can only visualize 2D priors")
    end

    plot()
    for i in 1:size(prior_μ)[end]
        μ = prior_μ[:, i]
        σ² = exp.(2 .* prior_logσ[:, i])
        Σ = Diagonal(σ²)
        covellipse!(μ, Σ, label=labels !== nothing ? labels[i] : nothing)
    end
    plot!()
end

if abspath(PROGRAM_FILE) == @__FILE__
    include("vae.jl")
    @load "output/ivae.bson" encoder decoder prior_encoder args
    y = 0:9
    u = Flux.onehotbatch(y, 0:9)

    prior_μ, prior_logσ = prior_encoder(u)
    visualize_priors_2d(prior_μ, prior_logσ, y)
end

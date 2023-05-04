using StatsPlots
using BSON: @load
using LinearAlgebra
using Flux: chunk, gpu
using LaTeXStrings
using Images
using GraphRecipes

function visualize_priors_2d(prior_μ, prior_logσ, labels=nothing)
    if size(prior_μ, 1) != 2
        error("Can only visualize 2D priors")
    end

    plot(title=L"Mean and variance of prior distribution $p(z|u)$")
    for i in 1:size(prior_μ)[end]
        μ = prior_μ[:, i]
        σ² = exp.(2 .* prior_logσ[:, i])
        Σ = Diagonal(σ²)
        covellipse!(μ, Σ, label=labels !== nothing ? labels[i] : nothing)
        annotate!(μ[1], μ[2], text(labels[i], 10, :hcenter, :vcenter))
    end
    plot!()
end


function visualize_prior_mean_history_2d(prior_μ_history, prior_logσ=nothing)
    prior_μ_history = permutedims(cat(prior_μ_history..., dims=3), (3, 1, 2))

    if size(prior_μ_history, 2) != 2
        error("Can only visualize 2D priors")
    end

    classes = 0:9

    plot(title=L"Change of prior $p(z|u)$ over time")
    for c in classes
        plot!(prior_μ_history[:, 1, c+1], prior_μ_history[:, 2, c+1], label=c, color=c)

        if prior_logσ !== nothing
            μ = prior_μ_history[end, :, c+1]
            σ² = exp.(2 .* prior_logσ[:, c+1])
            Σ = Diagonal(σ²)
            covellipse!(μ, Σ, label=false, color=c, alpha=0.3)
        end
    end
    plot!()
end

function visualize_latent_space(args, dataloader, encoder)
    # TODO: use embedding for higher dim latent space

    X, y, u = map(zip(collect(dataloader)...)) do A
        return hcat(A...)
    end
    if args[:ivae]
        μ, logσ = encoder(X, u)
    else
        μ, logσ = encoder(X)
    end

    classes = 0:9
    labels = Flux.onecold(u, 0:9)

    μ = μ |> cpu
    labels = labels |> cpu

    # TODO: the way the points are plotted makes later classes overlap earlier ones which makes crowded areas seem too homogenous
    plot(title=L"Means of $p(z|X, u)$")
    for c in classes
        scatter!(μ[1, labels.==c], μ[2, labels.==c], label=c)
    end

    # plot()
    # scatter!(μ[1, :], μ[2, :], color=labels, groups=labels)
    # scatter(μs[1, :], μs[2, :], color=labels, labels=reshape(Flux.onecold(u, 0:9), 1, :))
    # scatter!(μ[1, :], μ[2, :], mc=labels)
    # scatter!(μ[1, :], μ[2, :], groups=labels)
    # scatter!(μ[1, :], μ[2, :], color=labels)
    # scatter(μ[1, :], μ[2, :], groups=labels, z_order=shuffle(1:size(X)[end]))
    # scatter(μ[1, :], μ[2, :], mc=labels, ma=0.7)

    plot!()
end

function plot_causal_graph_heatmap(A; concepts)
    maxval = maximum(abs.(A))
    heatmap(concepts,
        concepts,
        A,
        title="Concepts",
        c=cgrad([:blue, :white, :red]),
        clims=(-maxval, maxval))
end

function plot_causal_graph(A; concepts)
    graphplot(A, names=concepts, edgelabel=round.(A, digits=2), edgewidth=abs.(A), markersize=0.05, self_edge_size=0.2, nodeshape=:circle)
end

function convert_to_image(X; num_rows, img_dim=28, num_channels=1)
    imgs = eachslice(X, dims=2)
    img_list = num_channels == 1 ?
               permutedims.(reshape.(imgs, img_dim, img_dim), ((2, 1),)) :
               reshape.(imgs, num_channels, img_dim, img_dim)
    if num_channels == 1
        img_list = colorview.(Gray, (img_list))
    elseif num_channels == 3
        img_list = colorview.(RGB, (img_list))
    elseif num_channels == 4
        img_list = colorview.(RGBA, (img_list))
    end
    img_rows = map(chunk(img_list, num_rows)) do imgs
        hcat(imgs...)
    end
    img_grid = vcat(img_rows...)
    return img_grid
end

module VAEs

include("generate.jl")
include("visualize.jl")
include("models.jl")

export Encoder, Decoder, iVAE, elbo_loss
export CausalVAE, elbo_loss_causal, H
export generate_digits
export visualize_priors_2d, visualize_prior_mean_history_2d, visualize_latent_space, plot_causal_graph_heatmap, plot_causal_graph, convert_to_image

end

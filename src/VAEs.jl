module VAEs

include("generate.jl")
include("visualize.jl")
include("models.jl")

export Encoder, Decoder, iVAE, elbo_loss
export generate_digits
export visualize_priors_2d, visualize_prior_mean_history_2d, visualize_latent_space, convert_to_image

end

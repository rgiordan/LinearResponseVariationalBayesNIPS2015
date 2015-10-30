# This is only designed for Julia 0.3
@assert VERSION < v"0.4.0-dev"

using Distributions
using DataFrames
using JuMP

import DataFramesIO
import JSON

# The environment variable GIT_REPO_LOC must be set to the place where you
# cloned the repository.
library_location =
	joinpath(ENV["GIT_REPO_LOC"],
				 	 "LinearResponseVariationalBayesNIPS2015/linear_regression/")

data_path = joinpath(library_location, "data")

include(joinpath(library_location, "random_effects_regression_lib.jl"))

# Objects to export to R.
function build_dict(objs)
	# objs should be an array of symbols.
	result = Dict()
	for obj in objs
		result[string(obj)] = eval(obj)
	end
	result
end

# The list of symbols to export to R.
export_symbols = [:df_json,
               	  :true_nu, :true_beta, :true_tau, :true_gamma,
               	  :vb_nu, :vb_beta, :vb_tau, :vb_gamma, :vb_log_tau, :vb_log_nu,
               	  :mfvb_cov_beta, :lrvb_cov_beta,
               	  :mfvb_var_tau, :lrvb_var_tau,
               	  :mfvb_var_nu, :lrvb_var_nu,
               	  :mfvb_var_log_tau, :lrvb_var_log_tau,
               	  :mfvb_var_log_nu, :lrvb_var_log_nu,
               	  :mfvb_var_gamma, :lrvb_var_gamma,
               	  :mfvb_cov_main, :lrvb_cov_main,
                  :beta_prior_info_scale, :tau_prior_alpha, :tau_prior_gamma,
                  :nu_prior_alpha, :nu_prior_gamma,
                  :n, :k_tot, :z_sd ];

# Because of how eval() only works in the global scope,
# these variables must be declared globally for this to work.
for symb in export_symbols
	@eval $symb = 0
end

# Generate data from a model.
function rand_bound(lower, upper)
	u = rand()
	u * (upper - lower) + lower
end


#############################
# Run the simulations

analysis_name = "re_regression_full_cov"

save_full_cov = true
export_data = true

k_tot = 2
re_num = 30
re_obs_num = 10
re_ind = collect(repmat(1:re_num, 1, re_obs_num)');
n = re_num * re_obs_num
x = reshape(rand(Normal(0, 1), n * k_tot), n, k_tot);
z_sd = 0.4
z = x[:, 1] + rand(Normal(0, z_sd), n);

# Priors
beta_prior_info_scale = 0.1
beta_prior_info = beta_prior_info_scale * eye(k_tot)
beta_prior_mean = zeros(k_tot)
tau_prior_alpha = 2.0
tau_prior_gamma = 2.0
nu_prior_alpha = 2.0
nu_prior_gamma = 2.0

n_sims = 100

for sim_id = 1:n_sims

	println("------------------------------")
	println("---- sim $sim_id ----------->>>>>")
	println("------------------------------")

	true_nu = rand_bound(0.5, 4.0)
	true_beta = Float64[ rand_bound(0.0, 3.0) for k=1:k_tot ]
	true_gamma = rand(Normal(0, 1 / sqrt(true_nu)), re_num)

	true_tau = rand_bound(0.2, 1.0)

	row_offset = Float64[ dot(x[i, :][:], true_beta) + true_gamma[re_ind[i]] * z[i] for i=1:n];
	y = Float64[ rand(Normal(row_offset[i], 1 / sqrt(true_tau))) for i=1:n ];

	priors = Priors(beta_prior_mean, beta_prior_info,
	                tau_prior_alpha, tau_prior_gamma,
		              nu_prior_alpha, nu_prior_gamma)

	# Unfortunately, you can't see to change y, x, and z in vb_reg and have
	# them be re-used in a new regression.  So we pay the model building price
	# for every iteration.
	vb_reg = VBRandomEffectsRegression(x, z, y, re_ind, priors);

	################################
	# Set initial values and fit.
	set_beta(true_beta, vb_reg; beta2_eps = 0.0)
	for m=1:re_num
		set_gamma(true_gamma[m], m, vb_reg, gamma2_eps=0.0)
	end
	set_tau(true_tau, vb_reg)
	set_nu(true_nu, vb_reg)

	fit_model(vb_reg, 1000, 1e-9)


	################################
	# Save results
	vb_beta = get_e_beta(vb_reg)
	vb_tau = getValue(vb_reg.e_tau)
	vb_nu = getValue(vb_reg.e_nu)
	vb_log_tau = getValue(vb_reg.e_log_tau)
	vb_log_nu = getValue(vb_reg.e_log_nu)
	vb_gamma = get_e_gamma(vb_reg);

	beta_i = Int64[ vb_reg.e_beta[k].col for k=1:k_tot ]
	ud_tot = (vb_reg.k_tot + 1) * vb_reg.k_tot / 2
	beta2_i = Int64[ vb_reg.e_beta2_ud[k].col for k=1:ud_tot ]

	tau_i = vb_reg.e_tau.col
	log_tau_i = vb_reg.e_log_tau.col
	nu_i = vb_reg.e_nu.col
	log_nu_i = vb_reg.e_log_nu.col

	gamma_i = Int64[ vb_reg.e_gamma[m].col for m=1:re_num ]
	gamma2_i = Int64[ vb_reg.e_gamma2[m].col for m=1:re_num ]

	mfvb_cov = get_variational_covariance(vb_reg);
	lrvb_cov = get_lrvb_cov(vb_reg);

	mfvb_diag = diag(mfvb_cov);
	lrvb_diag = diag(lrvb_cov);

	mfvb_cov_beta = full(mfvb_cov[beta_i, beta_i])
	lrvb_cov_beta = full(lrvb_cov[beta_i, beta_i])

	mfvb_var_tau = mfvb_diag[tau_i]
	lrvb_var_tau = lrvb_diag[tau_i]

	mfvb_var_log_tau = mfvb_diag[log_tau_i]
	lrvb_var_log_tau = lrvb_diag[log_tau_i]

	mfvb_var_nu = mfvb_diag[nu_i]
	lrvb_var_nu = lrvb_diag[nu_i]

	mfvb_var_log_nu = mfvb_diag[log_nu_i]
	lrvb_var_log_nu = lrvb_diag[log_nu_i]

	mfvb_var_gamma = mfvb_diag[gamma_i];
	lrvb_var_gamma = lrvb_diag[gamma_i];

	main_i = setdiff(1:length(vb_reg.m.colVal), [gamma_i, gamma2_i, beta2_i])
	mfvb_cov_main = full(mfvb_cov[main_i, main_i]);
	lrvb_cov_main = full(lrvb_cov[main_i, main_i]);

	if export_data
		# Save the data in JSON so R can do its thing.

		filename = joinpath(data_path, "$(analysis_name)_$(sim_id).json")
		println("Saving data for $(sim_id) in $filename.")

		df = DataFrame(z=z, y=y, re_ind=re_ind);
		for i=1:size(x, 2)
			df[symbol(string("x", i))] = x[:, i]
		end
		df_json = DataFramesIO.df2json(df);

		analysis_dat = build_dict(export_symbols);

		f = open(filename, "w")
		write(f, JSON.json(analysis_dat))
		close(f)
	end
end

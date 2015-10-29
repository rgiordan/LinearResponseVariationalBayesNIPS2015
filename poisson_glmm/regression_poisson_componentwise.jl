
install_packages = false
if install_packages
	# Pkg.available() doens't pick up cloned packages.
	available_packages = keys(Pkg.installed())
	required_packages =
		["DataFrames", "Distributions", "JSON",
		 "JuMP", "Ipopt", "ReverseDiffSparse"]
	for package in setdiff(required_packages, available_packages)
		Pkg.add(package)
	end

	# This package hasn't been released yet.
	if !("DataFramesIO" in available_packages)
		Pkg.clone("https://github.com/johnmyleswhite/DataFramesIO.jl")
	end
end

#using JuMP
#using ReverseDiffSparse

using DataFrames
import Distributions
import JSON
import DataFramesIO
#import PyPlot

# The environment variable GIT_REPO_LOC must be set to the place where you
# cloned the repository.
library_location =
	joinpath(ENV["GIT_REPO_LOC"],
	         "LinearResponseVariationalBayesNIPS2015/poisson_glmm/")
include(joinpath(library_location, "regression_poisson_componentwise_lib.jl"))
include(joinpath(library_location, "regression_poisson_loglik_lib.jl"))
data_path = joinpath(library_location, "data")

#################################################################
# Functions to build a dictionary of symbols for export to R.
function build_dict(objs)
	# objs should be an array of symbols.
	result = Dict()
	for obj in objs
		result[string(obj)] = eval(obj)
	end
	result
end

# The list of symbols to export to R.
export_symbols = [:df_json, :n, :x_mean, :true_mu, :true_epsilon,
	               :sim_id, :mu_lrvb_var, :mu_mfvb_var,
								 :z_lrvb_var, :z_mfvb_var,
	               :tau_lrvb_var, :tau_mfvb_var,
								 :log_tau_lrvb_var, :log_tau_mfvb_var,
	               :e_mu_val, :e_tau_val, :e_log_tau_val, :e_z_val,
	               :mu_prior_var, :tau_prior_alpha, :tau_prior_beta,
	               :mu_cols_i, :tau_cols_i, :z_i, :z2_i,
	               :lrvb_cov_thetaz, :theta_lrvb_cov, :theta_mfvb_cov ]

# Because of how eval() only works in the global scope,
# these variables must be declared globally for this to work.
# There is a fix with @debug but in the meantime this works.  See
# http://stackoverflow.com/questions/30552637/populating-a-julia-dictionary-with-an-array-of-symbols
for symb in export_symbols
	@eval $symb = 0
end


###############
# Analysis

# Generate a uniform random number between two bounds.
function rand_bound(lower, upper)
	u = rand()
	u * (upper - lower) + lower
end

# The name is used in the filename that saves the data.
analysis_name = "poisson_glmm_z_theta_cov"

# Number of data points.
n = 500

# Whether or not to save the results in JSON format.
export_data = true

# Whether or not to calculate the LRVB covariance for the z variables.
calculate_z_var = true

# The true values of x.
x_mean = 0.0
x_sigma = 1.0 # The variance of the regressors
x_vec = rand(Distributions.Normal(x_mean, x_sigma), n);

# The priors.
mu_prior_var = 10.;
tau_prior_alpha = 1.;
tau_prior_beta = 1.;
prior = Prior(mu_prior_var, tau_prior_alpha, tau_prior_beta)

# Run n_sims simulations starting at sim_offset and fit LRVB estimates for each.
n_sims = 100
sim_offset = 1
for sim_id = sim_offset:(sim_offset + n_sims - 1)
	println("------------------------------")
	println("         SIM $sim_id")
	println("------------------------------")

	# These are set to a regime where MCMC and MFVB have reasonable correspondence
	# in their means, but in which the MFVB covariances are bad.

	# tau = 9 is about where the correspondence between MFVB and MCMC breaks
	# down with n = 500.
	true_mu = rand_bound(0.0, 3.0)
	true_epsilon = rand_bound(0.3, 1.0)

	true_z =
		[ rand(Distributions.Normal(true_mu * x_vec[i], true_epsilon)) for i=1:n ];
	y_vec = [ rand(Distributions.Poisson(exp(z_i))) for z_i in true_z ];

	println("Fitting the model.")
	start_z_val = true_z;
	start_z2_val = true_z .^ 2 + 0.1 * ones(n);
	start_mu_val = true_mu;
	start_mu2_val = true_mu ^ 2 + 0.01;
	start_tau_val = 1 / true_epsilon^2;
	start_log_tau_val = log(start_tau_val);

	e_mu_val, e_mu2_val, e_tau_val, e_log_tau_val, e_z_val, e_z2_val =
		fit_model!(x_vec, y_vec,
			       start_mu_val, start_mu2_val, start_tau_val, start_log_tau_val,
			       start_z_val, start_z2_val, prior;
				   max_iter=50, tol=1e-13)

	[ true_mu, e_mu_val ]
	[ 1 / true_epsilon^2, e_tau_val ]
	sqrt(e_mu2_val - e_mu_val^2)

	println("Getting LRVB correction.")
	m_np = NormalPoissonModel(x_vec, y_vec, prior);

	# Set the values to the optimum found above.
	for i=1:n
		setValue(m_np.e_z[i], e_z_val[i])
		setValue(m_np.e_z2[i], e_z2_val[i])
	end
	setValue(m_np.e_mu, e_mu_val)
	setValue(m_np.e_mu2, e_mu2_val)
	setValue(m_np.e_tau, e_tau_val)
	setValue(m_np.e_log_tau, e_log_tau_val)

	ll_hess = full(get_loglik_hess(m_np));

	q_cov = get_variational_cov(m_np);
	q_cov_full = full(q_cov);

	# Get LRVB.
	vb_corr_term = eye(m_np.n_params) - q_cov * ll_hess;

	lrvb_cov = NaN;
	z_lrvb_var = NaN;
	z_mfvb_var = NaN;

	mu_cols_i = Int64[ m_np.e_mu.col, m_np.e_mu2.col ]
	tau_cols_i = Int64[ m_np.e_tau.col, m_np.e_log_tau.col ]
	z_i = Int64[ m_np.e_z[i].col for i=1:n ]
	z2_i = Int64[ m_np.e_z2[i].col for i=1:n ]
	z_cols_i = vcat(z_i, z2_i);
	theta_cols_i = setdiff(1:m_np.n_params, z_cols_i)
	if calculate_z_var
		# Get the full correction.  With small models this is fast enough.
		lrvb_cov = vb_corr_term \ full(q_cov);
		z_lrvb_var = diag(lrvb_cov)[z_i];
		z_mfvb_var = diag(q_cov)[z_i];
		lrvb_cov_thetaz = lrvb_cov[theta_cols_i, z_i];
	else
		# Only calculate for mu and tau
		main_i = setdiff(1:m_np.n_params, z_cols_i)
		main_corr_term =
			vb_corr_term[main_i, z_cols_i] *
			(vb_corr_term[z_cols_i, z_cols_i] \ vb_corr_term[z_cols_i, main_i])
		vb_corr_main = vb_corr_term[main_i, main_i] - main_corr_term
		lrvb_cov = full(vb_corr_main \ full(q_cov[main_i, main_i]));
		lrvb_cov_thetaz = 0.;
		mfvb_cov_thetaz = 0.;
	end

	theta_lrvb_cov = lrvb_cov[theta_cols_i, theta_cols_i];
	theta_mfvb_cov = q_cov_full[theta_cols_i, theta_cols_i];

	mu_col = m_np.e_mu.col
	mu_lrvb_var = lrvb_cov[mu_col, mu_col];
	mu_mfvb_var = q_cov[mu_col, mu_col];

	tau_col = m_np.e_tau.col
	tau_lrvb_var = lrvb_cov[tau_col, tau_col];
	tau_mfvb_var = q_cov[tau_col, tau_col];

	log_tau_col = m_np.e_log_tau.col
	log_tau_lrvb_var = lrvb_cov[log_tau_col, log_tau_col];
	log_tau_mfvb_var = q_cov[log_tau_col, log_tau_col];

	println(e_mu_val, " ", sqrt(mu_lrvb_var), " ", sqrt(mu_mfvb_var))
	println(e_tau_val, " ", sqrt(tau_lrvb_var), " ", sqrt(tau_mfvb_var))

	if export_data
		# Save the data in JSON so R can do its thing.

		filename = joinpath(data_path, "$(analysis_name)_$(sim_id).json")
		println("Saving data for $sim_id in $filename.")

		mu_prior_var = prior.mu_prior_var
		tau_prior_alpha = prior.tau_prior_alpha
		tau_prior_beta = prior.tau_prior_beta

		df_json = DataFramesIO.df2json(DataFrame(y=y_vec, x=x_vec, z=true_z));

		analysis_dat = build_dict(export_symbols);

		# Test read into R, modifying, and returning:
		println(filename)

		f = open(filename, "w")
		write(f, JSON.json(analysis_dat))
		close(f)
	end
end

println("Done.")
# Now run analyze_results.R.

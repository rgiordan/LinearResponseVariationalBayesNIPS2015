
using JuMP
using ReverseDiffSparse

# Define a model containing the full log likelihood of the normal poisson model.

type Prior
	mu_prior_var::Float64;
	tau_prior_alpha::Float64;
	tau_prior_beta::Float64;
end

type ModelFunctions
	m_eval::JuMP.JuMPNLPEvaluator

	# Stuff for the hessian
	hess_struct::(Array{Int64,1},Array{Int64,1})
	hess_vec::Array{Float64,1}
	numconstr::Int64

	# An in-place gradient
	grad::Array{Float64, 1}

	ModelFunctions(m_full::Model) = begin
		m_const_mat = JuMP.prepConstrMatrix(m_full);
		m_eval = JuMP.JuMPNLPEvaluator(m_full, m_const_mat);
		MathProgBase.initialize(m_eval, [:ExprGraph, :Grad, :Hess])

		# Structures for the hessian.
		hess_struct = MathProgBase.hesslag_structure(m_eval);
		hess_vec = zeros(length(hess_struct[1]));
		numconstr = (length(m_eval.m.linconstr) +
			         length(m_eval.m.quadconstr) +
			         length(m_eval.m.nlpdata.nlconstr))

		grad = zeros(length(m_full.colVal))
		new(m_eval, hess_struct, hess_vec, numconstr, grad)
	end
end

type NormalPoissonModel
	m_full::Model
	m_func::ModelFunctions
	n::Int64
	n_params::Int64
	prior::Prior

	x_vec::Array{Float64, 1}
	y_vec::Array{Float64, 1}

	e_mu::Any
	e_mu2::Any
	e_tau::Any
	e_log_tau::Any
	e_z::Any
	e_z2::Any

	NormalPoissonModel(x_vec::Array{Float64, 1}, y_vec::Array{Int64, 1}, prior::Prior) = begin
		# Question: to what are the expressions x_vec and y_vec bound here?
		n = length(x_vec)
		@assert length(x_vec) == length(y_vec)

		m_full = Model()

		# The global variational parameters:
		@defVar(m_full, e_mu);
		@defVar(m_full, e_mu2);
		@defVar(m_full, e_tau);
		@defVar(m_full, e_log_tau);

		# The variational parameters of z:
		@defVar(m_full, e_z[i=1:n]);
		@defVar(m_full, e_z2[i=1:n]);

		# Define the log likelihood.

		# Define some convenience expressions:
		@defNLExpr(var_mu, e_mu2 - e_mu ^ 2);
		@defNLExpr(var_z[i=1:n], e_z2[i] - e_z[i] ^ 2);
		@defNLExpr(e_exp_z[i=1:n], exp(e_z[i] + 0.5 * var_z[i]));

		# Define the elbo.
		@defNLExpr(loglik_z[i=1:n],
				   e_tau * (-0.5 * e_z2[i] +
				            e_z[i] * e_mu * x_vec[i] - 
				            0.5 * e_mu2 * (x_vec[i] ^ 2)) +
				   0.5 * e_log_tau -
				   e_exp_z[i] + e_z[i] * y_vec[i]);

		# The priors.
		@defNLExpr(mu_prior, -0.5 * e_mu2 / prior.mu_prior_var)
		@defNLExpr(tau_prior, -prior.tau_prior_beta * e_tau + (prior.tau_prior_alpha - 1) * e_log_tau)

		@defNLExpr(loglik, sum{loglik_z[i], i=1:n} + mu_prior + tau_prior);

		@setNLObjective(m_full, Max, loglik)

		m_func = ModelFunctions(m_full)
		n_params = length(m_full.colVal)
		
		new(m_full, m_func, n, n_params, prior, x_vec, y_vec, e_mu, e_mu2, e_tau, e_log_tau, e_z, e_z2)
	end
end



function get_model_vals(m_np::NormalPoissonModel)
	(getValue(m_np.e_mu), 	getValue(m_np.e_mu2),
	 getValue(m_np.e_tau), getValue(m_np.e_log_tau),
	 [ getValue(m_np.e_z[i]) for i = 1:m_np.n],
	 [ getValue(m_np.e_z2[i]) for i = 1:m_np.n])
end


function get_loglik_hess(m_np::NormalPoissonModel)
	m_func = m_np.m_func
	n_params = m_np.n_params
	MathProgBase.eval_hesslag(m_func.m_eval, m_func.hess_vec,
		                      m_np.m_full.colVal, 1.0, zeros(m_func.numconstr))
	this_hess_ld = sparse(m_func.hess_struct[1], m_func.hess_struct[2],
		                  m_func.hess_vec, n_params, n_params)
	this_hess_ld + this_hess_ld' - sparse(diagm(diag(this_hess_ld)))
end


function get_tau_variational_covariance(m_np::NormalPoissonModel)
	# Return an array of triplets (i, j, q_cov[i, j]) that can
	# be used to populate a sparse matrix representing the variational
	# covariance for the tau parameters.

	e_mu_val, e_mu2_val, e_tau_val, e_log_tau_val, e_z_val, e_z2_val =
		get_model_vals(m_np)

	n = m_np.n

	zx_sum = dot(e_z_val, m_np.x_vec)
	xx_sum = dot(m_np.x_vec, m_np.x_vec)
	zz_sum = sum(e_z2_val)

	tau_alpha, tau_beta = get_tau_params(e_mu_val, e_mu2_val,
		                                 zx_sum, zz_sum, xx_sum, n,
		                                 m_np.prior.tau_prior_alpha,
		                                 m_np.prior.tau_prior_beta)

	e_tau_col = m_np.e_tau.col
	e_log_tau_col = m_np.e_log_tau.col

	tau_cov = (Int64, Int64, Float64)[]
	push!(tau_cov, (e_tau_col,     e_tau_col,     tau_alpha / (tau_beta ^ 2)))
	push!(tau_cov, (e_log_tau_col, e_log_tau_col, trigamma(tau_alpha)))
	push!(tau_cov, (e_tau_col,     e_log_tau_col, 1 / tau_beta))
	push!(tau_cov, (e_log_tau_col, e_tau_col,     1 / tau_beta))
	tau_cov
end


function get_normal_variational_covariance(e_norm, e_norm2, e_col, e2_col)
	# Get the normal covariance for a scalar normal with expectation
	# e_norm, expecation of the square e_norm2, and in columns
	# e_col and e2_col respectively.

	norm_cov = (Int64, Int64, Float64)[]

	norm_var = e_norm2 - e_norm ^ 2
	# Get the linear term variance
	push!(norm_cov, (e_col, e_col, norm_var))

	# Get the covariance between the linear and quadratic terms.
	this_cov = 2 * e_norm * norm_var
	push!(norm_cov, (e_col, e2_col, this_cov))
	push!(norm_cov, (e2_col, e_col, this_cov))			

	# Get the covariance between the quadratic terms.
	this_cov = 2 * norm_var ^ 2 + 4 * norm_var * (e_norm ^ 2)
	push!(norm_cov, (e2_col, e2_col, this_cov))

	norm_cov
end


function sparse_mat_from_tuples(tup_array)
	sparse(Int64[x[1] for x=tup_array],
		   Int64[x[2] for x=tup_array],
		   Float64[x[3] for x=tup_array])
end


function get_variational_cov(m_np::NormalPoissonModel)
	# Variational covariance matrix
	m_full = m_np.m_full
	e_mu_val, e_mu2_val, e_tau_val, e_log_tau_val, e_z_val, e_z2_val = get_model_vals(m_np)

	q_cov_tuples = (Int64, Int64, Float64)[]
	append!(q_cov_tuples, get_tau_variational_covariance(m_np))
	append!(q_cov_tuples, get_normal_variational_covariance(e_mu_val, e_mu2_val,
		m_np.e_mu.col, m_np.e_mu2.col))
	for i=1:n
		append!(q_cov_tuples,
			    get_normal_variational_covariance(e_z_val[i], e_z2_val[i],
					m_np.e_z[i].col, m_np.e_z2[i].col))
	end
	q_cov = sparse_mat_from_tuples(q_cov_tuples);
end




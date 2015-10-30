using JuMP
using Distributions
using MathProgBase

type Priors
	beta_prior_mean::Array{Float64, 1}
	beta_prior_info::Array{Float64, 2}
	tau_prior_alpha::Float64
	tau_prior_gamma::Float64
	nu_prior_alpha::Float64
	nu_prior_gamma::Float64
end


type VBRandomEffectsRegression
	m::Model
	m_eval::JuMP.JuMPNLPEvaluator
	e_beta::Any  # Is there a stronger type I can use?
	e_beta2_ud::Any

	e_tau::Any
	e_log_tau::Any

	# The random effects' location and varaince.
	# (For now assume their covariance is a multiple of the identity.)
	e_gamma::Any
	e_gamma2::Any

	e_nu::Any
	e_log_nu::Any

	x::Array{Float64, 2}
	z::Array{Float64, 1}
	y::Array{Float64, 1}
	priors::Priors

	n::Int64
	k_tot::Int64
	re_num::Int64
	beta2_ind::Array{Int64, 2}

	# Stuff for the hessian
	hess_struct::(Array{Int64,1},Array{Int64,1})
	hess_vec::Array{Float64,1}
	numconstr::Int64

	# An in-place gradient
	grad::Array{Float64, 1}

	VBRandomEffectsRegression(x::Array{Float64, 2}, z::Array{Float64, 1}, y::Array{Float64, 1},
		                      re_ind::Array{Int64, 1}, priors::Priors) = begin
		# The model is y_i = x_i beta + z_i gamma_re_ind[i] + epsilon_i
		# Currently only scalar z_i are supported.

		n = size(x)[1]
		k_tot = size(x)[2]
		k_ud = int(k_tot * (k_tot + 1) / 2)
		re_num = maximum(re_ind)

		# Make an array for indexing into the e_beta2_ud variable.
		function make_ud_index_matrix(k)
			ud_mat = Array(Int64, (k, k))
			for k1=1:k, k2=1:k
				ud_mat[k1, k2] =
					(k1 <= k2 ? (k1 + (k2 - 1) * k2 / 2) :
						        (k2 + (k1 - 1) * k1 / 2))
			end
			ud_mat
		end
		beta2_ind = make_ud_index_matrix(k_tot)

		@assert size(x)[1] == size(y)[1] == size(z)[1] == size(re_ind)[1]

		m = Model()

		# Fixed effects:
		@defVar(m, e_beta[1:k_tot])
		@defVar(m, e_beta2_ud[1:k_ud])

		# Random effects:
		@defVar(m, e_gamma[1:re_num])
		@defVar(m, e_gamma2[1:re_num])

		# Residual information:
		@defVar(m, e_tau);
		@defVar(m, e_log_tau);

		# Random effect information:
		@defVar(m, e_nu);
		@defVar(m, e_log_nu);

		# Define the elbo.

		# First, convenience expressions for inner products:
		@defNLExpr(xt_beta[i=1:n], sum{e_beta[k] * x[i, k], k=1:k_tot});

		@defNLExpr(e_beta2[k1=1:k_tot, k2=1:k_tot], e_beta2_ud[beta2_ind[k1, k2]])

		@defNLExpr(trace_x_beta[i=1:n],
			       sum{x[i, k1] * x[i, k2] * e_beta2[k1, k2],
			           k1=1:k_tot, k2=1:k_tot})

		# The actual likelihood.
		# TODO: use sufficient statistics?
		@defNLExpr(loglik_obs[i=1:n],
					-0.5 * e_tau * ((y[i] ^ 2) -
						             2 * y[i] * (xt_beta[i] + z[i] * e_gamma[re_ind[i]]) +
						             (trace_x_beta[i] +
						              (z[i] ^ 2) * e_gamma2[re_ind[i]] +
						              2 * xt_beta[i] * z[i] * e_gamma[re_ind[i]])) +
					0.5 * e_log_tau);

		# The random effect likelihoods.
		@defNLExpr(re_obs[m=1:re_num], -0.5 * e_nu * e_gamma2[m] + 0.5 * e_log_nu)

		# The priors:
		@defNLExpr(beta_prior,
					-0.5 * sum{priors.beta_prior_info[k1, k2] * e_beta2[k1, k2],
					           k1=1:k_tot, k2=1:k_tot} +
					sum{priors.beta_prior_info[k1, k2] * e_beta[k1] * priors.beta_prior_mean[k2],
					    k1=1:k_tot, k2=1:k_tot});
		@defNLExpr(tau_prior,
			(priors.tau_prior_alpha - 1) * e_log_tau - priors.tau_prior_gamma * e_tau);
		@defNLExpr(nu_prior,
			(priors.nu_prior_alpha - 1) * e_log_nu - priors.nu_prior_gamma * e_nu);

		# Sum over data points.
		@defNLExpr(loglik, sum{loglik_obs[i], i=1:n} + sum{re_obs[m], m=1:re_num} +
			               beta_prior + tau_prior + nu_prior);

		########################
		# This is not the actual objective, but the entropy is not easily expressible in
		# JuMP, so we will just use the "objective" for its derivatives.
		@setNLObjective(m, Max, loglik)

		# Set up the evaluator object.  This can be slow.
		println("Setting up the evaulator object.  (This parses the model and can be slow.)")
		m_const_mat = JuMP.prepConstrMatrix(m);
		m_eval = JuMP.JuMPNLPEvaluator(m);
		MathProgBase.initialize(m_eval, [:ExprGraph, :Grad, :Hess])

		# Structures for the hessian.
		hess_struct = MathProgBase.hesslag_structure(m_eval);
		hess_vec = zeros(length(hess_struct[1]));
		numconstr = (length(m_eval.m.linconstr) +
			         length(m_eval.m.quadconstr) +
			         length(m_eval.m.nlpdata.nlconstr))

		grad = zeros(length(m.colVal))

		new(m, m_eval, e_beta, e_beta2_ud, e_tau, e_log_tau,
			e_gamma, e_gamma2, e_nu, e_log_nu,
			x, z, y, priors, n, k_tot, re_num, beta2_ind,
			hess_struct, hess_vec, numconstr, grad)
	end
end



##############
function set_beta(beta::Array{Float64, 1}, vb_reg::VBRandomEffectsRegression; beta2_eps=1.0)
	# Set the model e_beta and e_beta2, with eps extra
	# variance added to beta2 to avoid singularity.
	for k=1:vb_reg.k_tot
		setValue(vb_reg.e_beta[k], beta[k])
	end

	beta2 = beta * beta' + beta2_eps * eye(vb_reg.k_tot)
	for k1=1:vb_reg.k_tot, k2=1:k1
		setValue(vb_reg.e_beta2_ud[ vb_reg.beta2_ind[k1, k2] ], beta2[k1, k2])
	end
end

function set_gamma(gamma::Float64, m::Int64, vb_reg::VBRandomEffectsRegression; gamma2_eps=1.0)
	setValue(vb_reg.e_gamma[m], gamma)
	setValue(vb_reg.e_gamma2[m], gamma ^ 2 + gamma2_eps)
end

function set_tau(tau::Float64, vb_reg::VBRandomEffectsRegression)
	setValue(vb_reg.e_tau, tau)
	setValue(vb_reg.e_log_tau, log(tau))
end

function set_nu(nu::Float64, vb_reg::VBRandomEffectsRegression)
	setValue(vb_reg.e_nu, nu)
	setValue(vb_reg.e_log_nu, log(nu))
end


function get_e_beta(vb_reg::VBRandomEffectsRegression)
	[ getValue(vb_reg.e_beta[k]) for k=1:vb_reg.k_tot ]
end

function get_e_beta2(vb_reg::VBRandomEffectsRegression)
	[ getValue(vb_reg.e_beta2_ud[vb_reg.beta2_ind[k1, k2]])
	    for k1=1:vb_reg.k_tot, k2=1:vb_reg.k_tot ]
end

function get_e_gamma(vb_reg::VBRandomEffectsRegression)
	[ getValue(vb_reg.e_gamma[m]) for m=1:vb_reg.re_num ]
end

function get_beta_ind_model(vb_reg::VBRandomEffectsRegression)
	# The indices of beta in the model.
	Int64[ vb_reg.e_beta[i].col for i=1:vb_reg.k_tot ]
end

function get_beta2_ind_model(vb_reg::VBRandomEffectsRegression)
	# The indices of beta2 in the model.
	Int64[ vb_reg.e_beta2_ud[vb_reg.beta2_ind[k1, k2]].col for
	       k1=1:vb_reg.k_tot, k2=1:vb_reg.k_tot ]
end

function get_gamma_ind_model(vb_reg::VBRandomEffectsRegression, m::Int64)
	# The indices of a gamma random effect in the model.
	vb_reg.e_gamma[m].col
end

function get_gamma2_ind_model(vb_reg::VBRandomEffectsRegression, m::Int64)
	# The indices of a gamma2 random effect in the model.
	vb_reg.e_gamma2[m].col
end


############

function get_loglik(param_val::Array{Float64,1}, vb_reg::VBRandomEffectsRegression)
	@assert length(param_val) == length(vb_reg.m.colVal)
	MathProgBase.eval_f(vb_reg.m_eval, param_val)
end

function get_loglik_deriv!(param_val::Array{Float64,1}, grad::Array{Float64,1},
	                       vb_reg::VBRandomEffectsRegression)
	@assert length(param_val) == length(vb_reg.m.colVal)
	@assert length(grad) == length(vb_reg.m.colVal)

	MathProgBase.eval_grad_f(vb_reg.m_eval, grad, param_val)
end

function get_loglik_deriv(param_val::Array{Float64,1}, vb_reg::VBRandomEffectsRegression)
	grad = fill(0., length(param_val))
	MathProgBase.eval_grad_f(vb_reg.m_eval, grad, param_val)
	grad
end

function get_loglik_hess(param_val::Array{Float64,1}, vb_reg::VBRandomEffectsRegression)
	# The fourth argument (1.0) multiplies the objective, and the last argument
	# (zeros) multiplies the constraints.
	n_params = length(param_val)
	MathProgBase.eval_hesslag(vb_reg.m_eval, vb_reg.hess_vec,
		                      param_val, 1.0, zeros(vb_reg.numconstr))
	this_hess_ld = sparse(vb_reg.hess_struct[1], vb_reg.hess_struct[2],
		                  vb_reg.hess_vec, n_params, n_params)
	this_hess = this_hess_ld + this_hess_ld' - sparse(diagm(diag(this_hess_ld)))
	this_hess
end



###################################
# Get the variational updates using
# the derivatives of the log likelihood.

function get_beta_natural_parameters(vb_reg::VBRandomEffectsRegression)
	function unpack_ud_matrix(ud_vector)
		# Convert a vector of upper diagonal entries into a
		# matrix with halved off-diagonal entries.
		# This is what's needed to convert the coefficients
		# of the derivative wrt beta2 into a matrix V such that
		# tr(V * beta2) = coeffs' * beta2

		k_tot = vb_reg.k_tot

		ud_mat = Array(Float64, (k_tot, k_tot))
		for k1=1:k_tot, k2=1:k_tot
			ud_mat[k1, k2] =
				(k1 <= k2 ? ud_vector[(k1 + (k2 - 1) * k2 / 2)] :
					        ud_vector[(k2 + (k1 - 1) * k1 / 2)])
			ud_mat[k1, k2] *= k1 != k2 ? 0.5: 1.
		end
		ud_mat
	end

	# Col values can be referenced like so:
	beta_i = get_beta_ind_model(vb_reg)
	beta2_i = sort(unique(get_beta2_ind_model(vb_reg)))
	# Functions to get the values.
	get_loglik_deriv!(vb_reg.m.colVal, vb_reg.grad, vb_reg)

	beta_suff = vb_reg.grad[beta_i]
	beta2_suff = -2 * unpack_ud_matrix(vb_reg.grad[beta2_i])

	beta_suff, beta2_suff
end


function update_beta_params(vb_reg::VBRandomEffectsRegression)
	beta_suff, beta2_suff = get_beta_natural_parameters(vb_reg)

	# Mean params from the canonincal representation is provided
	# by Distributions.jl:
	beta_q = MvNormalCanon(beta_suff, beta2_suff)
	e_beta_update = mean(beta_q)
	beta_q_cov = cov(beta_q)
	e_beta2_ud_update = beta_q_cov + e_beta_update * e_beta_update'

	for k1=1:vb_reg.k_tot
		setValue(vb_reg.e_beta[k1], e_beta_update[k1])
	end

	for k2=1:1:vb_reg.k_tot, k1=1:k2
		setValue(vb_reg.e_beta2_ud[vb_reg.beta2_ind[k1, k2]],
			     e_beta2_ud_update[k1, k2])
	end

	# Return the entropy.
	0.5 * logdet(beta_q_cov)
end

function get_gamma_natural_parameters(vb_reg::VBRandomEffectsRegression)
	# Since the gamma terms don't co-occur you can update them
	# all at once.

	# Col values can be referenced like so:
	gamma_i = Int64[ get_gamma_ind_model(vb_reg, m) for m=1:vb_reg.re_num ]
	gamma2_i = Int64[ get_gamma2_ind_model(vb_reg, m) for m=1:vb_reg.re_num ]

	# Functions to get the values.
	get_loglik_deriv!(vb_reg.m.colVal, vb_reg.grad, vb_reg)

	gamma_suff = vb_reg.grad[gamma_i]
	gamma2_suff = -2 * vb_reg.grad[gamma2_i]

	gamma_suff, gamma2_suff
end

function update_gamma_params(vb_reg::VBRandomEffectsRegression)
	gamma_suff, gamma2_suff = get_gamma_natural_parameters(vb_reg)

	# Mean params from the canonincal representation is provided
	# by Distributions.jl:
	gamma_q = [ NormalCanon(gamma_suff[m], gamma2_suff[m]) for m=1:vb_reg.re_num ]
	e_gamma_update = [ mean(gamma_q[m]) for m=1:vb_reg.re_num ]
	gamma_q_var = [ var(gamma_q[m]) for m=1:vb_reg.re_num ]
	e_gamma2_update = gamma_q_var + e_gamma_update .^ 2

	for m=1:vb_reg.re_num
		setValue(vb_reg.e_gamma[m], e_gamma_update[m])
		setValue(vb_reg.e_gamma2[m], e_gamma2_update[m])
	end

	# Return the entropy.
	0.5 * sum(log(gamma_q_var))
end

function get_tau_natural_parameters(vb_reg::VBRandomEffectsRegression)
	ll_deriv = get_loglik_deriv!(vb_reg.m.colVal, vb_reg.grad, vb_reg)

	tau_alpha = vb_reg.grad[vb_reg.e_log_tau.col] + 1
	tau_beta = -vb_reg.grad[vb_reg.e_tau.col]

	@assert tau_alpha >= 0
	@assert tau_beta >= 0

	tau_alpha, tau_beta
end

function update_tau_params(vb_reg::VBRandomEffectsRegression)
	tau_alpha, tau_beta = get_tau_natural_parameters(vb_reg)

	setValue(vb_reg.e_tau, tau_alpha / tau_beta)
	setValue(vb_reg.e_log_tau, digamma(tau_alpha) - log(tau_beta))

	# Return the entropy.
	tau_alpha - log(tau_beta) + lgamma(tau_alpha) +
		(1 - tau_alpha) * digamma(tau_alpha)
end

function get_nu_natural_parameters(vb_reg::VBRandomEffectsRegression)
	ll_deriv = get_loglik_deriv!(vb_reg.m.colVal, vb_reg.grad, vb_reg)

	nu_alpha = vb_reg.grad[vb_reg.e_log_nu.col] + 1
	nu_beta = -vb_reg.grad[vb_reg.e_nu.col]

	@assert nu_alpha >= 0
	@assert nu_beta >= 0

	nu_alpha, nu_beta
end

function update_nu_params(vb_reg::VBRandomEffectsRegression)
	nu_alpha, nu_beta = get_nu_natural_parameters(vb_reg)

	setValue(vb_reg.e_nu, nu_alpha / nu_beta)
	setValue(vb_reg.e_log_nu, digamma(nu_alpha) - log(nu_beta))

	# Return the entropy.
	nu_alpha - log(nu_beta) + lgamma(nu_alpha) +
		(1 - nu_alpha) * digamma(nu_alpha)
end


function fit_model(vb_reg::VBRandomEffectsRegression, max_iter, tol)
	last_val = -Inf
	for i=1:max_iter
		normal_entropy = update_beta_params(vb_reg)
		gamma_entropy = update_gamma_params(vb_reg)
		tau_entropy = update_tau_params(vb_reg)
		nu_entropy = update_nu_params(vb_reg)

		new_val = get_loglik(vb_reg.m.colVal, vb_reg) +
		          sum(normal_entropy) + gamma_entropy + tau_entropy + nu_entropy
		diff = new_val - last_val
		if abs(diff) < tol
			println("Convergence reached.")
			break
		end
		last_val = new_val
		#println("$i: $diff")
	end
end


#########################################
# Get the variational covariances

function get_tau_variational_covariance(vb_reg::VBRandomEffectsRegression)
	# Return an array of triplets (i, j, q_cov[i, j]) that can
	# be used to populate a sparse matrix representing the variational
	# covariance for the tau parameters.

	tau_alpha, tau_beta = get_tau_natural_parameters(vb_reg)
	e_tau_col = vb_reg.e_tau.col
	e_log_tau_col = vb_reg.e_log_tau.col

	tau_cov = (Int64, Int64, Float64)[]
	push!(tau_cov, (e_tau_col,     e_tau_col,     tau_alpha / (tau_beta ^ 2)))
	push!(tau_cov, (e_log_tau_col, e_log_tau_col, trigamma(tau_alpha)))
	push!(tau_cov, (e_tau_col,     e_log_tau_col, 1 / tau_beta))
	push!(tau_cov, (e_log_tau_col, e_tau_col,     1 / tau_beta))
	tau_cov
end


function get_nu_variational_covariance(vb_reg::VBRandomEffectsRegression)
	# Return an array of triplets (i, j, q_cov[i, j]) that can
	# be used to populate a sparse matrix representing the variational
	# covariance for the nu parameters.

	nu_alpha, nu_beta = get_nu_natural_parameters(vb_reg)
	e_nu_col = vb_reg.e_nu.col
	e_log_nu_col = vb_reg.e_log_nu.col

	nu_cov = (Int64, Int64, Float64)[]
	push!(nu_cov, (e_nu_col,     e_nu_col,     nu_alpha / (nu_beta ^ 2)))
	push!(nu_cov, (e_log_nu_col, e_log_nu_col, trigamma(nu_alpha)))
	push!(nu_cov, (e_nu_col,     e_log_nu_col, 1 / nu_beta))
	push!(nu_cov, (e_log_nu_col, e_nu_col,     1 / nu_beta))
	nu_cov
end

function get_univariate_normal_variational_covariance(e_norm, e_norm2, e_col, e2_col)
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

function get_gamma_variational_covariance(vb_reg)
	gamma_cov = (Int64, Int64, Float64)[]
	for m=1:vb_reg.re_num
		this_cov = get_univariate_normal_variational_covariance(
			getValue(vb_reg.e_gamma[m]),
			getValue(vb_reg.e_gamma2[m]),
			vb_reg.e_gamma[m].col, vb_reg.e_gamma2[m].col)
		append!(gamma_cov, this_cov)
	end
	gamma_cov
end

function get_normal_fourth_order_cov(beta_mean, beta_cov, k1, k2, k3, k4)
	# Return the covariance between (beta_k1 beta_k2, beta_k3 beta_k4)
	# given the means and covariances of beta.
    cov_12 = beta_cov[k1, k2];
    cov_13 = beta_cov[k1, k3];
    cov_14 = beta_cov[k1, k4];
    cov_23 = beta_cov[k2, k3];
    cov_24 = beta_cov[k2, k4];
    cov_34 = beta_cov[k3, k4];

	m_1 = beta_mean[k1];
	m_2 = beta_mean[k2];
	m_3 = beta_mean[k3];
	m_4 = beta_mean[k4];

  return (cov_13 * cov_24 +
  	      cov_14 * cov_23 +
  	      cov_13 * m_2 * m_4 +
  	      cov_14 * m_2 * m_3 +
	      cov_23 * m_1 * m_4 +
	      cov_24 * m_1 * m_3);
end


function get_beta_variational_covariance(vb_reg::VBRandomEffectsRegression)
	beta_suff, beta2_suff = get_beta_natural_parameters(vb_reg)

	# Mean params from the canonincal representation is provided
	# by Distributions.jl:
	beta_q = MvNormalCanon(beta_suff, beta2_suff)

	# v stands for "value" because "e_beta" is already taken.
	v_beta = mean(beta_q)
	v_beta_cov = cov(beta_q)

	beta_cov = (Int64, Int64, Float64)[]

	# Get the linear covarainces
	beta_ind_model = get_beta_ind_model(vb_reg)
	beta2_ind_model = get_beta2_ind_model(vb_reg)

	# Get the linear covariances.
	for k1=1:vb_reg.k_tot, k2=1:vb_reg.k_tot
		i1 = beta_ind_model[k1]
		i2 = beta_ind_model[k2]
		push!(beta_cov, (i1, i2, v_beta_cov[k1, k2]))
	end

	# Get the covariance between the linear and quadratic terms.
	# This will be cov(mu_k1 mu_k2, mu_i3) := cov(mu2_i12, mu_i3).
	# Avoid double counting since only one mu_i1 mu_i2 is recorded.
	for k1=1:vb_reg.k_tot, k2=1:k1, k3=1:vb_reg.k_tot
		i12 = beta2_ind_model[k1, k2]
		i3 = beta_ind_model[k3]
		this_cov = v_beta[k1] * v_beta_cov[k2, k3] + v_beta[k2] * v_beta_cov[k1, k3]
		push!(beta_cov, (i3, i12, this_cov))
		push!(beta_cov, (i12, i3, this_cov))
	end

	# Get the covariance between the quadratic terms.
	# This will be cov(mu_k1 mu_k2, mu_k3 mu_k4) := cov(mu2_i12, mu_i34).
	# Avoid double counting since only one mu_k1 mu_k2
	# and mu_k3 mu_k4 is recorded.
	for k1=1:vb_reg.k_tot, k2=1:k1, k3=1:vb_reg.k_tot, k4=1:k3
		i12 = beta2_ind_model[k1, k2]
		i34 = beta2_ind_model[k3, k4]
		this_cov = get_normal_fourth_order_cov(v_beta, v_beta_cov,
			k1, k2, k3, k4)

		push!(beta_cov, (i12, i34, this_cov))
	end

	beta_cov
end

function sparse_mat_from_tuples(tup_array)
	sparse(Int64[x[1] for x=tup_array],
		     Int64[x[2] for x=tup_array],
		     Float64[x[3] for x=tup_array])
end

function get_variational_covariance(vb_reg::VBRandomEffectsRegression)
	q_cov = (Int64, Int64, Float64)[]
	append!(q_cov, get_tau_variational_covariance(vb_reg))
	append!(q_cov, get_nu_variational_covariance(vb_reg))
	append!(q_cov, get_gamma_variational_covariance(vb_reg))
	append!(q_cov, get_beta_variational_covariance(vb_reg))

	sparse_mat_from_tuples(q_cov)
end

function get_lrvb_cov(vb_reg::VBRandomEffectsRegression)
	mfvb_cov = get_variational_covariance(vb_reg)

	ll_hess = get_loglik_hess(vb_reg.m.colVal, vb_reg)

	vb_corr_term = eye(length(vb_reg.m.colVal)) - mfvb_cov * ll_hess
	lrvb_cov = vb_corr_term \ full(mfvb_cov);

	lrvb_cov
end

using JuMP
using ReverseDiffSparse
using Ipopt
using NLopt

# For now, define a bunch of stuff in the global namespace.
# This must be run after every redefinition of the model
# parameters (x, y, etc.)

# For the z convergence step:
z_tol = 1e-4
z_max_iter = 5000

##################
# Define the model

# This model is only for a z.
m = Model(solver=IpoptSolver(max_iter=z_max_iter, tol=z_tol, print_level=0))
#m = Model(solver=NLoptSolver(algorithm=:LD_LBFGS, maxeval=z_max_iter, ftol_rel=z_tol))
#m = Model(solver=NLoptSolver(algorithm=:LD_MMA, ftol_rel=z_tol)) # v bad
#m = Model(solver=NLoptSolver(algorithm=:LN_NELDERMEAD, ftol_rel=z_tol)) # v bad

@defVar(m, e_mu == 1.0);
@defVar(m, e_mu2 == 2.0);
@defVar(m, e_tau == 1);
@defVar(m, e_log_tau == 1);
@defVar(m, x == 1.0)
@defVar(m, y == 1.0)

# The variational parameters of z:
@defVar(m, e_z);
@defVar(m, e_z2);

# Define the elbo.

# Define some convenience expressions:
# Unconditional variance of z:
@defNLExpr(var_mu, e_mu2 - e_mu ^ 2);
@defNLExpr(var_z, e_z2 - e_z ^ 2);
@defNLExpr(e_exp_z, exp(e_z + 0.5 * var_z));

# Define the elbo.  See notes.
@defNLExpr(loglik_z,
		   e_tau * (-0.5 * e_z2 +
		            e_z * e_mu * x - 
		            0.5 * e_mu2 * (x ^ 2)) +
		   0.5 * e_log_tau -
		   e_exp_z + e_z * y);

# The priors aren't actually used here since we don't numerically optimize them.
# @defNLExpr(mu_prior, -0.5 * e_mu2 / mu_prior_var)
# @defNLExpr(tau_prior, -tau_prior_beta * e_tau + (tau_prior_alpha - 1) * e_log_tau)

@defNLExpr(entropy_z, 0.5 * log(var_z));
@defNLExpr(elbo, loglik_z + entropy_z);

@setNLObjective(m, Max, elbo)


ReverseDiffSparse.getvalue(elbo, m.colVal)
function update_z!(m::Model, x_vec, y_vec,
	               e_mu_val, e_mu2_val,
	               e_tau_val, e_log_tau_val,
	               e_z_val, e_z2_val)

	# Solve each z optimization one at a time.
	setValue(e_mu, e_mu_val)
	setValue(e_mu2, e_mu2_val)
	setValue(e_tau, e_tau_val)
	setValue(e_log_tau, e_log_tau_val)

	n = length(e_z_val)
	for i=1:n
		if i % 50 == 1
			print(".")
		end
		setValue(e_z, e_z_val[i])
		setValue(e_z2, e_z2_val[i])

		setValue(x, x_vec[i])
		setValue(y, y_vec[i])
	    solve(m)
	    e_z_val[i] = getValue(e_z)
	    e_z2_val[i] = getValue(e_z2)
	end
end

function get_tau_params(e_mu_val, e_mu2_val,
	   	                zx_sum, zz_sum, xx_sum, n,
	   	                tau_prior_alpha, tau_prior_beta)
	tau_beta = 0.5 * zz_sum -
	            zx_sum * e_mu_val +
	            0.5 * e_mu2_val * xx_sum + tau_prior_beta
	tau_alpha = 0.5 * n + 1 + tau_prior_alpha
	tau_alpha, tau_beta
end

function fit_model!(x_vec, y_vec,
	                e_mu_val, e_mu2_val,
	   	            e_tau_val, e_log_tau_val,
		            e_z_val, e_z2_val, prior;
		            max_iter=10, tol=1e-6)

	# This uses global variables defined above.

    # Fit the model for the given starting values.
  	diff = Inf
  	last_mu = e_mu_val
  	last_tau = e_tau_val
  	last_log_tau = e_log_tau_val

  	last_z = e_z_val

	n = length(x_vec)
	xx_sum = sum(x_vec .* x_vec)

	for i=1:max_iter
		print("z step...")
		update_z!(m, x_vec, y_vec, e_mu_val, e_mu2_val,
			      e_tau_val, e_log_tau_val, e_z_val, e_z2_val)
		print("...done.\n")

		zx_sum = sum(e_z_val .* x_vec)
		zz_sum = sum(e_z2_val)

		# Perform the mu and tau updates by hand using conjugacy.

		# Update the mean parameters
		mu_mean_param = e_tau_val * zx_sum
		mu_var_param = e_tau_val * xx_sum + 1 / prior.mu_prior_var
		e_mu_val = mu_mean_param / mu_var_param
		e_mu2_val = e_mu_val ^ 2 + 1 / mu_var_param
		@assert e_mu2_val > 0

		# Update the information parameters
		tau_alpha, tau_beta = get_tau_params(e_mu_val, e_mu2_val,
											 zx_sum, zz_sum, xx_sum, n,
			                                 prior.tau_prior_alpha,
			                                 prior.tau_prior_beta)
		e_tau_val = tau_alpha / tau_beta
		e_log_tau_val = digamma(tau_alpha) - log(tau_beta)

		@assert e_tau_val > 0

		mu_diff = abs(e_mu_val - last_mu) / (abs(last_mu) + 0.1)
		tau_diff = abs(e_tau_val - last_tau) / abs(last_tau) +
		           abs(e_log_tau_val - last_log_tau) / abs(last_log_tau)
		z_diff = mean(abs(e_z_val - last_z) / abs(last_z + 0.1))

		diff = mu_diff + tau_diff + z_diff
		last_mu = e_mu_val
		last_tau = e_tau_val
		last_log_tau = e_log_tau_val
		last_z = e_z_val
		if diff < tol
			println("Convergence reached: $diff")
			break
		end	
		println("$i: $diff")
	end
	e_mu_val, e_mu2_val, e_tau_val, e_log_tau_val, e_z_val, e_z2_val
end

println("Done.")

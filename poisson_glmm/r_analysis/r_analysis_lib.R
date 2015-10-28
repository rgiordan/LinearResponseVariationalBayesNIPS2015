library(lme4)
library(MCMCglmm)
library(coda)
library(jsonlite)

result.list <- list()

AppendResult <- function(res) {
  result.list[[length(result.list) + 1]] <<- res
}

ResultRow <- function(sim.id, variable, measurement, method, value) {
  # Produce a dataframe that can be appended to the results.
  AppendResult(data.frame(sim.id=sim.id, variable=variable, measurement=measurement,
                          method=method, value=as.numeric(value)))
}

ReadVBResults <- function(my.dat, sim.id) {
  # Append to the global <results.list> list the results from a VB
  # run as read in from the JSON data list my.dat.
  
  ResultRow(sim.id, "mu", "mean", "truth", value=my.dat$true_mu)
  ResultRow(sim.id, "tau", "mean", "truth", value=1 / my.dat$true_epsilon ^ 2)
  ResultRow(sim.id, "log_tau", "mean", "truth", value=-2 * log(my.dat$true_epsilon))
  
  # Get the VB results.
  ResultRow(sim.id, "mu", "mean", "mfvb", value=my.dat$e_mu_val)
  ResultRow(sim.id, "tau", "mean", "mfvb", value=my.dat$e_tau_val)
  ResultRow(sim.id, "log_tau", "mean", "mfvb", value=my.dat$e_log_tau_val)
  #mfvb.z.mean <- my.dat$e_z_val
  
  ResultRow(sim.id, "mu", "sd", "mfvb", value=sqrt(my.dat$mu_mfvb_var))
  ResultRow(sim.id, "tau", "sd", "mfvb", value=sqrt(my.dat$tau_mfvb_var))
  ResultRow(sim.id, "log_tau", "sd", "mfvb", value=sqrt(my.dat$log_tau_mfvb_var))
  #mfvb.z.var <- my.dat$z_mfvb_var
  
  ResultRow(sim.id, "mu", "sd", "lrvb", value=sqrt(my.dat$mu_lrvb_var))
  ResultRow(sim.id, "tau", "sd", "lrvb", value=sqrt(my.dat$tau_lrvb_var))
  ResultRow(sim.id, "log_tau", "sd", "lrvb", value=sqrt(my.dat$log_tau_lrvb_var))
  #lrvb.z.var <- my.dat$z_lrvb_var
}

GenerateMCMCResults <- function(my.dat, sim.id, mcmc.iters, save.z=FALSE) {
  # Append to the global <results.list> list the results from MCMC
  # based on data read in from the JSON data list my.dat.
  
  d <- fromJSON(my.dat$df_json)
  
  # MCMC:
  # It appears that MCMCglmm actually adds noise without any random effect.
  prior <- list(B=list(V=my.dat$mu_prior_var, mu=0),
                R=list(V=1 / my.dat$tau_prior_beta,
                       n=my.dat$tau_prior_alpha))
  mcmc.res <- MCMCglmm(y ~ x - 1, data=d, family="poisson", prior=prior,
                       nitt=mcmc.iters, pl=save.z)
  
  # Save the results.
  ResultRow(sim.id, "mu", "mean", "mcmc", mean(mcmc.res$Sol))
  ResultRow(sim.id, "mu", "sd", "mcmc", sd(mcmc.res$Sol))
  ResultRow(sim.id, "mu", "effsize", "mcmc", effectiveSize(mcmc.res$Sol))
  
  ResultRow(sim.id, "tau", "mean", "mcmc", mean(1 / mcmc.res$VCV))
  ResultRow(sim.id, "tau", "sd", "mcmc", sd(1 / mcmc.res$VCV))
  ResultRow(sim.id, "tau", "effsize", "mcmc", effectiveSize(1 / mcmc.res$VCV))
  
  ResultRow(sim.id, "log_tau", "mean", "mcmc", mean(-log(mcmc.res$VCV)))
  ResultRow(sim.id, "log_tau", "sd", "mcmc", sd(-log(mcmc.res$VCV)))

  if (save.z) {
    for (i in 1:nrow(d)) {
      ResultRow(sim.id, sprintf("z_%d", i), "mean", "mcmc", mean(mcmc.res$Liab[, i]))      
      ResultRow(sim.id, sprintf("z_%d", i), "sd",   "mcmc",sd(mcmc.res$Liab[, i]))      
    }
  }
  return(mcmc.res)
}

GetMCMCCov <- function(mcmc.res, my.dat) {
  # Save the full covariance matrix.

  d <- fromJSON(my.dat$df_json)
  
  # It is important that the order of these columns match those from julia.  I'll do
  # it carefully and inefficiently rather than assume that julia's order doesn't change.

  n.par <- 4 + 2 * nrow(d)
  mu_cols_i <- my.dat$mu_cols_i
  tau_cols_i <- my.dat$tau_cols_i
  z_i <- my.dat$z_i
  z2_i <- my.dat$z2_i
  
  stopifnot(n.par == length(mu_cols_i) + length(tau_cols_i) + length(z_i) + length(z2_i))
  
  mu.draws <- mcmc.res$Sol
  mu2.draws <- mcmc.res$Sol ^ 2
  tau.draws <- 1 / mcmc.res$VCV
  log.tau.draws <- -log(mcmc.res$VCV)
  z.draws <- mcmc.res$Liab
  
  n.theta.par <- length(c(mu_cols_i, tau_cols_i))
  mcmc.theta.draws <- matrix(NA, nrow(mcmc.res$Sol), n.theta.par)
  mcmc.theta.draws[, mu_cols_i] <- cbind(mu.draws, mu2.draws)  
  mcmc.theta.draws[, tau_cols_i] <- cbind(tau.draws, log.tau.draws)  
    
  mcmc.cov <- list(theta.z.cov=cov(mcmc.theta.draws, z.draws),
                   theta.cov=cov(mcmc.theta.draws))
  return(mcmc.cov)
}

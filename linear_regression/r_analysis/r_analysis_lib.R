library(lme4)
library(MCMCglmm)
library(coda)
library(jsonlite)

#####################
# Linear regression #
#####################

rl <- list()
AppendResult <- function(res) {
  # Appends <res> to the end of <rl>, the "result list", in the encolsing space.
  rl[[length(rl) + 1]] <<- res
}

ResultRow <- function(sim.id, measurement, method, variable, component, value) {
  # Produce a dataframe that can be appended to the result list <rl>.
  AppendResult(data.frame(sim.id=sim.id, measurement=measurement,
                          method=method, variable=variable,
                          component=component,
                          value=as.numeric(value)))
}


ReadJSONFile <- function(data.path, analysis.name, sim.id) {
  filename <- sprintf("%s_%d.json", analysis.name, sim.id)
  json.file <- file(file.path(data.path, filename), "r")
  my.dat <- fromJSON(readLines(json.file, warn=FALSE))
  close(json.file)
  return(my.dat)
}

GetCovNames <- function(k.tot) {
  return(c(paste("beta", 1:k.tot, sep=""), "tau", "log_tau", "nu", "log_nu"))
}

ReadVBResults <- function(my.dat, sim.id, save.gamma=FALSE, save.main.cov=FALSE) {
  # Append to the global <results.list> list the results from a VB
  # run as read in from the JSON data list my.dat.

  ResultRow(sim.id, "mean", "truth", "tau", -1, value=my.dat$true_tau)
  ResultRow(sim.id, "mean", "mfvb",  "tau", -1, value=my.dat$vb_tau)
  ResultRow(sim.id, "sd",   "mfvb",  "tau", -1, value=sqrt(my.dat$mfvb_var_tau))
  ResultRow(sim.id, "sd",   "lrvb",  "tau", -1, value=sqrt(my.dat$lrvb_var_tau))
  
  ResultRow(sim.id, "mean", "truth", "nu", -1, value=my.dat$true_nu)
  ResultRow(sim.id, "mean", "mfvb",  "nu", -1, value=my.dat$vb_nu)
  ResultRow(sim.id, "sd",   "mfvb",  "nu", -1, value=sqrt(my.dat$mfvb_var_nu))
  ResultRow(sim.id, "sd",   "lrvb",  "nu", -1, value=sqrt(my.dat$lrvb_var_nu))

  ResultRow(sim.id, "mean", "truth", "log_tau", -1, value=log(my.dat$true_tau))
  ResultRow(sim.id, "mean", "mfvb",  "log_tau", -1, value=my.dat$vb_log_tau)
  ResultRow(sim.id, "sd",   "mfvb",  "log_tau", -1, value=sqrt(my.dat$mfvb_var_log_tau))
  ResultRow(sim.id, "sd",   "lrvb",  "log_tau", -1, value=sqrt(my.dat$lrvb_var_log_tau))
  
  ResultRow(sim.id, "mean", "truth", "log_nu", -1, value=log(my.dat$true_nu))
  ResultRow(sim.id, "mean", "mfvb",  "log_nu", -1, value=my.dat$vb_log_nu)
  ResultRow(sim.id, "sd",   "mfvb",  "log_nu", -1, value=sqrt(my.dat$mfvb_var_log_nu))
  ResultRow(sim.id, "sd",   "lrvb",  "log_nu", -1, value=sqrt(my.dat$lrvb_var_log_nu))
  
  for (i in 1:length(my.dat$vb_beta)) {
    ResultRow(sim.id, "mean", "truth", "beta", i, value=my.dat$true_beta[i])
    ResultRow(sim.id, "mean", "mfvb",  "beta", i, value=my.dat$vb_beta[i])
    ResultRow(sim.id, "sd",   "mfvb",  "beta", i, value=sqrt(my.dat$mfvb_cov_beta[i, i]))
    ResultRow(sim.id, "sd",   "lrvb",  "beta", i, value=sqrt(my.dat$lrvb_cov_beta[i, i]))      
  }

  if (save.gamma) {
    for (i in 1:length(my.dat$true_gamma)) {
      ResultRow(sim.id, "mean", "truth", "gamma", i, value=my.dat$true_gamma[i])        
      ResultRow(sim.id, "mean", "mfvb",  "gamma", i, value=my.dat$vb_gamma[i])
      ResultRow(sim.id, "sd",   "mfvb",  "gamma", i, value=sqrt(my.dat$mfvb_var_gamma[i]))
      ResultRow(sim.id, "sd",   "lrvb",  "gamma", i, value=sqrt(my.dat$lrvb_var_gamma[i]))      
    }    
  }
  
  if (save.main.cov) {
    # The order of variables are (as of now) beta, tau, log tau, nu, log nu
    cov.names <- GetCovNames(my.dat$k_tot)
    lrvb.cov.main <- my.dat$lrvb_cov_main
    mfvb.cov.main <- my.dat$mfvb_cov_main
    main.n <- dim(lrvb.cov.main)[1]
    stopifnot(main.n == my.dat$k_tot + 4) # k_tot beta components and four variances
    for (i in 2:main.n) {
      for (j in 1:(i - 1)) {
        row.name <- paste("cov", cov.names[i], cov.names[j], sep="_")
        ResultRow(sim.id, "cov", "lrvb", row.name, -1, value=lrvb.cov.main[i, j])        
        ResultRow(sim.id, "cov", "mfvb", row.name, -1, value=mfvb.cov.main[i, j])                
      }
    }
  }
}


GenerateMCMCResults <- function(my.dat, sim.id, mcmc.iters,
                                save.gamma=FALSE, save.main.cov=FALSE) {
  # Append to the global <results.list> list the results from MCMC
  # based on data read in from the JSON data list my.dat.
  # MCMC fit.

  d <- fromJSON(my.dat$df_json)
  
  k.tot <- my.dat$k_tot
  reg.string <- paste(sprintf("x%d", 1:k.tot), collapse=" + ")
  mcmc.formula <- formula(sprintf("y ~ %s - 1", reg.string))
  
  prior <- list(B=list(V=my.dat$beta_prior_info_scale * diag(k.tot),
                       mu=rep(0, k.tot)),
                G=list(G1=list(V=1 / my.dat$nu_prior_gamma,
                               n=my.dat$nu_prior_alpha)),
                R=list(V=1 / my.dat$tau_prior_gamma,
                       n=my.dat$tau_prior_alpha))
  
  # It does not make any difference if you use idv or idh (and shouldn't
  # with only one effect, if I understand right)
  mcmc.res <- MCMCglmm(mcmc.formula,
                       random= ~ idv(z):re_ind,
                       rcov= ~ units,
                       data=d, prior=prior,
                       family="gaussian", pl=TRUE, pr=save.gamma, nitt=mcmc.iters,
                       verbose=FALSE)
  
  #plot(mcmc.res)
  
  # Save results.
  beta.draws <- mcmc.res$Sol[, 1:k.tot]
  beta.mean <- colMeans(beta.draws)
  beta.sd <- apply(beta.draws, 2, sd)
  beta.effsize <- effectiveSize(beta.draws)
  for (i in 1:length(beta.mean)) {
    ResultRow(sim.id, "mean", "mcmc", "beta", i, value=beta.mean[i])    
    ResultRow(sim.id, "sd",   "mcmc", "beta", i, value=beta.sd[i])
    ResultRow(sim.id, "effsize", "mcmc", "beta", i, value=beta.effsize[i])    
  }
  
  tau.draws <- 1 / mcmc.res$VCV[, "units"]
  ResultRow(sim.id, "mean", "mcmc", "tau", -1, value=mean(tau.draws))  
  ResultRow(sim.id, "sd",   "mcmc", "tau", -1, value=sd(tau.draws))    
  ResultRow(sim.id, "effsize",   "mcmc", "tau", -1, value=effectiveSize(tau.draws))    
  
  nu.draws <- 1 / mcmc.res$VCV[, "z.re_ind"]
  ResultRow(sim.id, "mean", "mcmc", "nu", -1, value=mean(nu.draws))
  ResultRow(sim.id, "sd",   "mcmc", "nu", -1, value=sd(nu.draws))    
  ResultRow(sim.id, "effsize",   "mcmc", "nu", -1, value=effectiveSize(nu.draws))    

  log.tau.draws <- log(tau.draws)
  ResultRow(sim.id, "mean", "mcmc", "log_tau", -1, value=mean(log.tau.draws))  
  ResultRow(sim.id, "sd",   "mcmc", "log_tau", -1, value=sd(log.tau.draws))    
  ResultRow(sim.id, "effsize",   "mcmc", "log_tau", -1, value=effectiveSize(log.tau.draws))    
  
  log.nu.draws <- log(nu.draws)
  ResultRow(sim.id, "mean", "mcmc", "log_nu", -1, value=mean(log.nu.draws))
  ResultRow(sim.id, "sd",   "mcmc", "log_nu", -1, value=sd(log.nu.draws))    
  ResultRow(sim.id, "effsize",   "mcmc", "log_nu", -1, value=effectiveSize(log.nu.draws))    
  
  if (save.gamma) {
    gamma.draws <- mcmc.res$Sol[, -(1:k.tot)]
    gamma.means <- colMeans(gamma.draws)
    gamma.sds <- apply(gamma.draws, 2, sd)
    gamma.effsize <- effectiveSize(gamma.draws)
    for (i in 1:ncol(gamma.draws)) {
      ResultRow(sim.id, "mean", "mcmc", "gamma", i, value=gamma.means[i])
      ResultRow(sim.id, "sd",   "mcmc", "gamma", i, value=gamma.sds[i]) 
      ResultRow(sim.id, "effsize",   "mcmc", "gamma", i, value=gamma.effsize[i])          
    }
  }
  
  if (save.main.cov) {
    main.draws <- cbind(beta.draws, tau.draws, log.tau.draws, nu.draws, log.nu.draws)
    mcmc.cov.main <- cov(main.draws)
    cov.names <- GetCovNames(ncol(beta.draws))
    main.n <- dim(mcmc.cov.main)[1]
    for (i in 2:main.n) {
      for (j in 1:(i - 1)) {
        row.name <- paste("cov", cov.names[i], cov.names[j], sep="_")
        ResultRow(sim.id, "cov", "mcmc", row.name, -1, value=mcmc.cov.main[i, j])             
      }
    }
    
  }
}

GetFrequentistResults <- function(my.dat, sim.id) {
  # Minimal frequentist results for sanity checking.
  
  d <- fromJSON(my.dat$df_json)
  
  # Note that the lmer estimate matches well with the MFVB estimate of
  # nu, which does not match MCMC that well...
  lmm.res <- lmer(y ~ x1 + x2 + x3 - 1 + (z|re_ind), d)
  lmm.sum <- summary(lmm.res)
  #print(lmm.sum)
  lmm.nu <- 1/ attr(lmm.sum$varcor$re_ind, "stddev")["z"]
  lmm.beta <- lmm.sum$coefficients[, "Estimate"]
  
  for (i in 1:length(beta.mean)) {
    ResultRow(sim.id, "mean", "lmer", "beta", i, value=lmm.beta[i])    
  }
  
  ResultRow(sim.id, "mean", "lmer", "nu", -1, value=lmm.nu)     
}


GetAnalysisData <- function(my.dat, mcmc.iters) {
  ResultRow(-1, "truth", "analysis", "n", -1, value=my.dat$n)
  ResultRow(-1, "truth", "analysis", "re_num", -1, value=length(my.dat$true_gamma))
  ResultRow(-1, "truth", "analysis", "z_sd", -1, value=my.dat$z_sd)
  ResultRow(-1, "truth", "analysis", "mcmc.iters", -1, value=mcmc.iters)
  
  ResultRow(-1, "truth", "analysis", "beta_prior_info_scale", -1,
            value=my.dat$beta_prior_info_scale)  
  ResultRow(-1, "truth", "analysis", "nu_prior_alpha", -1, value=my.dat$nu_prior_alpha)  
  ResultRow(-1, "truth", "analysis", "nu_prior_gamma", -1, value=my.dat$nu_prior_gamma)  
  ResultRow(-1, "truth", "analysis", "tau_prior_alpha", -1, value=my.dat$tau_prior_alpha)  
  ResultRow(-1, "truth", "analysis", "tau_prior_gamma", -1, value=my.dat$tau_prior_gamma)    
}

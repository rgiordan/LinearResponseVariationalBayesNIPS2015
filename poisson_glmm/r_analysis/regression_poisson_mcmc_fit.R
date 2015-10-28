library(lme4)
library(MCMCglmm)
library(coda)
library(jsonlite)

base.dir <- "~/Documents/git_repos/variational_bayes/poisson_glmm/"
setwd(base.dir)

source(file.path(base.dir, "r_analysis/r_analysis_lib.R"))

data.path <- file.path(base.dir, "data")
#analysis.name <- "poisson_glmm"
#analysis.name <- "poisson_glmm_low_epsilon"
analysis.name <- "poisson_glmm_z_theta_cov"

save.results <- TRUE
save.z <- TRUE
save.cov <- TRUE
if (!save.z) {
  stopifnot(!save.cov)
}

mcmc.iters <- 20000

result.list <- list()
theta.cov.list <- list()
theta.z.cov.list <- list()
id_range = c(1, 100)
for (sim.id in id_range[1]:id_range[2]) {
  print(sprintf("Simulation %d", sim.id))
  filename <- sprintf("%s_%d.json", analysis.name, sim.id)
  json.file <- file(file.path(data.path, filename), "r")
  my.dat <- fromJSON(readLines(json.file, warn=FALSE))
  close(json.file)
  
  ReadVBResults(my.dat, sim.id)
  mcmc.res <- GenerateMCMCResults(my.dat, sim.id, mcmc.iters, save.z=save.z)
  
  if (save.cov) {
    stopifnot(save.z) # You need the full thing for this to work.
    sim.char <- as.character(sim.id)
    mcmc.cov <- GetMCMCCov(mcmc.res, my.dat)
    
    mcmc.theta.cov <- mcmc.cov$theta.cov
    mcmc.theta.z.cov <- t(mcmc.cov$theta.z.cov)

    lrvb.theta.z.cov <- my.dat$lrvb_cov_thetaz
    lrvb.theta.cov <- my.dat$theta_lrvb_cov
    mfvb.theta.cov <- my.dat$theta_mfvb_cov
    diag(lrvb.theta.cov) <- 0
    diag(mfvb.theta.cov) <- 0
    diag(mcmc.theta.cov) <- 0

    theta.z.cov.len <- length(as.numeric(lrvb.theta.z.cov))
    theta.z.cov.list[[length(theta.z.cov.list) + 1]] <-
      data.frame(sim.id=sim.id, method="lrvb", component=1:theta.z.cov.len,
                  x=as.numeric(lrvb.theta.z.cov))
    theta.z.cov.list[[length(theta.z.cov.list) + 1]] <-
      data.frame(sim.id=sim.id, method="mcmc", component=1:theta.z.cov.len,
                 x=as.numeric(mcmc.theta.z.cov))

    theta.cov.len <- length(as.numeric(lrvb.theta.cov))
    theta.cov.list[[length(theta.cov.list) + 1]] <-
      data.frame(sim.id=sim.id, method="lrvb", component=1:theta.cov.len,
                 x=as.numeric(lrvb.theta.cov))
    theta.cov.list[[length(theta.cov.list) + 1]] <-
      data.frame(sim.id=sim.id, method="mfvb", component=1:theta.cov.len,
                 x=as.numeric(mfvb.theta.cov))
    theta.cov.list[[length(theta.cov.list) + 1]] <-
      data.frame(sim.id=sim.id, method="mcmc", component=1:theta.cov.len,
                 x=as.numeric(mcmc.theta.cov))
    
    #plot(lrvb.theta.z.cov, mcmc.theta.z.cov); abline(0,1)
    
    #plot(mcmc.theta.cov, lrvb.theta.cov, col="red", pch="o")    
    #points(mcmc.theta.cov, mfvb.theta.cov, col="blue")
    #abline(0,1)
  
  }
  
  # # NB: Frequentist methods beat everything.
  # d$c <- factor(1:nrow(d))
  # glmm.res <- glmer(y ~ x - 1 + (1|c), d, family=poisson)
  # summary(glmm.res)  
}

# Append some values common to all analyses
ResultRow(-1, "analysis", "n", "truth", value=my.dat$n)
ResultRow(-1, "analysis", "mcmc_iters", "truth", value=mcmc.iters)
ResultRow(-1, "analysis", "mu_prior_var", "truth", value=my.dat$mu_prior_var)
ResultRow(-1, "analysis", "tau_prior_alpha", "truth", value=my.dat$tau_prior_alpha)
ResultRow(-1, "analysis", "tau_prior_beta", "truth", value=my.dat$tau_prior_beta)

id_range_string = sprintf("%d_%d", id_range[1], id_range[2])

if (save.results) {
  out.filename <- file.path(data.path, sprintf("%s_%s_results.csv", analysis.name, id_range_string))
  write.csv(do.call(rbind, result.list), file=out.filename, quote=F, row.names=F)
  
  theta.out.filename <- file.path(data.path,
                                  sprintf("%s_%s_theta_cov_results.csv",
                                          analysis.name, id_range_string))
  write.csv(do.call(rbind, theta.cov.list), file=theta.out.filename, quote=F, row.names=F)
  theta.z.out.filename <- file.path(data.path,
                                    sprintf("%s_%s_theta_z_cov_results.csv",
                                            analysis.name, id_range_string))
  write.csv(do.call(rbind, theta.z.cov.list), file=theta.z.out.filename, quote=F, row.names=F)  
}

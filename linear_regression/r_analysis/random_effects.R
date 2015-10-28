library(lme4)
library(MCMCglmm)
library(coda)
library(jsonlite)

library(ggplot2)
library(dplyr)
library(reshape2)

base.dir <- "~/Documents/git_repos/variational_bayes/linear_regression/"
source(file.path(base.dir, "r_analysis/r_analysis_lib.R"))

data.path <- file.path(base.dir, "data")
#analysis.name <- "re_regression_less_corr"
#analysis.name <- "re_regression_z_x1_x2"
analysis.name <- "re_regression_full_cov"

# Options
mcmc.iters <- 20000
save.results <- FALSE
id_range = c(1, 100)
save.gamma <- TRUE
save.main.cov <- TRUE
save.freq.results <- FALSE

# Load VB results
rl <- list()
for (sim.id in id_range[1]:id_range[2]) {
  
  print(sprintf("Simulation %d", sim.id))
  my.dat <- ReadJSONFile(data.path, analysis.name, sim.id)  
  ReadVBResults(my.dat, sim.id, save.gamma=save.gamma, save.main.cov=save.main.cov)
}
vb.results <- do.call(rbind, rl)

# Generate MCMC results
rl <- list()
for (sim.id in id_range[1]:id_range[2]) {

  print(sprintf("Simulation %d", sim.id))
  my.dat <- ReadJSONFile(data.path, analysis.name, sim.id)  

  GenerateMCMCResults(my.dat, sim.id, mcmc.iters,
                      save.gamma=save.gamma, save.main.cov=save.main.cov)
  if (save.freq.results) {
    GetFrequentistResults(my.dat, sim.id)    
  }
}
mcmc.results <- do.call(rbind, rl)

# Load analysis results
rl <- list()
my.dat <- ReadJSONFile(data.path, analysis.name, 1)  
GetAnalysisData(my.dat, mcmc.iters)
analysis.results <- do.call(rbind, rl)

# Combine and save
results <- rbind(vb.results, mcmc.results, analysis.results)
results$component <- ordered(results$component)

if (save.results) {
  #save(results, file=file.path(data.path, sprintf("%s_final.Rdata", analysis.name)))
  save.filename <- file.path(data.path, sprintf("%s_final.csv", analysis.name))
  write.csv(results, file=save.filename, quote=FALSE, row.names=FALSE)  
}


#############

base.results <-
  filter(results,
         measurement %in%  c("mean", "sd"),
         method %in% c("mfvb", "lrvb", "mcmc", "truth")) %>%
  dcast(sim.id + variable + measurement + component ~ method)

########## The mean:
ggplot(filter(base.results, measurement == "mean")) +
  geom_point(aes(x=mcmc, y=mfvb, color="vb"), size=3) +
  geom_abline(aes(intercept=0, slope=1)) +
  expand_limits(x=0, y=0) +
  geom_abline(aes(slope=1, intercept=0), color="gray") +
  facet_grid(~ variable)

ggplot(filter(base.results, measurement == "mean", variable %in% c("gamma", "beta"))) +
  geom_point(aes(x=mcmc, y=mfvb, color="vb"), size=3) +
  geom_abline(aes(intercept=0, slope=1)) +
  expand_limits(x=0, y=0) +
  geom_abline(aes(slope=1, intercept=0), color="gray") +
  facet_grid(~ variable)

ggplot(filter(base.results, measurement == "mean", variable %in% c("nu", "log_nu"))) +
  geom_point(aes(x=mcmc, y=mfvb, color="vb"), size=3) +
  geom_abline(aes(intercept=0, slope=1)) +
  expand_limits(x=0, y=0) +
  geom_abline(aes(slope=1, intercept=0), color="gray") +
  facet_grid(~ variable)

########## The standard deviations:
ggplot(filter(base.results, measurement == "sd", variable == "beta")) +
  geom_point(aes(x=mcmc, y=mfvb, color="mfvb"), size=3) +
  geom_point(aes(x=mcmc, y=lrvb, color="lrvb"), size=3) +
  geom_abline(aes(intercept=0, slope=1)) +
  expand_limits(x=0, y=0) +
  geom_abline(aes(slope=1, intercept=0), color="gray") +
  facet_grid(~ component)

ggplot(filter(base.results, measurement == "sd", variable == "tau")) +
  geom_point(aes(x=mcmc, y=mfvb, color="mfvb"), size=3) +
  geom_point(aes(x=mcmc, y=lrvb, color="lrvb"), size=3) +
  geom_abline(aes(intercept=0, slope=1)) +
  expand_limits(x=0, y=0) +
  geom_abline(aes(slope=1, intercept=0), color="gray")

ggplot(filter(base.results, measurement == "sd", variable == "nu")) +
  geom_point(aes(x=mcmc, y=mfvb, color="mfvb"), size=3) +
  geom_point(aes(x=mcmc, y=lrvb, color="lrvb"), size=3) +
  geom_abline(aes(intercept=0, slope=1)) +
  expand_limits(x=0, y=0) +
  geom_abline(aes(slope=1, intercept=0), color="gray")

ggplot(filter(base.results, measurement == "sd", variable == "gamma", component == 1)) +
  geom_point(aes(x=mcmc, y=mfvb, color="mfvb"), size=3) +
  geom_point(aes(x=mcmc, y=lrvb, color="lrvb"), size=3) +
  geom_abline(aes(intercept=0, slope=1)) +
  expand_limits(x=0, y=0) +
  geom_abline(aes(slope=1, intercept=0), color="gray")



### Not sure this is working still.
lmm.compare <- filter(results, measurement == "mean") %>%
  dcast(sim.id + variable + component ~ method)

ggplot(filter(lmm.compare, variable == "beta")) +
  geom_point(aes(x=truth, y=mfvb, color="mfvb"), size=3) +
  geom_point(aes(x=truth, y=mcmc, color="mcmc"), size=3) +
  geom_point(aes(x=truth, y=lmer, color="lmer"), size=3) +
  geom_abline(aes(intercept=0, slope=1)) +
  facet_grid( ~ component)

ggplot(filter(lmm.compare, variable == "nu")) +
  geom_point(aes(x=mfvb, y=mcmc, color="mcmc"), size=3) +
  geom_point(aes(x=mfvb, y=lmer, color="lmer"), size=3) +
  geom_abline(aes(intercept=0, slope=1))

###############
# Check the knitr stuff

if (F) {
  setwd("/home/rgiordan/Documents/git_repos/variational_bayes/writing/nips_2015")

  # Load and pre-process the simulation results.
  #re.results <- read.csv("./data/re_regression_z_x1_x2_final.csv", header=T)
  re.results <- read.csv("./data/re_regression_full_cov_final.csv", header=T)
  
  # A data frame for graphing:
  re.graph.df <- filter(re.results, measurement %in%  c("mean", "sd")) %>%
    dcast(sim.id + variable + measurement + component ~ method)
  
  # A data frame with analysis details:
  re.analysis.df <- filter(re.results, method == "analysis") %>%
    dcast(measurement ~ variable)
  
}

#######################
# Compare covariance matrices separately

mfvb.cov.list <- list()
lrvb.cov.list <- list()
mcmc.cov.list <- list()

for (sim.id in 1:20) {
  print(sprintf("Simulation %d", sim.id))
  my.dat <- ReadJSONFile(data.path, analysis.name, sim.id)

  # The order of these is
  # beta, tau, log_tau, nu, log_nu
  mfvb.cov.offdiag <- my.dat$mfvb_cov_main
  lrvb.cov.offdiag <- my.dat$lrvb_cov_main
  diag(mfvb.cov.offdiag) <- 0
  diag(lrvb.cov.offdiag) <- 0
  
  # Ignore nu
  k.tot <- my.dat$k_tot
  mfvb.cov.offdiag <- mfvb.cov.offdiag[1:(k.tot + 2), 1:(k.tot + 2)]
  lrvb.cov.offdiag <- lrvb.cov.offdiag[1:(k.tot + 2), 1:(k.tot + 2)]
  
  mfvb.cov.list[[length(mfvb.cov.list) + 1]] <- as.numeric(mfvb.cov.offdiag)  
  lrvb.cov.list[[length(mfvb.cov.list) + 1]] <- as.numeric(lrvb.cov.offdiag)
  
  d <- fromJSON(my.dat$df_json)
  
  reg.string <- paste(sprintf("x%d", 1:k.tot), collapse=" + ")
  mcmc.formula <- formula(sprintf("y ~ %s - 1", reg.string))
  
  prior <- list(B=list(V=my.dat$beta_prior_info_scale * diag(k.tot),
                       mu=rep(0, k.tot)),
                G=list(G1=list(V=1 / my.dat$nu_prior_gamma,
                               n=my.dat$nu_prior_alpha)),
                R=list(V=1 / my.dat$tau_prior_gamma,
                       n=my.dat$tau_prior_alpha))
  
  mcmc.res <- MCMCglmm(mcmc.formula,
                       random= ~ idv(z):re_ind,
                       rcov= ~ units,
                       data=d, prior=prior,
                       family="gaussian", pl=TRUE, pr=save.gamma, nitt=mcmc.iters,
                       verbose=FALSE)
    
  beta.draws <- mcmc.res$Sol[, 1:k.tot]
  tau.draws <- 1 / mcmc.res$VCV[, "units"]
  nu.draws <- 1 / mcmc.res$VCV[, "z.re_ind"]
  log.tau.draws <- log(tau.draws)
  log.nu.draws <- log(nu.draws)
  
  main.draws <- cbind(beta.draws, tau.draws, log.tau.draws)
  mcmc.cov.offdiag <- cov(main.draws)
  colnames(lrvb.cov.offdiag) <- rownames(lrvb.cov.offdiag) <- rownames(mcmc.cov.offdiag)
  
  mcmc.cov.offdiag
  lrvb.cov.offdiag
  
  diag(mcmc.cov.offdiag) <- 0
  mcmc.cov.list[[length(mfvb.cov.list) + 1]] <- as.numeric(mcmc.cov.offdiag)  
} 

mfvb.cov.df <- data.frame(do.call(rbind, mfvb.cov.list))
mfvb.cov.df$method <- "mfvb"
mfvb.cov.df$sim.id <- 1:nrow(mfvb.cov.df)

lrvb.cov.df <- data.frame(do.call(rbind, lrvb.cov.list))
lrvb.cov.df$method <- "lrvb"
lrvb.cov.df$sim.id <- 1:nrow(lrvb.cov.df)

mcmc.cov.df <- data.frame(do.call(rbind, mcmc.cov.list))
mcmc.cov.df$method <- "mcmc"
mcmc.cov.df$sim.id <- 1:nrow(mcmc.cov.df)

offdiag.cov.df <- rbind(mcmc.cov.df, lrvb.cov.df, mfvb.cov.df)

offdiag.cov.graph <- melt(offdiag.cov.df, id.vars = c("method", "sim.id")) %>%
  dcast(sim.id + variable ~ method)

ggplot(offdiag.cov.graph) +
  geom_point(aes(x=mcmc, y=lrvb, color="lrvb")) +
  geom_point(aes(x=mcmc, y=mfvb, color="mfvb")) +
  geom_abline(aes(slope=1, intercept=0), color="gray")
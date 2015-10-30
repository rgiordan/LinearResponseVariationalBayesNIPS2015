library(RcppEigen)
library(Rcpp)
library(Matrix)
library(mvtnorm)
library(ggplot2)
library(numDeriv)
library(reshape2)
library(dplyr)
library(coda)
library(digest)
library(gridExtra)
library(doParallel)
library(foreach)
library(optparse)
library(MVNMixtureLRVB)

setwd(file.path(Sys.getenv("GIT_REPO_LOC"),
                "LinearResponseVariationalBayesNIPS2015/",
                "variational_normal_mixture"))
source("simulate_mvn_mixture_lib.R")

option_list <- list(
  make_option(c("--cores"), type="integer", default=1,
              help="Number of cores to use [default %default]",
              metavar="cores")
)

# Get command line options.
opt <- parse_args(OptionParser(option_list=option_list))
registerDoParallel(cores = opt$cores)

kShowPlots <- FALSE
kSaveResults <- TRUE

#######
# Generate data

n <- 1e4
k <- 2
p <- 2
n.sims <- 100
burnin <- 200
n.gibbs.draws <- 6e3
vars.scale <- 0.5
anisotropy <- 1

analysis.name <- sprintf("n%d_k%d_p%d_sims%d_scale%0.1f_anisotropy%0.1f_%ddraws",
                         n, k, p, n.sims, vars.scale, anisotropy, n.gibbs.draws)

cov.list <- list()
sim.list <- list()
timing.list <- list()
sim.results <- list()
foreach.progress.file <- file(".foreach_progress", "w")
sim.results <- foreach(sim=1:n.sims) %dopar% {
  cat(file=foreach.progress.file, sim, " ", Sys.time(), "\n")

  par <- GenerateSampleParams(k=k, p=p, vars.scale=vars.scale,
                              anisotropy=anisotropy, random.rotation=FALSE)

  if (kShowPlots) {
    data <- GenerateMultivariateData(1e4, par$true.means, par$true.sigma, par$true.probs)
    df.col.names <- paste("X", 1:p, sep="")
    x <- data$x
    x.df <- data.frame(data$x)
    means.df <- data.frame(t(par$true.means))
    names(means.df) <- df.col.names
    x.df$label <- factor(data$component.labels)
    if (p >= 2) {
      ggplot() +
        geom_density2d(data=x.df, aes(x=X1, y=X2, group=label, color=label)) +
        geom_point(data=means.df, aes(x=X1, y=X2), color="red", size=5)
    } else if (p == 1) {
      ggplot() +
        geom_density(data=x.df, aes(x=X1, group=label, color=label, lwd=2)) +
        geom_vline(data=means.df, aes(xintercept=X1), color="red", size=1)
    }
  }

  results <- SimulateAndFitMVNMixture(n, k, p, par,
                                      fit.vb=TRUE, fit.gibbs=TRUE,
                                      n.gibbs.draws=n.gibbs.draws,
                                      burnin=burnin)
  CopyListToEnvironment(results, environment())

  ##################
  # Process results

  core.names <- CoreParameterNamesFromMLE(k, p)

  truth.means <- c(true.means, true.lambda, log(true.probs))

  core.vb.means <- CoreMeansFromVB(vb.optimum)
  core.vb.sd <- sqrt(CoreVarsFromVB(vb.optimum))

  lrvb.vars <- diag(lrvb.theta.cov)
  lrvb.vars.list <- UnpackVBThetaParameters(lrvb.vars, n, p, k)
  core.vb.indices <- GetCoreVBThetaIndices(n, p, k)
  core.vb.index.vec <- c(core.vb.indices$mu_indices,
                          core.vb.indices$lambda_indices,
                          core.vb.indices$log_pi_indices) + 1
  core.lrvb.vars <- c(lrvb.vars.list$e_mu, lrvb.vars.list$e_lambda, lrvb.vars.list$e_log_pi)
  core.lrvb.sd <- sqrt(core.lrvb.vars)
  core.lrvb.cov <- lrvb.theta.cov[core.vb.index.vec, core.vb.index.vec]

  sim.df <- data.frame(sim=sim,
                        var=core.names,
                        truth.mean=truth.means,
                        gibbs.mean=core.gibbs.results$means,
                        gibbs.sd=core.gibbs.results$sds,
                        gibbs.effsize=results$core.gibbs.effsize,
                        vb.mean=core.vb.means,
                        vb.sd=core.vb.sd,
                        lrvb.sd=core.lrvb.sd)

  timing.df <- data.frame(vb.time=as.numeric(results$vb.time, unit="secs"),
                          lrvb.time=as.numeric(results$lrvb.time, unit="secs"),
                          gibbs.time=as.numeric(results$gibbs.time, unit="secs"))
  sim.list[[sim]] <- sim.df
  timing.list[[sim]] <- timing.df
  cov.list[[sim]] <- list(lrvb.cov=core.lrvb.cov, gibbs.cov=results$core.gibbs.cov)
  return(list(timing.df=timing.df, sim.df=sim.df,
              lrvb.cov=core.lrvb.cov, gibbs.cov=results$core.gibbs.cov))
}
close(foreach.progress.file)
if (kSaveResults) {
  save(sim.list, timing.list, cov.list,
       sim.results,
       n, k, p, n.sims, vars.scale, anisotropy, par, n.gibbs.draws,
       file=paste("covariance_simulations_with_cov_", analysis.name, ".Rdata", sep=""))
}

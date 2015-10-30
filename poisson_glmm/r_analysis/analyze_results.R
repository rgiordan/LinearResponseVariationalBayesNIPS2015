# This reads in files produced by regression_poisson_mcmc_fit.R
# (which in turn processes the VB results from julia) and writes
# out a file called
#   data/<analysis name>_<analysis range>_theta_z_cov_results_compressed.csv
# ...which is read by knitr and produces a plots in the paper.

library(ggplot2)
library(dplyr)
library(reshape2)
library(jsonlite)
library(gridExtra)

base.dir <- file.path(Sys.getenv("GIT_REPO_LOC"),
                      "LinearResponseVariationalBayesNIPS2015/",
                      "poisson_glmm")
setwd(base.dir)

source(file.path(base.dir, "r_analysis/r_analysis_lib.R"))

data.path <- file.path(base.dir, "data")

analysis.name <- "poisson_glmm_z_theta_cov"
analysis.range <- "1_100"
full.cov <- FALSE

filename <- file.path(data.path, sprintf("%s_%s_results.csv", analysis.name, analysis.range))
results <- read.csv(filename, header=T)

results1 <- filter(results, measurement %in%  c("mean", "sd"),
                            variable %in% c("mu", "tau", "log_tau")) %>%
  dcast(sim.id + variable + measurement ~ method)

ggplot(filter(results1, measurement == "mean")) +
  geom_point(aes(x=mcmc, y=mfvb), size=3) +
  geom_abline(aes(intercept=0, slope=1)) +
  expand_limits(x=0, y=0) +
  geom_abline(aes(slope=1, intercept=0), color="gray") +
  facet_grid(~ variable)


ggplot(filter(results1, measurement == "sd", variable == "mu")) +
  geom_point(aes(x=mcmc, y=mfvb, color="mfvb"), size=3) +
  geom_point(aes(x=mcmc, y=lrvb, color="lrvb"), size=3) +
  geom_abline(aes(intercept=0, slope=1)) +
  expand_limits(x=0, y=0) +
  geom_abline(aes(slope=1, intercept=0), color="gray") +
  scale_color_discrete(name="Method:") +
  facet_grid(~ variable)


###############
# Off-diagonals

t.filename <- file.path(data.path, sprintf("%s_%s_theta_cov_results.csv", analysis.name, analysis.range))
t.cov.results <- read.csv(t.filename, header=T)

tz.filename <- file.path(data.path, sprintf("%s_%s_theta_z_cov_results.csv", analysis.name, analysis.range))
tz.cov.results <- read.csv(tz.filename, header=T)

t.cov.results.graph <- dcast(t.cov.results, sim.id + component ~ method, value.var="x")
tz.cov.results.graph <- dcast(tz.cov.results, sim.id + component ~ method, value.var="x")

grid.arrange(
  ggplot(tz.cov.results.graph) +
    geom_point(aes(x=mcmc, y=mfvb, color="mfvb"), size=3) +
    geom_point(aes(x=mcmc, y=lrvb, color="lrvb"), size=3) +
    geom_abline(aes(intercept=0, slope=1)) +
    expand_limits(x=0, y=0) +
    geom_abline(aes(slope=1, intercept=0), color="gray") +
    ggtitle("Main (beta, tau, log tau) off-diagonal covariance")
,  
  ggplot(tz.cov.results.graph) +
    geom_point(aes(x=mcmc, y=lrvb, color="lrvb"), size=3) +
    geom_abline(aes(intercept=0, slope=1)) +
    expand_limits(x=0, y=0) +
    geom_abline(aes(slope=1, intercept=0), color="gray") +
    ggtitle("Main (beta, tau, log tau) covariance with z")

 , ncol=2 
)

# I want to use tz.cov.results.graph in the paper but don't want to use
# all the rows since the file is unnecessarily large.  It will suffice
# to plot only a few simulations.

nrow(tz.cov.results.graph)
tz.cov.results.graph.compressed <- sample_n(tz.cov.results.graph, 20000)
ggplot(tz.cov.results.graph.compressed) +
  geom_point(aes(x=mcmc, y=lrvb, color="lrvb"), size=3) +
  geom_abline(aes(intercept=0, slope=1)) +
  expand_limits(x=0, y=0) +
  geom_abline(aes(slope=1, intercept=0), color="gray") +
  ggtitle("Main (beta, tau, log tau) covariance with z")

tz.compressed.filename <-
  file.path(data.path, sprintf("%s_%s_theta_z_cov_results_compressed.csv",
                               analysis.name, analysis.range))
write.csv(tz.cov.results.graph.compressed, file=tz.compressed.filename,
          quote=F, row.names=F)


# See how the difference depends on epsilon.
epsilon.df <- filter(results, (variable == "mu" & measurement == "sd") |
                              (variable == "tau" & measurement == "mean"),
                               method %in% c("mfvb", "lrvb")) %>%
  dcast(sim.id ~ measurement + variable + method)

ggplot(epsilon.df) +
  geom_point(aes(x=mean_tau_mfvb, y=sd_mu_lrvb - sd_mu_mfvb)) +
  expand_limits(x=0, y=0)

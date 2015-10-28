library(ggplot2)
library(dplyr)
library(reshape2)

data.path <- file.path(Sys.getenv("GIT_REPO_LOC"), "variational_bayes/linear_regression/data")
analysis.name <- "re_regression_full_cov_final"
filename <- file.path(data.path, sprintf("%s.csv", analysis.name))

results <- read.csv(filename, header=T)

results1 <- filter(results, measurement %in%  c("mean", "sd")) %>%
  dcast(sim.id + variable + measurement + component ~ method)

ggplot(filter(results1, measurement == "mean")) +
  geom_point(aes(x=truth, y=mfvb, color="vb"), size=3) +
  geom_point(aes(x=truth, y=mcmc, color="mcmc"), size=3) +
  geom_abline(aes(intercept=0, slope=1)) +
  expand_limits(x=0, y=0) +
  geom_abline(aes(slope=1, intercept=0), color="gray") +
  facet_grid(~ variable)

ggplot(filter(results1, measurement == "sd")) +
  #geom_point(aes(x=mcmc, y=mfvb, color="mfvb"), size=3) +
  geom_point(aes(x=mcmc, y=lrvb, color="lrvb"), size=3) +
  geom_abline(aes(intercept=0, slope=1)) +
  expand_limits(x=0, y=0) +
  geom_abline(aes(slope=1, intercept=0), color="gray") +
  facet_grid(~ variable)

results2 <- filter(results, measurement == "cov") %>%
  dcast(sim.id + variable + measurement + component ~ method)

ggplot(filter(results2, grepl("beta", variable), grepl("nu", variable))) +
  geom_point(aes(x=mcmc, y=mfvb, color="mfvb"), size=3) +
  geom_point(aes(x=mcmc, y=lrvb, color="lrvb"), size=3) +
  geom_abline(aes(intercept=0, slope=1)) +
  expand_limits(x=0, y=0) +
  geom_abline(aes(slope=1, intercept=0), color="gray") +
  facet_grid(~ variable)

ggplot(filter(results2, grepl("beta", variable), grepl("tau", variable))) +
  geom_point(aes(x=mcmc, y=mfvb, color="mfvb"), size=3) +
  geom_point(aes(x=mcmc, y=lrvb, color="lrvb"), size=3) +
  geom_abline(aes(intercept=0, slope=1)) +
  expand_limits(x=0, y=0) +
  geom_abline(aes(slope=1, intercept=0), color="gray") +
  facet_grid(~ variable)

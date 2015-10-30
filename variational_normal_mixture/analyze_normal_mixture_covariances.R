library(RcppEigen)
library(Rcpp)
library(Matrix)
library(ggplot2)
library(reshape2)
library(dplyr)
library(gridExtra)
library(xtable)
library(MVNMixtureLRVB)

setwd(file.path(Sys.getenv("GIT_REPO_LOC"),
                "LinearResponseVariationalBayesNIPS2015/",
                "variational_normal_mixture"))

kSaveResults <- FALSE
filename.base <-
  paste("covariance_simulations_with_cov_",
        "n10000_k2_p2_sims200_scale0.5_anisotropy1.0_10000draws", sep="")

load(file.path("data", sprintf("%s.Rdata", filename.base)))
analysis.metadata <- list()

# This is for foreach output
GetSimDf <- function(i) {
  sim.results[[i]]$sim.df
}
GetTimingDf <- function(i) {
  sim.results[[i]]$timing.df
}
core.vars <- do.call(rbind, lapply(1:length(sim.results), GetSimDf))

gibbs.effsize <- core.vars[c("var", "sim", "gibbs.effsize")]
min.eff.sizes <-
  group_by(gibbs.effsize, sim) %>%
  summarize(min.effsize=min(gibbs.effsize))

analysis.metadata$min.effsize <- 500

good.sims <- filter(min.eff.sizes, min.effsize > analysis.metadata$min.effsize)$sim
analysis.metadata$good.sims  <- length(good.sims)
core.vars <- core.vars[names(core.vars) != "gibbs.effsize"]
core.vars <- filter(core.vars, sim %in% good.sims)

core.vars.melt <- melt(core.vars, id.vars = c("var", "sim"))
core.vars.melt$measure <- sub("^.*\\.", "", core.vars.melt$variable)
core.vars.melt$method <- sub("\\..*$", "", core.vars.melt$variable)
core.vars.melt$parameter <- sub("\\_.*$", "", core.vars.melt$var)

core.vars.df <- dcast(core.vars.melt, sim + var + parameter + measure ~ method)

this.parameter <- "mu"
grid.arrange(
  ggplot(filter(core.vars.df, parameter == this.parameter, measure == "mean")) +
    geom_point(aes(x=truth, y=vb, color="vb"), size=3) +
    geom_point(aes(x=truth, y=gibbs, color="gibbs"), size=3) +
    ggtitle(paste(this.parameter, "point estimates")) +
    xlab("Truth") + ylab("estimates") +
    expand_limits(x=0, y=0) +
    geom_abline(aes(slope=1, intercept=0), color="gray") +
    geom_hline(aes(yintercept=0), color="gray") +
    geom_vline(aes(xintercept=0), color="gray"),

  ggplot(filter(core.vars.df, parameter == this.parameter, measure == "sd")) +
    geom_point(aes(x=gibbs, y=vb, color="vb"), size=3) +
    geom_point(aes(x=gibbs, y=lrvb, color="lrvb"), size=3) +
    ggtitle(paste(this.parameter, "standard deviations")) +
    xlab("Gibbs Stdev") + ylab("estimates") +
    expand_limits(x=0, y=0) +
    geom_abline(aes(slope=1, intercept=0), color="gray") +
    geom_hline(aes(yintercept=0), color="gray") +
    geom_vline(aes(xintercept=0), color="gray"),
  nrow=1)



# Cov plot
GetCovDf <- function(i) {
  x <- sim.results[[i]]
  cov.size <- nrow(x$lrvb.cov)
  cov.names <- outer(1:cov.size, 1:cov.size, function(x, y) { paste(x, y, sep=".")})
  cov.df <- data.frame(t(c(x$gibbs.cov, x$lrvb.cov)))
  names(cov.df) <- c(paste("gibbs", cov.names, sep="_"),
                     paste("lrvb", cov.names, sep="_"))
  return(cov.df)
}

raw.covs.df <- do.call(rbind, lapply(1:length(sim.results), GetCovDf))
p.diag <- sqrt(ncol(raw.covs.df) / 3)
diag.cols <- as.logical(diag(p.diag))
diag.covs.df <- raw.covs.df[, diag.cols]
offdiag.covs.df <- raw.covs.df[, !diag.cols]

covs.df <- offdiag.covs.df
covs.df$sim <- 1:nrow(covs.df)
covs.df <- filter(covs.df, sim %in% good.sims)
covs.df.melt <- melt(covs.df, id.vars="sim")

covs.df.melt$parameter <- sub("^.*_", "", covs.df.melt$variable)
covs.df.melt$method <- sub("_.*$", "", covs.df.melt$variable)
core.covs.df <- dcast(covs.df.melt, sim + parameter ~ method)

ggplot(core.covs.df) +
  geom_point(aes(x=gibbs, y=lrvb, color="lrvb"), size=2) +
  xlab("Gibbs off-diagonal covariance") + ylab("estimates") +
  expand_limits(x=0, y=0) +
  geom_abline(aes(slope=1, intercept=0), color="gray") +
  geom_hline(aes(yintercept=0), color="gray") +
  geom_vline(aes(xintercept=0), color="gray")

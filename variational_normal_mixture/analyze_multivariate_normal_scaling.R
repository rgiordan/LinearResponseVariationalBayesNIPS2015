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
library(MVNMixtureLRVB)

setwd(file.path(Sys.getenv("GIT_REPO_LOC"),
                "LinearResponseVariationalBayesNIPS2015/",
                "variational_normal_mixture"))
source("simulate_mvn_mixture_lib.R")

SafeGetNumericTiming <- function(results, field) {
  return(ifelse(is.null(results[[field]]), NA,
                as.numeric(results[[field]], unit="secs")))
}

GetTimingDf <- function(results) {
  gibbs.time <- SafeGetNumericTiming(results, "gibbs.time")
  mle.time <- SafeGetNumericTiming(results, "mle.time")
  mle.hess.time <- SafeGetNumericTiming(results, "gibbs.time")
  gibbs.effsize <- NA
  if (!any(is.null(results$core.gibbs.effsize))) {
    gibbs.effsize <- min(results$core.gibbs.effsize)
  }
  data.frame(n=results$n, p=results$p, k=results$k,
             vb.time=as.numeric(results$vb.time, unit="secs"),
             lrvb.time=as.numeric(results$lrvb.time, unit="secs"),
             mle.time=mle.time,
             mle.hess.time=mle.hess.time,
             gibbs.time=gibbs.time,
             gibbs.effsize=min(gibbs.effsize))
}

GetAllTimingDfs <- function(x) {
  do.call(rbind, lapply(x, GetTimingDf))
}

kSaveResults <- FALSE
load("new_scaling_p.Rdata")

timing.results <- group_by(timing.results, n, p, k) %>%
                  summarize(vb.time=mean(vb.time), lrvb.time=mean(lrvb.time),
                            mle.time=mean(mle.time), mle.hess.time=mean(mle.hess.time),
                            gibbs.time=mean(gibbs.time), gibbs.effsize=mean(gibbs.effsize))
timing.results$lrvb.proportion <- with(timing.results, lrvb.time / vb.time)
timing.results$gibbs.scaled.time <- with(timing.results,
                                         gibbs.time * (1000 / gibbs.effsize))
timing.melt <- melt(timing.results, id.vars=(c("n", "p", "k")))
timing.melt$variable <- sub(".time$", "", timing.melt$variable)
timing.melt <- timing.melt[!is.na(timing.melt$value), ]

if (kSaveResults) {
  write.csv(timing.melt, file="scaling_simulation_very_high_p.csv",
            quote=FALSE, row.names=FALSE)
}

# Plot the times for the covarainces and gibbs scaled by effective sample size.
method.vars <- c("lrvb", "gibbs.scaled")
method.labels <- c("LRVB", "Re-scaled Gibbs")

line.label <- "y=x"
ggplot(filter(timing.melt, variable %in% method.vars)) +
  geom_point(aes(x=n, y=value, color=variable, linetype=ordered(p)), size=3) +
  geom_line(aes(x=n, y=value, color=variable, linetype=ordered(p))) +
  scale_color_discrete(name="Method:", limits=c(method.vars, line.label),
                       labels=c(method.labels, line.label)) +
  geom_abline(aes(intercept=-5, slope=1, color=line.label), lwd=2) +
  scale_linetype_discrete(name="Problem dimension (P)") +
  xlab("log10 number of data points (N)") + ylab("Running time (log10 seconds)") +
  scale_x_log10() + scale_y_log10() +
  theme(legend.position="top")

ggplot(filter(timing.melt, variable %in% method.vars)) +
  geom_point(aes(x=p, y=value, color=variable, linetype=ordered(n)), size=3) +
  geom_line(aes(x=p, y=value, color=variable, linetype=ordered(n))) +
  scale_color_discrete(name="Method:", limits=method.vars,
                       labels=method.labels) +
  scale_linetype_discrete(name="Number of data points (N)(N)") +
  geom_abline(aes(intercept=-0.8, slope=2.8, color=line.label), lwd=1) +
  xlab("log10 dimension of problem (P)") + ylab("Running time (log10 seconds)") +
  scale_x_log10() + scale_y_log10() +
  theme(legend.position="top")

summary(lm(log10(value) ~ log10(n) + factor(p),
           filter(timing.melt, variable == "lrvb")))

# Use one of these depending on whether you've varied n in the simulation.
summary(lm(log10(value) ~ log10(p) + factor(n),
           filter(timing.melt, variable == "lrvb")))

summary(lm(log10(value) ~ log10(p),
           filter(timing.melt, variable == "lrvb", p > 5)))

summary(lm(log10(value) ~ log10(p) + factor(n) + factor(k),
           filter(timing.melt, variable == "gibbs.scaled")))
summary(lm(log10(value) ~ log10(p) + factor(n) + factor(k),
           filter(timing.melt, variable == "mle.hess")))
summary(lm(value ~ n + factor(p), filter(timing.melt, variable == "lrvb")))

# Plot the raw times including optimization and unnormalized for effective sample size.
#method.vars <- c("vb", "mle", "gibbs")
#method.labels <- c("VB", "MAP", "Gibbs")

# method.vars <- c("vb", "gibbs")
# method.labels <- c("VB", "Gibbs")

ggplot(filter(timing.melt, variable %in% method.vars)) +
  geom_point(aes(x=n, y=value, color=variable, linetype=ordered(p)), size=3) +
  geom_line(aes(x=n, y=value, color=variable, linetype=ordered(p))) +
  scale_color_discrete(name="Method:", limits=method.vars,
                       labels=method.labels) +
  scale_linetype_discrete(name="Problem dimension (P)") +
  xlab("log10 number of data points (N)") + ylab("Running time (log10 seconds)") +
  scale_x_log10(breaks=unique(timing.melt$n)) + scale_y_log10() +
  theme(legend.position="top")

ggplot(filter(timing.melt, variable %in% method.vars)) +
  geom_point(aes(x=p, y=value, color=variable, linetype=ordered(n)), size=3) +
  geom_line(aes(x=p, y=value, color=variable, linetype=ordered(n))) +
  scale_color_discrete(name="Method:", limits=method.vars,
                       labels=method.labels) +
  scale_linetype_discrete(name="Number of data points (N)(N)") +
  xlab("log10 dimension of problem (P)") + ylab("Running time (log10 seconds)") +
  scale_x_log10(breaks=unique(timing.melt$p)) + scale_y_log10() +
  geom_abline(aes(slope=3.44682, intercept=-3.31584))  +
  theme(legend.position="top")

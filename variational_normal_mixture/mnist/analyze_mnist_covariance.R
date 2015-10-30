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

# In fact, I just want to see what's in the knitr data.  Lame, I know.

load("/home/rgiordan/Documents/git_repos/variational_bayes/writing/nips_2015/data/mnist_covariance_comparison.Rdata")
mnist.test.accuracy <- sum(diag(test.results)) / sum(test.results)

ggplot(filter(mnist.core.vars.df, parameter == "lambda", measure == "sd")) +
  geom_point(aes(x=gibbs, y=lrvb, color="LRVB"), size=2) +
  geom_point(aes(x=gibbs, y=vb, color="MFVB"), size=2) +
  xlab("Gibbs std dev") + ylab("estimates") +
  expand_limits(x=0, y=0) +
  geom_abline(aes(slope=1, intercept=0), color="gray") +
  geom_hline(aes(yintercept=0), color="gray") +
  geom_vline(aes(xintercept=0), color="gray") +
  scale_color_discrete(name="Method:") +
  ggtitle(expression(Lambda)) +
  theme(legend.position="none")
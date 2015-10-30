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

base.dir <- file.path(Sys.getenv("GIT_REPO_LOC"),
                      "variational_bayes/variational_normal_mixture/")
setwd(file.path(base.dir, "mnist"))

kSaveCovarianceResults <- FALSE
kSaveDataForKnitr <- FALSE

#######
# Load the data

if (F) {
  analysis.name <- "test.data.50"
  load("mnist_50_features.Rdata")
  kmeans.seed <- 42
}
if (T) {
  analysis.name <- "full.data.25"
  load("full_mnist_25_features.Rdata")  
  kmeans.seed <- 43
}


digits <- c(0, 1); kmeans.seed <- 43
#digits <- c(1, 7)
digits.name <- paste(sort(digits), collapse="_")
x <- feat.mat[labels %in% digits, ]
true.labels <- labels[labels %in% digits]

test.x <- test.feat.mat[test.labels %in% digits, ]
test.true.labels <- test.labels[test.labels %in% digits]

stopifnot(ncol(test.x) == ncol(x))
stopifnot(nrow(test.x) == length(test.true.labels))
stopifnot(nrow(x) == length(true.labels))

n <- nrow(x)
k <- length(digits)
p <- ncol(x)
priors <- GenerateSamplePriors(x=x, k=k)

if (F) {
  # Warm start with kmeans
  set.seed(kmeans.seed) # Not every starting point is good...
  initial.kmeans <- kmeans(x, k)
  starting.means <- t(initial.kmeans$centers)
  starting.components <- initial.kmeans$cluster  
} else {
  # Or with the labels
  starting.means <- t(aggregate(x, by=list(true.labels), mean)[, -1])
  starting.components <- true.labels
}

starting.sigma <- lapply(1:k, function(k) { cov(x[starting.components == k, , drop=F]) })  
starting.lambda <- InvertLinearizedMatrices(VectorizeMatrixList(starting.sigma))
starting.log.det.lambda <- LogDeterminantOfLinearizedMatrices(starting.lambda)
starting.z <- matrix(0, n, k)
unique.starting.components <- unique(starting.components)
for (this.k in 1:k) {
  starting.z[starting.components == unique.starting.components[this.k],
             this.k] <- as.numeric(1) + 0.1
}
starting.z <- starting.z / rowSums(starting.z)
starting.probs <- colMeans(starting.z)

####################
# Compute the VB fit

fit.mu <- TRUE
fit.pi <- TRUE
fit.lambda <- TRUE
vb.opt.time <- Sys.time()
vb.optimum <-
  GetVariationalSolution(x=x,
                         e.mu=starting.means,
                         e.lambda=starting.lambda,
                         e.log.det.lambda=starting.log.det.lambda,
                         e.p=starting.probs,
                         e.log.pi=log(starting.probs),
                         fit.mu=fit.mu, fit.lambda=fit.lambda, fit.pi=fit.pi,
                         e.z=starting.z,
                         priors=priors, tolerance=1e-9)
vb.opt.time <- Sys.time() - vb.opt.time
# Check to make sure the classification is reasonable.
# If not, change kmeans.seed.  Later this will have to be diagonal.
class <- max.col(vb.optimum$e.z)
table(true.labels, class)

# Check it on the testing data.
test.n <- nrow(test.x)
test.z <- GetZMatrix(x=test.x,
                     e_mu=vb.optimum$e.mu, e_mu2=vb.optimum$e.mu2,
                     e_lambda=vb.optimum$e.lambda,
                     e_log_det_lambda=vb.optimum$e.log.det.lambda,
                     e_log_pi=vb.optimum$e.log.pi)
test.class <- max.col(test.z)
test.results <- table(test.true.labels, test.class)
sum(diag(test.results)) / sum(test.results)


lrvb.file <- sprintf("%s_%s_lrvb_theta_cov.Rdata", digits.name, analysis.name)
if (!file.exists(lrvb.file)) {
  lrvb.time <- Sys.time()
  theta.cov <- GetThetaCovariance(e.mu=vb.optimum$e.mu, e.mu2=vb.optimum$e.mu2,
                                  lambda.par=vb.optimum$lambda.par, n.par=vb.optimum$n.par,
                                  pi.par=vb.optimum$pi.par,
                                  fit.mu=fit.mu, fit.lambda=fit.lambda, fit.pi=fit.pi)

  lrvb.correction <- GetLRVBCorrectionTerm(x=x,
                                           e_mu=vb.optimum$e.mu,
                                           e_mu2=vb.optimum$e.mu2,
                                           e_lambda=vb.optimum$e.lambda,
                                           e_z=vb.optimum$e.z,
                                           theta_cov=theta.cov, verbose=TRUE)
  lrvb.theta.cov <- CPPGetLRVBCovarianceFromCorrection(lrvb_correction=lrvb.correction,
                                                       theta_cov=theta.cov,
                                                       verbose=TRUE)
  lrvb.time <- Sys.time() - lrvb.time
  
  # ~ 7 minutes on my laptop depending on p and n
  save(theta.cov, lrvb.theta.cov, lrvb.correction, lrvb.time, vb.opt.time, file=lrvb.file)
  
  
} else {
  load(lrvb.file)
}




##################
# Compute a Gibbs fit

library(bayesm)

burnin <- 200
gibbs.time <- Sys.time()
# Fit the normal mixture
prior.list <- list(ncomp=k)
out <- rnmixGibbs(Data=list(y=x),
                  Prior=prior.list,
                  Mcmc=list(R=2000 + burnin, keep=1))
print(gibbs.time <- Sys.time() - gibbs.time)

# Get a permutation that matches vb
mu.sizes <- colMeans(GetFieldsFromnNmix(out$nmix,
                                        GetMuFieldSizeFromCompdraw,
                                        1:k, p)[-(1:burnin), ])
vb.mu.sizes <- colSums(vb.optimum$e.mu^2)
mu.size.diffs <- outer(mu.sizes, vb.mu.sizes, function(x, y) { abs(x - y)})
k.vector <- apply(mu.size.diffs, 1, which.min)
if (length(unique(k.vector)) != k) {
  print("Error - not a unique mean size matching.")
}

original.mu.draws <- GetFieldsFromnNmix(out$nmix,GetMuFieldFromCompdraw, k.vector, p)
original.lambda.draws <- GetFieldsFromnNmix(out$nmix, GetLambdaFieldFromCompdraw, k.vector, p)
original.pi.draws <- out$nmix$probdraw[, k.vector]

original.mu.draws <- GetFieldsFromnNmix(out$nmix, GetMuFieldFromCompdraw, k.vector, p)
original.lambda.draws <- GetFieldsFromnNmix(out$nmix, GetLambdaFieldFromCompdraw, k.vector, p)
original.pi.draws <- out$nmix$probdraw[, k.vector]

mu.draws <- data.frame(original.mu.draws[-(1:burnin),])
names(mu.draws) <- GetMatrixVectorNames("mu", p=p, k=k)

lambda.draws <- data.frame(original.lambda.draws[-(1:burnin),])
names(lambda.draws) <- GetSymmetricMatrixVectorNames("lambda", p, k)

pi.draws <- data.frame(original.pi.draws[-(1:burnin),])
names(pi.draws) <- paste("pi", 1:k, sep="_")

core.gibbs.results <- CoreParametersFromGibbs(mu.draws, lambda.draws, pi.draws)
core.gibbs.effsize <- c(effectiveSize(mu.draws),
                        effectiveSize(lambda.draws),
                        effectiveSize(pi.draws))

if (F) {
  mu.draws$draw <- 1:nrow(mu.draws)
  pi.draws$draw <- 1:nrow(pi.draws)
  lambda.draws$draw <- 1:nrow(lambda.draws)
  
  ggplot(melt(mu.draws, id.var="draw")) +
    geom_point(aes(x=draw, y=value, color=variable)) +
    facet_grid(~ variable)
  ggplot(melt(lambda.draws, id.var="draw")) +
    geom_point(aes(x=draw, y=value, color=variable))
  ggplot(melt(pi.draws, id.var="draw")) +
    geom_point(aes(x=draw, y=value, color=variable))  
}


################### Compare


core.names <- CoreParameterNamesFromMLE(k, p)

core.vb.means <- CoreMeansFromVB(vb.optimum)
core.vb.sd <- sqrt(CoreVarsFromVB(vb.optimum))

lrvb.vars <- diag(lrvb.theta.cov)
lrvb.vars.list <- UnpackVBThetaParameters(lrvb.vars, n, p, k)
core.lrvb.vars <- c(lrvb.vars.list$e_mu, lrvb.vars.list$e_lambda, lrvb.vars.list$e_log_pi)
core.lrvb.sd <- sqrt(core.lrvb.vars)


core.vars <- data.frame(var=core.names,
                     gibbs.mean=core.gibbs.results$means,
                     gibbs.sd=core.gibbs.results$sds,
                     gibbs.effsize=core.gibbs.effsize,
                     vb.mean=core.vb.means,
                     vb.sd=core.vb.sd,
                     lrvb.sd=core.lrvb.sd)


core.vars.melt <- melt(core.vars, id.vars = c("var"))
core.vars.melt$measure <- sub("^.*\\.", "", core.vars.melt$variable)
core.vars.melt$method <- sub("\\..*$", "", core.vars.melt$variable)
core.vars.melt$parameter <- sub("\\_.*$", "", core.vars.melt$var)
core.vars.df <- dcast(core.vars.melt, var + parameter + measure ~ method)

this.parameter <- "lambda"
grid.arrange(
  ggplot(filter(core.vars.df, parameter == this.parameter, measure == "mean")) +
    geom_point(aes(x=gibbs, y=vb, color="vb"), size=3) + 
    ggtitle(paste(this.parameter, "point estimates")) +
    xlab("Gibbs") + ylab("estimates") +
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

mnist.core.vars.df <- core.vars.df


if (kSaveCovarianceResults) {
  cov.file <- sprintf("mnist_covariance_comparison_%s_%s.Rdata", digits.name, analysis.name)
  save(mnist.core.vars.df,
       test.n, n, p, k,
       test.results, digits,
       file=cov.file)  
}




################
# Get the leverage scores


leverage.file <- sprintf("%s_leverage_results.Rdata", analysis.name)

if (!file.exists(leverage.file)) {
  
  # Pick some indices with different concentrations in
  # the components.
  z.sum.squares <- apply(vb.optimum$e.z, MARGIN=1, function(x) { sum(x^2) })
  which(abs(z.sum.squares - 0.9) < 0.01)
  set.seed(137)
  x.ind.95 <- which(abs(z.sum.squares - 0.95) < 0.01)
  x.ind.8 <- which(abs(z.sum.squares - 0.8) < 0.01)
  x.ind.7 <- which(abs(z.sum.squares - 0.7) < 0.01)
  x.ind.6 <- which(abs(z.sum.squares - 0.6) < 0.01)
  x.ind.5 <- which(abs(z.sum.squares - 0.5) < 0.01)
  
  x.indices <- c(x.ind.95[1], x.ind.8[1], x.ind.7[1], x.ind.6[1], x.ind.5[1])
  
  # Get the leverage scores.
  vb.leverage.time <- Sys.time()
  # x.indices is zero-based for C!
  x.cov <- GetXVarianceSubset(x=x, x.indices - 1)
  htx <- GetHThetaXSubset(e_z=vb.optimum$e.z, e_mu=vb.optimum$e.mu,
                          e_lambda=vb.optimum$e.lambda, x.indices - 1)
  hzx <- GetHZXSubset(n_tot=n, e_mu=vb.optimum$e.mu,
                      e_lambda=vb.optimum$e.lambda, x.indices - 1)
  
  htz <- GetHThetaZ(x=x, e_mu=vb.optimum$e.mu,
                    e_mu2=vb.optimum$e.mu2,
                    e_lambda=vb.optimum$e.lambda)
  z.cov <- GetZCovariance(z_mat=vb.optimum$e.z)
  
  # Note that, while more robust, this takes more time since it has to
  # re-do the inversion every time.
  tx.cov <- CPPGetLeverageScores(z_cov=z.cov, x_cov=x.cov, htx=htx, htz=htz, hzx=hzx,
                                 lrvb_correction=lrvb.correction, theta_cov=theta.cov,
                                 verbose=TRUE)
  
  vb.leverage.time <- Sys.time() - vb.leverage.time
  
  ##############
  # Numerically calculate sensitivities
  delta <- 0.01
  
  mu.diff.list <- list()
  log.pi.diff.list <- list()
  lambda.diff.list <- list()
  mu.effect.list <- list()
  log.pi.effect.list <- list()
  lambda.effect.list <- list()
  perturbation.list <- list()
  iter <- 0
  
  perturbation.time <- Sys.time()
  small.p <- p
  
  for (x.col in 1:small.p) {
    for (x.index in 1:length(x.indices)) {
      x.row <- x.indices[x.index]
      
      iter <- iter + 1
      print(sprintf("Index %d, component %d", x.index, x.col))
      new.x <- x
      new.x[x.row, x.col] <- new.x[x.row, x.col] + delta
      
      # Note that x_n and x_p are zero-indexed in GetXCoordinate.
      linearized.x.index <- GetXCoordinate(x.index - 1, x.col - 1, length(x.indices), p) + 1
      raw.effect <- UnpackVBThetaParameters(tx.cov[, linearized.x.index] * delta, n, p, k)
      
      new.vb.optimum <-
        GetVariationalSolution(x=new.x,
                               e.mu=vb.optimum$e.mu,
                               e.mu2=vb.optimum$e.mu2,
                               e.lambda=vb.optimum$e.lambda,
                               e.log.det.lambda=vb.optimum$e.log.det.lambda,
                               e.pi=vb.optimum$e.pi,
                               e.log.pi=vb.optimum$e.log.pi,
                               fit.mu=fit.mu, fit.lambda=fit.lambda, fit.pi=fit.pi,
                               e.z=vb.optimum$e.z,
                               priors=priors, tolerance=1e-9, quiet=TRUE)
      
      mu.diff.list[[iter]] <- as.numeric(new.vb.optimum$e.mu - vb.optimum$e.mu) 
      log.pi.diff.list[[iter]] <- as.numeric(new.vb.optimum$e.log.pi - vb.optimum$e.log.pi) 
      lambda.diff.list[[iter]] <- as.numeric(new.vb.optimum$e.lambda - vb.optimum$e.lambda) 
      
      mu.effect.list[[iter]] <- as.numeric(raw.effect$e_mu)
      log.pi.effect.list[[iter]] <- as.numeric(raw.effect$e_log_pi)
      lambda.effect.list[[iter]] <- as.numeric(raw.effect$e_lambda)
      
      perturbation.list[[iter]] <- c(x.row, x.col)
    }
  }
  print(perturbation.time <- Sys.time() - perturbation.time)
  
  save(tx.cov, x.indices, vb.leverage.time, perturbation.time,
       perturbation.list, small.p,
       mu.diff.list, log.pi.diff.list, lambda.diff.list,
       mu.effect.list, log.pi.effect.list, lambda.effect.list,
       file=leverage.file)
} else {
  load(leverage.file)
}


print((p / small.p) *
        as.numeric(perturbation.time, units="secs") /
        as.numeric(vb.leverage.time, units="secs"))

perturbations <- data.frame(do.call(rbind, perturbation.list))
names(perturbations)  <- c("row", "col")

mu.diff <- do.call(rbind, mu.diff.list)
log.pi.diff <- do.call(rbind, log.pi.diff.list)
lambda.diff <- do.call(rbind, lambda.diff.list)

mu.effect <- do.call(rbind, mu.effect.list)
log.pi.effect <- do.call(rbind, log.pi.effect.list)
lambda.effect <- do.call(rbind, lambda.effect.list)

if (kSaveDataForKnitr) {
  # Save a small Rdata file to make the report graphs.
  save(file="mnist_perturbation_data.Rdata",
       n, p, k, x.indices,
       test.results, digits, small.p, delta, test.n,
       perturbation.time, vb.leverage.time, lrvb.time, vb.opt.time,
       mu.diff, log.pi.diff, lambda.diff,
       mu.effect, log.pi.effect, lambda.effect)
}


# Graphs
base.title <- sprintf("\nn=%d, k=%d, p=%d", n, k, p)
dataset.name <- sprintf("MNIST")
grid.arrange(
  ggplot() +
    geom_point(aes(x=as.numeric(mu.diff), y=as.numeric(mu.effect)), size=2) +
    xlab("Actual change") + ylab("Leverage score") +
    geom_abline(aes(slope=1, intercept=0)) +
    ggtitle(paste(dataset.name,  "dataset\nMu leverage scores", base.title)),
  ggplot() +
    geom_point(aes(x=as.numeric(log.pi.diff), y=as.numeric(log.pi.effect)), size=2) +
    xlab("Actual change") + ylab("Leverage score") +
    geom_abline(aes(slope=1, intercept=0)) +
    ggtitle(paste(dataset.name, "dataset\nLog pi leverage scores", base.title)),
  ggplot() +
    geom_point(aes(x=as.numeric(lambda.diff), y=as.numeric(lambda.effect)), size=2) +
    xlab("Actual change") + ylab("Leverage score") +
    geom_abline(aes(slope=1, intercept=0)) +
    ggtitle(paste(dataset.name, "dataset\nLambda leverage scores", base.title)),
  ncol=3
)


# This was only for debugging.
# Mu
mu.effect.df <- data.frame(mu.effect)
names(mu.effect.df) <- GetMatrixVectorNames("lrvb_mu", p, k)

mu.diff.df <- data.frame(mu.diff)
names(mu.diff.df) <- GetMatrixVectorNames("diff_mu", p, k)

mu.df <- cbind(mu.effect.df, mu.diff.df, perturbations)
mu.df.melt <- melt(mu.df, id.vars=c("row", "col"))
head(mu.df.melt)
mu.df.melt$method <- sub("_.*$", "", mu.df.melt$variable)
mu.df.melt$parameter <- sub("^[^_]*_", "", mu.df.melt$variable)
mu.df.cast <- dcast(mu.df.melt, row + col + parameter ~ method)

ggplot(mu.df.cast) + 
  geom_point(aes(x=diff, y=lrvb, color=parameter)) +
  geom_abline(aes(slope=1, intercept=0)) +
  facet_grid(row ~ .)

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
library(bayesm)
library(MVNMixtureLRVB)


DefaultPriors <- function(p, k, prior.obs=1, x.prior.scale=rep(10, p)) {
  # Set some reasonably diffuse default priors.
  mu.prior.mean  <- matrix(0, nrow=p, ncol=k)
  if (p > 1) {
    mu.prior.info.mat <- diag(x=prior.obs / (x.prior.scale ^ 2))    
  } else {
    # Seriously, R?  diag() of a single number is a 0x0 matrix.
    mu.prior.info.mat <- matrix(prior.obs / (x.prior.scale ^ 2))
  }
  matrix.size <- (p * (p + 1)) / 2
  mu.prior.info <- matrix(ConvertSymmetricMatrixToVector(mu.prior.info.mat),
                          nrow=matrix.size, ncol=k)
  
  # The lambda prior.
  lambda.prior.n <- rep(prior.obs, k)
  lambda.prior.v.inv.list <- list()
  for (this.k in 1:k) {
    lambda.prior.v.inv.list[[this.k]] <- prior.obs * diag(x.prior.scale ^ 2)
  }
  lambda.prior.v.inv <- VectorizeMatrixList(lambda.prior.v.inv.list)
  
  # More alpha prior mass helps with stability (especially for MCMC).
  p.prior.alpha <- rep(5 * prior.obs, k)
  priors <- list(mu.prior.mean=mu.prior.mean, mu.prior.info=mu.prior.info,
                 lambda.prior.v.inv=lambda.prior.v.inv, lambda.prior.n=lambda.prior.n,
                 p.prior.alpha=p.prior.alpha)
  return(priors)
}


SimulateAndFitMVNMixture <- function(n, k, p, par,
                                     fit.vb=TRUE, fit.gibbs=TRUE,
                                     n.gibbs.draws=1e4, burnin=1000,
                                     quiet=TRUE, priors=NULL) {

  # Generate data
  data <- GenerateMultivariateData(n, par$true.means, par$true.sigma, par$true.probs)
  if (is.null(priors)) {
    # Set broad data-based priors.  By using the data, we can avoid accidentally choosing
    # informative prior variances due to unusually scaled data.
    if (!quiet) print("Using data-based priors.")
    priors <- GenerateSamplePriors(x=data$x, k=k)
  }
  analysis.hash <- digest(list(data, par))

  true.means <- par$true.means
  true.probs <- par$true.probs
  true.sigma <- par$true.sigma
  true.lambda <- InvertLinearizedMatrices(VectorizeMatrixList(true.sigma))

  ####################
  # Compute the VB fit
  if (fit.vb) {
    vb.opt.time <- Sys.time()
    vb.optimum <-
      GetVariationalSolution(x=data$x,
                             e.mu=par$true.means,
                             e.lambda=true.lambda,
                             e.log.det.lambda=true.log.det.lambda,
                             e.p=par$true.probs,
                             e.log.pi=GetELogDirichlet(par$true.probs * n),
                             fit.mu=TRUE, fit.lambda=TRUE, fit.pi=TRUE,
                             e.z=data$components,
                             priors=priors, tolerance=1e-9, quiet=TRUE)
    vb.opt.time <- Sys.time() - vb.opt.time

    lrvb.time <- Sys.time()
    theta.cov <- GetThetaCovariance(e.mu=vb.optimum$e.mu, e.mu2=vb.optimum$e.mu2,
                                    lambda.par=vb.optimum$lambda.par, n.par=vb.optimum$n.par,
                                    pi.par=vb.optimum$pi.par,
                                    fit.mu=TRUE, fit.lambda=TRUE, fit.pi=TRUE)
    lrvb.theta.cov <- CPPGetLRVBCovariance(x=data$x,
                                           e_mu=vb.optimum$e.mu,
                                           e_mu2=vb.optimum$e.mu2,
                                           e_lambda=vb.optimum$e.lambda,
                                           e_z=vb.optimum$e.z,
                                           theta_cov=theta.cov, verbose=FALSE)
    lrvb.time <- Sys.time() - lrvb.time
    cat("VB time: ")
    print(vb.time <- lrvb.time + vb.opt.time)
  } else {
    vb.optimum <- lrvb.theta.cov <- theta.cov <- NULL
    lrvb.time <- vb.opt.time <- vb.time <- NULL
  }

  if (fit.gibbs) {
    #######################
    # Get draws from a gibbs sampler

    gibbs.time <- Sys.time()
    # Fit the normal mixture
    
    # The priors cannot be exactly the same since I wrote the VB library with the
    # non-conjugate prior.  :(  (There is no need to use the conjugate prior with 
    # Gibbs or MFVB.)  Also, the rnmixGibbs prior is less flexible.
    prior.list <- list(ncomp=k,
                       Mubar=priors$mu.prior.mean[,1],
                       A=matrix(priors$mu.prior.info[1,1], 1, 1),
                       nu=priors$lambda.prior.n[1],
                       V=solve(ConvertVectorToSymmetricMatrix(priors$lambda.prior.v.inv[,1])),
                       a=priors$p.prior.alpha)
    out <- rnmixGibbs(Data=list(y=data$x),
                      Prior=prior.list,
                      Mcmc=list(R=n.gibbs.draws + burnin, keep=1))
    print(gibbs.time <- Sys.time() - gibbs.time)

    # Re-order the outputs by distance from the origin.
    mu.sizes <- colMeans(GetFieldsFromnNmix(out$nmix,
                                            GetMuFieldSizeFromCompdraw,
                                            1:k, p)[-(1:burnin), ])
    k.vector <- order(mu.sizes)
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
    core.gibbs.cov <- CoreCovarianceFromGibbs(mu.draws, lambda.draws, pi.draws)
    core.gibbs.effsize <- c(effectiveSize(mu.draws),
                            effectiveSize(lambda.draws),
                            effectiveSize(pi.draws))
    if ((min.effsize <- min(core.gibbs.effsize)) < 300) {
      #browser()
      mu.draws$draw <- 1:nrow(mu.draws)
      pi.draws$draw <- 1:nrow(pi.draws)
      lambda.draws$draw <- 1:nrow(lambda.draws)

      print(
      grid.arrange(
        ggplot(melt(mu.draws, id.var="draw")) + geom_point(aes(x=draw, y=value, color=variable)),
        ggplot(melt(lambda.draws, id.var="draw")) + geom_point(aes(x=draw, y=value, color=variable)),
        ggplot(melt(pi.draws, id.var="draw")) + geom_point(aes(x=draw, y=value, color=variable)),
      ncol=3))

      print(sprintf("Small effective gibbs sample size: %f", min.effsize))
    }
  } else {
    core.gibbs.results <- core.gibbs.effsize <- gibbs.time <- core.gibbs.cov <- NULL
  }

  return(list(n=n, k=k, p=p, analysis.hash=analysis.hash,
              true.means=true.means, true.lambda=true.lambda, true.probs=true.probs,
              vb.optimum=vb.optimum, theta.cov=theta.cov, lrvb.theta.cov=lrvb.theta.cov,
              lrvb.time=lrvb.time, vb.opt.time=vb.opt.time, vb.time=vb.time,
              core.gibbs.results=core.gibbs.results, core.gibbs.effsize=core.gibbs.effsize,
              core.gibbs.cov=core.gibbs.cov, gibbs.time=gibbs.time))
}

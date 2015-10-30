library(RcppEigen)
library(Rcpp)
library(Matrix)
library(mvtnorm)
library(ggplot2)
library(numDeriv)
library(reshape2)
library(dplyr)
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

if (FALSE) {
  n.range <- 5000
  k.range <- 2
  reps <- 4
  p.range <- c(2, 3, 4, 5, 7, 8, 9, 10, 12, 15, 18, 22)
  analysis.name <- "new_scaling_p"
} else {
  n.range <- c(2500, 5000, 10000, 25000, 50000, 100000)
  k.range <- 3
  reps <- 8
  p.range <- 3
  analysis.name <- "new_scaling_n"
}



anisotrophy <- 0
vars.scale <- 0.4
burnin <- 200
n.gibbs.draws <- 6e3

iters <- reps * length(n.range) * length(k.range) * length(p.range)
all.results <- list()
iter <- 1

foreach.progress.file <- file(".foreach_progress_scaling", "w")
all.results <- foreach (p.index=1:length(p.range)) %dopar% {
  iter <- 0
  results.list <- list()
  p <- p.range[p.index]
  for (k in k.range) {
    for (n in n.range) {
      par <- GenerateSampleParams(k=k, p=p, vars.scale=vars.scale,
                                  anisotropy=anisotrophy, random.rotation=FALSE)
      for (rep in 1:reps) {
        cat(file=foreach.progress.file, "simulation progress: ", n, k, p, rep, Sys.time(), "\n")
        if (n <= 10000) {
            fit.gibbs <- TRUE
        } else {
            # Gibbs is prohibitively slow on very large datasets.
            fit.gibbs <- FALSE
        }
        priors <- DefaultPriors(p=p, k=k)
        results <- SimulateAndFitMVNMixture(n, k, p, par, priors=priors,
                                            fit.vb=TRUE, fit.gibbs=fit.gibbs,
                                            n.gibbs.draws=n.gibbs.draws,
                                            burnin=burnin)
        iter <- iter + 1
        results.list[[iter]] <- results
       }
    }
  }
  return(results.list)
}
close(foreach.progress.file)

if (kSaveResults) {
  save(all.results, n.range, k.range, p.range, reps,
       file=file.path("data", paste(analysis.name, "Rdata", sep=".")))
}

## Linear Response Variational Bayes NIPS 2015

This repo contains the code and text for our 2015 paper,
"Linear Response Methods for Accurate Covariance Estimates from Mean Field Variational Bayes".

Change to your favorite directory for git repos, and check out this repo with

```git clone https://github.com/rgiordan/LinearResponseVariationalBayesNIPS2015```

# How to regenerate the paper data

The scripts require an environment variable to be set to the location of the
repo.  So if you cloned into the directory ```/home/user/git_repos```,
run the following command in your shell:

```export GIT_REPO_LOC=/home/user/git_repos```

The paper contained experiments from three models, each of which is in a different
directory.

- ```poisson_glmm```: The poisson random effects model
- ```linear_regression```: The random effects linear model
- ```variational_normal_mixture```: The Gaussian mixture models, including the MNIST experiments.

The ```paper``` directory contains the knitr and LaTeX.  Each directory has a number
of scripts in Julia and R that produce various data files.  These files must be
copied to the folder ```paper/data``` to be used to generate the exact plots in the paper.

The scripts require numerous external libraries.  The scripts ```install_julia_packages.jl```
and ```install_R_packages.R``` should check your system and install any that are missing.

Interested users may also want to look at two other stand-alone repos:

- [LinearResponseVariationalBayes.jl](https://github.com/rgiordan/LinearResponseVariationalBayes.jl),
  a Julia package for facilitiating fitting LRVB models in Julia.  (NB: I developed this package based
  on the work for this paper, and as of writing have not yet retrofitted the papers' models to use the new package.)
- [MVNMixtureLRVB](https://github.com/rgiordan/MVNMixtureLRVB),
  an R package (mostly in C++) for fitting LRVB (and, consequently, MFVB) models in R.


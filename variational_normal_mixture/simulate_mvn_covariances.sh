#!/bin/bash
# qsub -pe smp 20 simulate_mvn_covariances.sh
export OMP_NUM_THREADS=1
R CMD BATCH --args --cores=20 simulate_multivariate_normal_mixture_covariances.R simulate_mvn_covariances_cluster.Rout

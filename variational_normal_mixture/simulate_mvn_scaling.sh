#!/bin/bash
# qsub -pe smp 4 simulate_mvn_scaling.sh
export OMP_NUM_THREADS=1
R CMD BATCH --args --cores=4 simulate_multivariate_normal_mixture_scaling.R simulate_mvn_scaling_smaller_cluster.Rout

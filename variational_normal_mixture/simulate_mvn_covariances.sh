#!/bin/bash

SH_FILE=/tmp/simulate_mvn_covariances.sh

echo export OMP_NUM_THREADS=1 > $SH_FILE
echo R CMD BATCH --args --cores=20 simulate_multivariate_normal_mixture_covariances.R simulate_mvn_covariances_cluster.Rout >> $SH_FILE

chmod 700 $SH_FILE
qsub -pe smp 20 $SH_FILE

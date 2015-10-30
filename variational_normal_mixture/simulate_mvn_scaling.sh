#!/bin/bash

SH_FILE=/tmp/simulate_mvn_scaling.sh

echo export OMP_NUM_THREADS=1 > $SH_FILE
echo R CMD BATCH --args --cores=4 simulate_multivariate_normal_mixture_scaling.R simulate_mvn_scaling_smaller_cluster.Rout >> $SH_FILE
chmod 700 $SH_FILE

qsub -pe smp 4 $SH_FILE

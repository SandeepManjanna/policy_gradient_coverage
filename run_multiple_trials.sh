#!/bin/bash
#f='../pickleFiles/'$1'_feat_15x15_500_'$2'_uniform_trial'
f='../pickleFiles/'$1'_feat_50x50_500_'$2'_trial'
for i in {4,5}
do
    echo $f$i
    python feature_aggregated_pg_coverage.py $f$i $1 0.$2
done

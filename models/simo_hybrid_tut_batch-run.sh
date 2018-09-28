#!/bin/bash

# for i in {1..9}
for i in {1..2}					# test
do
	cw=`echo "$i * 0.2" | bc`
	# echo $cw
	python3 ./simo_hybrid_tut_batch-run.py --coordinates_weight $cw
done

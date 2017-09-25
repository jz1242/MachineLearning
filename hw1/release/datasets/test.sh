#!/bin/bash 
python classify.py --mode train --algorithm $1 --model-file $2 --data $3
python classify.py --mode test --model-file $2 --data $4 --predictions-file $5
python compute_accuracy.py $4 $5
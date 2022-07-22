#!/bin/bash -l
#COBALT -q full-node
#COBALT -t 300
#COBALT -n 1
#COBALT -A BirdAudio

module load conda/2021-09-22
conda activate

python ae_train.py
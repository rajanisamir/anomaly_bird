#!/bin/bash -l
#COBALT -q full-node
#COBALT -t 180
#COBALT -n 1
#COBALT -A BirdAudio

module load conda/2021-09-22
conda activate

python -m torch.distributed.launch --nproc_per_node=8 ae_train.py --epochs 300
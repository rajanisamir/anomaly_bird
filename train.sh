#!/bin/bash -l
#COBALT -q full-node
#COBALT -t 100
#COBALT -n 1
#COBALT -A BirdAudio

module load conda/2021-09-22
conda activate

python -m torch.distributed.launch --nproc_per_node 8 ae_train.py --epochs 2000 --single-file /grand/projects/BirdAudio/Morton_Arboretum/audio/set3/00004879/20210816_STUDY/20210816T063139-0500_Rec.wav --bottleneck-dim 100 --prune-anomalies --prune-frequency 500 --exp-dir exp-single-file-ten-dim-single-gpu-prune
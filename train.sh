#!/bin/bash -l
#COBALT -q full-node
#COBALT -t 180
#COBALT -n 1
#COBALT -A BirdAudio

module load conda/2021-09-22
conda activate

# python ae_train.py --batch-size 32 --lr 0.001 --wd 0.000001
# python ae_train.py --batch-size 32 --lr 0.0001 --wd 0.000001
# python ae_train.py --batch-size 32 --lr 0.00001 --wd 0.000001
# python ae_train.py --batch-size 256 --lr 0.001 --wd 0.000001
# python ae_train.py --batch-size 256 --lr 0.0001 --wd 0.000001
# python ae_train.py --batch-size 256 --lr 0.00001 --wd 0.000001 --epochs 1000
python ae_train.py --batch-size 32 --lr 0.001 --wd 0.00001
python ae_train.py --batch-size 32 --lr 0.0001 --wd 0.00001
python ae_train.py --batch-size 32 --lr 0.00001 --wd 0.00001
python ae_train.py --batch-size 256 --lr 0.001 --wd 0.00001
python ae_train.py --batch-size 256 --lr 0.0001 --wd 0.00001
python ae_train.py --batch-size 256 --lr 0.00001 --wd 0.00001
# python ae_train.py --batch-size 256 --lr 0.0001 --wd 0.0000001
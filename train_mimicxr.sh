#!/bin/bash
#SBATCH --gres=gpu:titanxp:1
#SBATCH -o fich-mimicxr-%j.out
#SBATCH -e fich-mimicxr-%j.err
python code/models_train.py --dataset MIMICXR --model SEVGG16 --batchsize 4
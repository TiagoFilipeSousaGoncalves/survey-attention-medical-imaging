#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o fich-mimicxr-%j.out
#SBATCH -e fich-mimicxr-%j.err
python code/mimicxr_train.py
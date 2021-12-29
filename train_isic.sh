#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o fich-isic-%j.out
#SBATCH -e fich-isic-%j.err
python code/isic_train.py
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o fich-cbis-%j.out
#SBATCH -e fich-cbis-%j.err
python code/cbis_train.py
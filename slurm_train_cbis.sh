#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o fich-cbis-train-%j.out
#SBATCH -e fich-cbis-train-%j.err
python code/cbis_train.py
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o fich-cbis-interpretation-%j.out
#SBATCH -e fich-cbis-interpretation-%j.err
python code/cbis_interpretation.py
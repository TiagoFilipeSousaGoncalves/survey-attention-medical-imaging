#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o fich-cbis-test-%j.out
#SBATCH -e fich-cbis-test-%j.err
python code/cbis_test.py
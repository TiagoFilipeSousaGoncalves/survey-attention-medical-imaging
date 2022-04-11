#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err



# python code/models_train.py --data_dir /ctm-hdd-pool01/wjsilva19/MedIA --dataset MIMICCXR --model DeiT --low_data_regimen --perc_train 0.01 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 1 --save_freq 1000
# python code/models_train.py --data_dir /ctm-hdd-pool01/wjsilva19/MedIA --dataset MIMICCXR --model DeiT --low_data_regimen --perc_train 0.1 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 1 --save_freq 1000
# python code/models_train.py --data_dir /ctm-hdd-pool01/wjsilva19/MedIA --dataset MIMICCXR --model DeiT --low_data_regimen --perc_train 0.5 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 1 --save_freq 1000
python code/models_train.py --data_dir /ctm-hdd-pool01/wjsilva19/MedIA --dataset MIMICCXR --model DeiT-T-LRP --batchsize 32 --classweights --epochs 300 --lr 1e-6 --num_workers 2 --gpu_id 0 --save_freq 1000
python code/models_train.py --data_dir /ctm-hdd-pool01/wjsilva19/MedIA --dataset MIMICCXR --model DeiT-T-LRP --low_data_regimen --perc_train 0.01 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --num_workers 2 --gpu_id 0 --save_freq 1000
python code/models_train.py --data_dir /ctm-hdd-pool01/wjsilva19/MedIA --dataset MIMICCXR --model DeiT-T-LRP --low_data_regimen --perc_train 0.1 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --num_workers 2 --gpu_id 0 --save_freq 1000
python code/models_train.py --data_dir /ctm-hdd-pool01/wjsilva19/MedIA --dataset MIMICCXR --model DeiT-T-LRP --low_data_regimen --perc_train 0.5 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --num_workers 2 --gpu_id 0 --save_freq 1000

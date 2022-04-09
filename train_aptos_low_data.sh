#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err



# python code/models_train.py --data_dir /ctm-hdd-pool01/tgoncalv/datasets/APTOS2019_resized/ --dataset APTOS --model DenseNet121 --low_data_regimen --perc_train 0.5 --batchsize 32 --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 0 --save_freq 1000
# python code/models_train.py --data_dir /ctm-hdd-pool01/tgoncalv/datasets/APTOS2019_resized/ --dataset APTOS --model SEDenseNet121 --low_data_regimen --perc_train 0.5 --batchsize 32 --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 0 --save_freq 1000
# python code/models_train.py --data_dir /ctm-hdd-pool01/tgoncalv/datasets/APTOS2019_resized/ --dataset APTOS --model CBAMDenseNet121 --low_data_regimen --perc_train 0.5 --batchsize 32 --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 0 --save_freq 1000
# python code/models_train.py --data_dir /ctm-hdd-pool01/tgoncalv/datasets/APTOS2019_resized/ --dataset APTOS --model ResNet50 --low_data_regimen --perc_train 0.5 --batchsize 32 --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 0 --save_freq 1000
# python code/models_train.py --data_dir /ctm-hdd-pool01/tgoncalv/datasets/APTOS2019_resized/ --dataset APTOS --model SEResNet50 --low_data_regimen --perc_train 0.5 --batchsize 32 --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 0 --save_freq 1000
# python code/models_train.py --data_dir /ctm-hdd-pool01/tgoncalv/datasets/APTOS2019_resized/ --dataset APTOS --model CBAMResNet50 --low_data_regimen --perc_train 0.5 --batchsize 32 --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 0 --save_freq 1000
# python code/models_train.py --data_dir /ctm-hdd-pool01/tgoncalv/datasets/APTOS2019_resized/ --dataset APTOS --model DeiT --low_data_regimen --perc_train 0.5 --batchsize 32 --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 0 --save_freq 1000
python code/models_train.py --data_dir /ctm-hdd-pool01/tgoncalv/datasets/APTOS2019_resized/ --dataset APTOS --model DeiT-LRP --batchsize 16 --epochs 300 --lr 1e-6 --num_workers 2 --gpu_id 0 --save_freq 1000

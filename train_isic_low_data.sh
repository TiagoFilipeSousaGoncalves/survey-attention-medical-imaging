#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o fich.out
#SBATCH -e fich.err



# Save in ctm-hdd-pool
# python code/models_train.py --data_dir /ctm-hdd-pool01/tgoncalv/datasets/ISIC2020/ --dataset ISIC2020 --model DenseNet121 --low_data_regimen --perc_train 0.1 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --outdir /ctm-hdd-pool01/tgoncalv/survey-attention-medical-imaging/results --num_workers 4 --gpu_id 1 --save_freq 50
# python code/models_train.py --data_dir /ctm-hdd-pool01/tgoncalv/datasets/ISIC2020/ --dataset ISIC2020 --model SEDenseNet121 --low_data_regimen --perc_train 0.1 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --outdir /ctm-hdd-pool01/tgoncalv/survey-attention-medical-imaging/results --num_workers 4 --gpu_id 1 --save_freq 50
# python code/models_train.py --data_dir /ctm-hdd-pool01/tgoncalv/datasets/ISIC2020/ --dataset ISIC2020 --model CBAMDenseNet121 --low_data_regimen --perc_train 0.1 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --outdir /ctm-hdd-pool01/tgoncalv/survey-attention-medical-imaging/results --num_workers 4 --gpu_id 1 --save_freq 50
# python code/models_train.py --data_dir /ctm-hdd-pool01/tgoncalv/datasets/ISIC2020/ --dataset ISIC2020 --model ResNet50 --low_data_regimen --perc_train 0.1 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --outdir /ctm-hdd-pool01/tgoncalv/survey-attention-medical-imaging/results --num_workers 4 --gpu_id 1 --save_freq 50
# python code/models_train.py --data_dir /ctm-hdd-pool01/tgoncalv/datasets/ISIC2020/ --dataset ISIC2020 --model SEResNet50 --low_data_regimen --perc_train 0.1 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --outdir /ctm-hdd-pool01/tgoncalv/survey-attention-medical-imaging/results --num_workers 4 --gpu_id 1 --save_freq 50
# python code/models_train.py --data_dir /ctm-hdd-pool01/tgoncalv/datasets/ISIC2020/ --dataset ISIC2020 --model CBAMResNet50 --low_data_regimen --perc_train 0.1 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --outdir /ctm-hdd-pool01/tgoncalv/survey-attention-medical-imaging/results --num_workers 4 --gpu_id 1 --save_freq 50

# Save in Home Directory
# python code/models_train.py --data_dir /ctm-hdd-pool01/tgoncalv/datasets/ISIC2020/ --dataset ISIC2020 --model DenseNet121 --low_data_regimen --perc_train 0.5 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 1 --save_freq 50
# python code/models_train.py --data_dir /ctm-hdd-pool01/tgoncalv/datasets/ISIC2020/ --dataset ISIC2020 --model SEDenseNet121 --low_data_regimen --perc_train 0.5 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 1 --save_freq 50
# python code/models_train.py --data_dir /ctm-hdd-pool01/tgoncalv/datasets/ISIC2020/ --dataset ISIC2020 --model CBAMDenseNet121 --low_data_regimen --perc_train 0.5 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 1 --save_freq 50
# python code/models_train.py --data_dir /ctm-hdd-pool01/tgoncalv/datasets/ISIC2020/ --dataset ISIC2020 --model ResNet50 --low_data_regimen --perc_train 0.5 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 1 --save_freq 50
python code/models_train.py --data_dir /ctm-hdd-pool01/tgoncalv/datasets/ISIC2020/ --dataset ISIC2020 --model SEResNet50 --low_data_regimen --perc_train 0.5 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 1 --save_freq 100
python code/models_train.py --data_dir /ctm-hdd-pool01/tgoncalv/datasets/ISIC2020/ --dataset ISIC2020 --model CBAMResNet50 --low_data_regimen --perc_train 0.5 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 1 --save_freq 100
python code/models_train.py --data_dir /ctm-hdd-pool01/tgoncalv/datasets/ISIC2020/ --dataset ISIC2020 --model DeiT --low_data_regimen --perc_train 0.5 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 1 --save_freq 100
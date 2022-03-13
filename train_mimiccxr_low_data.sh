python code/models_train.py --data_dir /ctm-hdd-pool01/wjsilva19/MedIA --dataset MIMICCXR --model DenseNet121 --low_data_regimen --perc_train 0.1 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 0 --save_freq 100
python code/models_train.py --data_dir /ctm-hdd-pool01/wjsilva19/MedIA --dataset MIMICCXR --model SEDenseNet121 --low_data_regimen --perc_train 0.1 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 0 --save_freq 100
python code/models_train.py --data_dir /ctm-hdd-pool01/wjsilva19/MedIA --dataset MIMICCXR --model CBAMDenseNet121 --low_data_regimen --perc_train 0.1 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 0 --save_freq 100
python code/models_train.py --data_dir /ctm-hdd-pool01/wjsilva19/MedIA --dataset MIMICCXR --model ResNet50 --low_data_regimen --perc_train 0.1 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 0 --save_freq 100
python code/models_train.py --data_dir /ctm-hdd-pool01/wjsilva19/MedIA --dataset MIMICCXR --model SEResNet50 --low_data_regimen --perc_train 0.1 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 0 --save_freq 100
python code/models_train.py --data_dir /ctm-hdd-pool01/wjsilva19/MedIA --dataset MIMICCXR --model CBAMResNet50 --low_data_regimen --perc_train 0.1 --batchsize 32 --classweights --epochs 300 --lr 1e-6 --num_workers 4 --gpu_id 0 --save_freq 100

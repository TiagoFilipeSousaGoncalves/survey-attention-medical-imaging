python code/models_train.py --dataset MIMICCXR --model ResNet50 --batchsize 32 --classweights --lr 1e-6 --num_workers 4 --gpu_id 1 --save_freq 50
python code/models_train.py --dataset MIMICCXR --model SEResNet50 --batchsize 32 --classweights --lr 1e-6 --num_workers 4 --gpu_id 1 --save_freq 50
python code/models_train.py --dataset MIMICCXR --model CBAMResNet50 --batchsize 32 --classweights --lr 1e-6 --num_workers 4 --gpu_id 1 --save_freq 50

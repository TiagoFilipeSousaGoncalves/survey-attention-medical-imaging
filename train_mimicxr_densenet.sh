python code/models_train.py --dataset MIMICCXR --model DenseNet121 --batchsize 32 --classweights --lr 1e-6 --num_workers 4 --gpu_id 0 --save_freq 50
python code/models_train.py --dataset MIMICCXR --model SEDenseNet121 --batchsize 32 --classweights --lr 1e-6 --num_workers 4 --gpu_id 0 --save_freq 50
python code/models_train.py --dataset MIMICCXR --model CBAMDenseNet121 --batchsize 32 --classweights --lr 1e-6 --num_workers 4 --gpu_id 0 --save_freq 50

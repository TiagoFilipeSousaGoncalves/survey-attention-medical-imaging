python code/models_test.py --data_dir data/MedIA --dataset MIMICCXR --split Train --model DeiT --low_data_regimen --perc_train 0.01 --modelckpt results/mimiccxr/deit/2022-03-28_18-06-59 --batchsize 32 --num_workers 4 --gpu_id 0
python code/models_test.py --data_dir data/MedIA --dataset MIMICCXR --split Validation --model DeiT --low_data_regimen --perc_train 0.01 --modelckpt results/mimiccxr/deit/2022-03-28_18-06-59 --batchsize 32 --num_workers 4 --gpu_id 0
python code/models_test.py --data_dir data/MedIA --dataset MIMICCXR --split Test --model DeiT --low_data_regimen --perc_train 0.01 --modelckpt results/mimiccxr/deit/2022-03-28_18-06-59 --batchsize 32 --num_workers 4 --gpu_id 0

python code/models_test.py --data_dir data/MedIA --dataset MIMICCXR --split Train --model DeiT --low_data_regimen --perc_train 0.1 --modelckpt results/mimiccxr/deit/2022-03-28_18-25-13 --batchsize 32 --num_workers 4 --gpu_id 0
python code/models_test.py --data_dir data/MedIA --dataset MIMICCXR --split Validation --model DeiT --low_data_regimen --perc_train 0.1 --modelckpt results/mimiccxr/deit/2022-03-28_18-25-13 --batchsize 32 --num_workers 4 --gpu_id 0
python code/models_test.py --data_dir data/MedIA --dataset MIMICCXR --split Test --model DeiT --low_data_regimen --perc_train 0.1 --modelckpt results/mimiccxr/deit/2022-03-28_18-25-13 --batchsize 32 --num_workers 4 --gpu_id 0

python code/models_test.py --data_dir data/MedIA --dataset MIMICCXR --split Train --model DeiT --low_data_regimen --perc_train 0.5 --modelckpt results/mimiccxr/deit/2022-03-28_20-20-43 --batchsize 32 --num_workers 4 --gpu_id 0
python code/models_test.py --data_dir data/MedIA --dataset MIMICCXR --split Validation --model DeiT --low_data_regimen --perc_train 0.5 --modelckpt results/mimiccxr/deit/2022-03-28_20-20-43 --batchsize 32 --num_workers 4 --gpu_id 0
python code/models_test.py --data_dir data/MedIA --dataset MIMICCXR --split Test --model DeiT --low_data_regimen --perc_train 0.5 --modelckpt results/mimiccxr/deit/2022-03-28_20-20-43 --batchsize 32 --num_workers 4 --gpu_id 0

# Train
# python code/models_test.py --data_dir data/MedIA --dataset MIMICCXR --split Train --model DeiT-T-LRP --modelckpt results/mimiccxr/deit-t-lrp/2022-04-11_12-31-47 --batchsize 32 --num_workers 4 --gpu_id 0
# python code/models_test.py --data_dir data/MedIA --dataset MIMICCXR --split Train --model DeiT-T-LRP --low_data_regimen --perc_train 0.01 --modelckpt results/mimiccxr/deit-t-lrp/2022-04-12_08-27-44 --batchsize 32 --num_workers 4 --gpu_id 0
# python code/models_test.py --data_dir data/MedIA --dataset MIMICCXR --split Train --model DeiT-T-LRP --low_data_regimen --perc_train 0.1 --modelckpt results/mimiccxr/deit-t-lrp/2022-04-12_08-43-56 --batchsize 32 --num_workers 4 --gpu_id 0
python code/models_test.py --data_dir data/MedIA --dataset MIMICCXR --split Train --model DeiT-T-LRP --low_data_regimen --perc_train 0.5 --modelckpt results/mimiccxr/deit-t-lrp/2022-04-12_10-13-45 --batchsize 32 --num_workers 4 --gpu_id 0

# Validation
# python code/models_test.py --data_dir data/MedIA --dataset MIMICCXR --split Validation --model DeiT-T-LRP --modelckpt results/mimiccxr/deit-t-lrp/2022-04-11_12-31-47 --batchsize 32 --num_workers 4 --gpu_id 0
# python code/models_test.py --data_dir data/MedIA --dataset MIMICCXR --split Validation --model DeiT-T-LRP --low_data_regimen --perc_train 0.01 --modelckpt results/mimiccxr/deit-t-lrp/2022-04-12_08-27-44 --batchsize 32 --num_workers 4 --gpu_id 0
# python code/models_test.py --data_dir data/MedIA --dataset MIMICCXR --split Validation --model DeiT-T-LRP --low_data_regimen --perc_train 0.1 --modelckpt results/mimiccxr/deit-t-lrp/2022-04-12_08-43-56 --batchsize 32 --num_workers 4 --gpu_id 0
python code/models_test.py --data_dir data/MedIA --dataset MIMICCXR --split Validation --model DeiT-T-LRP --low_data_regimen --perc_train 0.5 --modelckpt results/mimiccxr/deit-t-lrp/2022-04-12_10-13-45 --batchsize 32 --num_workers 4 --gpu_id 0

# Test
# python code/models_test.py --data_dir data/MedIA --dataset MIMICCXR --split Test --model DeiT-T-LRP --modelckpt results/mimiccxr/deit-t-lrp/2022-04-11_12-31-47 --batchsize 32 --num_workers 4 --gpu_id 0
# python code/models_test.py --data_dir data/MedIA --dataset MIMICCXR --split Test --model DeiT-T-LRP --low_data_regimen --perc_train 0.01 --modelckpt results/mimiccxr/deit-t-lrp/2022-04-12_08-27-44 --batchsize 32 --num_workers 4 --gpu_id 0
# python code/models_test.py --data_dir data/MedIA --dataset MIMICCXR --split Test --model DeiT-T-LRP --low_data_regimen --perc_train 0.1 --modelckpt results/mimiccxr/deit-t-lrp/2022-04-12_08-43-56 --batchsize 32 --num_workers 4 --gpu_id 0
python code/models_test.py --data_dir data/MedIA --dataset MIMICCXR --split Test --model DeiT-T-LRP --low_data_regimen --perc_train 0.5 --modelckpt results/mimiccxr/deit-t-lrp/2022-04-12_10-13-45 --batchsize 32 --num_workers 4 --gpu_id 0

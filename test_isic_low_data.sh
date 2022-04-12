# Train
# python code/models_test.py --data_dir data/ISIC2020/ --dataset ISIC2020 --split Train --model DeiT-T-LRP  --modelckpt results/isic2020/deit-t-lrp/2022-04-11_12-31-47 --batchsize 32 --num_workers 4 --gpu_id 1
# python code/models_test.py --data_dir data/ISIC2020/ --dataset ISIC2020 --split Train --model DeiT-T-LRP --low_data_regimen --perc_train 0.01 --modelckpt results/isic2020/deit-t-lrp/2022-04-12_00-27-56 --batchsize 32 --num_workers 4 --gpu_id 0
# python code/models_test.py --data_dir data/ISIC2020/ --dataset ISIC2020 --split Train --model DeiT-T-LRP --low_data_regimen --perc_train 0.1 --modelckpt results/isic2020/deit-t-lrp/2022-04-12_02-48-55 --batchsize 32 --num_workers 4 --gpu_id 0
python code/models_test.py --data_dir data/ISIC2020/ --dataset ISIC2020 --split Train --model DeiT-T-LRP --low_data_regimen --perc_train 0.5 --modelckpt results/isic2020/deit-t-lrp/2022-04-12_05-51-38 --batchsize 32 --num_workers 4 --gpu_id 0

# Validation
# python code/models_test.py --data_dir data/ISIC2020/ --dataset ISIC2020 --split Validation --model DeiT-T-LRP  --modelckpt results/isic2020/deit-t-lrp/2022-04-11_12-31-47 --batchsize 32 --num_workers 4 --gpu_id 1
# python code/models_test.py --data_dir data/ISIC2020/ --dataset ISIC2020 --split Validation --model DeiT-T-LRP --low_data_regimen --perc_train 0.01 --modelckpt results/isic2020/deit-t-lrp/2022-04-12_00-27-56 --batchsize 32 --num_workers 4 --gpu_id 0
# python code/models_test.py --data_dir data/ISIC2020/ --dataset ISIC2020 --split Validation --model DeiT-T-LRP --low_data_regimen --perc_train 0.1 --modelckpt results/isic2020/deit-t-lrp/2022-04-12_02-48-55 --batchsize 32 --num_workers 4 --gpu_id 0
python code/models_test.py --data_dir data/ISIC2020/ --dataset ISIC2020 --split Validation --model DeiT-T-LRP --low_data_regimen --perc_train 0.5 --modelckpt results/isic2020/deit-t-lrp/2022-04-12_05-51-38 --batchsize 32 --num_workers 4 --gpu_id 0

# Test
# python code/models_test.py --data_dir data/ISIC2020/ --dataset ISIC2020 --split Test --model DeiT-T-LRP  --modelckpt results/isic2020/deit-t-lrp/2022-04-11_12-31-47 --batchsize 32 --num_workers 4 --gpu_id 1
# python code/models_test.py --data_dir data/ISIC2020/ --dataset ISIC2020 --split Test --model DeiT-T-LRP --low_data_regimen --perc_train 0.01 --modelckpt results/isic2020/deit-t-lrp/2022-04-12_00-27-56 --batchsize 32 --num_workers 4 --gpu_id 0
# python code/models_test.py --data_dir data/ISIC2020/ --dataset ISIC2020 --split Test --model DeiT-T-LRP --low_data_regimen --perc_train 0.1 --modelckpt results/isic2020/deit-t-lrp/2022-04-12_02-48-55 --batchsize 32 --num_workers 4 --gpu_id 0
python code/models_test.py --data_dir data/ISIC2020/ --dataset ISIC2020 --split Test --model DeiT-T-LRP --low_data_regimen --perc_train 0.5 --modelckpt results/isic2020/deit-t-lrp/2022-04-12_05-51-38 --batchsize 32 --num_workers 4 --gpu_id 0

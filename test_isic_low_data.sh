# Train
# python code/models_test.py --data_dir data/ISIC2020/ --dataset ISIC2020 --split Train --model DeiT --low_data_regimen --perc_train 0.01 --modelckpt results/isic2020/deit/2022-03-28_10-28-49 --batchsize 32 --num_workers 4 --gpu_id 1
python code/models_test.py --data_dir data/ISIC2020/ --dataset ISIC2020 --split Train --model DeiT-T-LRP  --modelckpt results/isic2020/deit-t-lrp/2022-04-11_12-31-47 --batchsize 32 --num_workers 4 --gpu_id 1

# Validation
# python code/models_test.py --data_dir data/ISIC2020/ --dataset ISIC2020 --split Validation --model DeiT --low_data_regimen --perc_train 0.01 --modelckpt results/isic2020/deit/2022-03-28_10-28-49 --batchsize 32 --num_workers 4 --gpu_id 1
python code/models_test.py --data_dir data/ISIC2020/ --dataset ISIC2020 --split Validation --model DeiT-T-LRP  --modelckpt results/isic2020/deit-t-lrp/2022-04-11_12-31-47 --batchsize 32 --num_workers 4 --gpu_id 1

# Test
# python code/models_test.py --data_dir data/ISIC2020/ --dataset ISIC2020 --split Test --model DeiT --low_data_regimen --perc_train 0.01 --modelckpt results/isic2020/deit/2022-03-28_10-28-49 --batchsize 32 --num_workers 4 --gpu_id 1
python code/models_test.py --data_dir data/ISIC2020/ --dataset ISIC2020 --split Test --model DeiT-T-LRP  --modelckpt results/isic2020/deit-t-lrp/2022-04-11_12-31-47 --batchsize 32 --num_workers 4 --gpu_id 1

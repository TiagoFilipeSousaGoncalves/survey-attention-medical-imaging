# python code/models_test.py --data_dir data/APTOS2019/resized/APTOS2019_resized/ --dataset APTOS --split Train --model DeiT --low_data_regimen --perc_train 0.5 --modelckpt results/aptos/deit/2022-03-30_23-09-48 --batchsize 32 --num_workers 4 --gpu_id 0
python code/models_test.py --data_dir data/APTOS2019/resized/APTOS2019_resized/ --dataset APTOS --split Train --model DeiT-T-LRP --modelckpt results/aptos/deit-t-lrp/2022-04-11_12-29-28 --batchsize 32 --num_workers 4 --gpu_id 0

# python code/models_test.py --data_dir data/APTOS2019/resized/APTOS2019_resized/ --dataset APTOS --split Validation --model DeiT --low_data_regimen --perc_train 0.5 --modelckpt results/aptos/deit/2022-03-30_23-09-48 --batchsize 32 --num_workers 4 --gpu_id 0
python code/models_test.py --data_dir data/APTOS2019/resized/APTOS2019_resized/ --dataset APTOS --split Validation --model DeiT-T-LRP --modelckpt results/aptos/deit-t-lrp/2022-04-11_12-29-28 --batchsize 32 --num_workers 4 --gpu_id 0

# python code/models_test.py --data_dir data/APTOS2019/resized/APTOS2019_resized/ --dataset APTOS --split Test --model DeiT --low_data_regimen --perc_train 0.5 --modelckpt results/aptos/deit/2022-03-30_23-09-48 --batchsize 32 --num_workers 4 --gpu_id 0
python code/models_test.py --data_dir data/APTOS2019/resized/APTOS2019_resized/ --dataset APTOS --split Test --model DeiT-T-LRP --modelckpt results/aptos/deit-t-lrp/2022-04-11_12-29-28 --batchsize 32 --num_workers 4 --gpu_id 0

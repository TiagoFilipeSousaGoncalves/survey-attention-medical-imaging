# Imports
import numpy as np
import os
from PIL import Image

# PyTorch Imports
import torch
from torch.utils.data import Dataset


# Data Directories
data_dir = "/ctm-hdd-pool01/tgoncalv/datasets/PH2Dataset"



# Function: Get images and labels from directory files
def map_images_and_labels(data_dir):

    # The directory should have a directory named "images" inside...
    images_dir = os.path.join(data_dir, "images")

    # ... and a .TXT file named "PH2_dataset.txt"
    txt_info_file = os.path.join(data_dir, "PH2_dataset.txt")

    # ... and a .XLSX file named "PH2_dataset.xlsx"
    xlsx_info_file = os.path.join(data_dir, "PH2_dataset.xlsx")



    # We start by getting the directories (which will contain the images)
    images_folders = [name for name in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, name))]
    # Uncomment if you want to know the number of images in the directory
    print(f"Number of images in the images' directory: {len(images_folders)}")



    return


_ = map_images_and_labels(data_dir=data_dir)
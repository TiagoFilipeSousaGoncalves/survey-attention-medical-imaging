# Imports
import numpy as np
import pandas as pd
import os
from PIL import Image

# PyTorch Imports
import torch
from torch.utils.data import Dataset


# Data Directories
data_dir = "/ctm-hdd-pool01/tgoncalv/datasets/PH2Dataset"


# Legend for Clinical Diagnosis
clinical_diagnosis_labels_dict = {
    0:"Common Nevus",
    1:"Atypical Nevus",
    2:"Melanoma"
}


# Legends for Asymmetry
asymmetry_labels_dict = {
    0:"Fully Symmetric",
    1:"Symetric in 1 axe",
    2:"Fully Asymmetric"
}


# Legends for Pigment Network, Dots/Globules, Streaks, Regression Areas, and Blue-Whitish Veil
pigment_labels_dict = {
    "A":"Absent",
    "AT":"Atypical",
    "P":"Present",
    "T":"Typical"
}


# Legends for Colours
colours_labels_dict = {
    1:"White",
    2:"Red",
    3:"Light-Brown",
    4:"Dark-Brown",
    5:"Blue-Gray",
    6:"Black"
}



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
    # print(f"Number of images in this directory: {len(images_folders)}")


    # Open the .TXT file with the data set information
    # ph2_dataset = np.genfromtxt(fname=txt_info_file, dtype=object, delimiter="|")
    ph2_dataset = pd.read_csv(txt_info_file, delimiter="|")
    # Uncomment to see the output of this file
    print(f"PH2Dataset: {ph2_dataset}")



    return


_ = map_images_and_labels(data_dir=data_dir)
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
    # images_dir = os.path.join(data_dir, "images")

    # ... and a .TXT file named "PH2_dataset.txt"
    txt_info_file = os.path.join(data_dir, "PH2_dataset.txt")

    # ... and a .XLSX file named "PH2_dataset.xlsx"
    # xlsx_info_file = os.path.join(data_dir, "PH2_dataset.xlsx")


    # We start by getting the directories (which will contain the images)
    # images_folders = [name for name in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, name))]
    
    # Uncomment if you want to know the number of images in the directory
    # print(f"Number of images in this directory: {len(images_folders)}")


    # Open the .TXT file with the data set information
    # ph2_dataset = np.genfromtxt(fname=txt_info_file, dtype=object, delimiter="|")
    ph2_dataset = pd.read_csv(txt_info_file, delimiter="|")
    
    # Uncomment to see the output of this file
    # print(f"PH2Dataset: {ph2_dataset}")


    # Get dataset columns (in case you need to adapt this function to other purposes)
    # ph2_dataset_columns = ph2_dataset.columns
    
    # Uncomment to see the output of this file
    # print(f"PH2Dataset: {ph2_dataset_columns}")

    # Get separated variables with this information
    # Names
    ph2_imgs = ph2_dataset['   Name '].values
    
    # Labels (Clinical Diagnosis)
    ph2_labels = ph2_dataset[' Clinical Diagnosis '].values
    
    # Uncomment to see these variables
    # print(f"PH2 Images: {ph2_imgs} and PH2 Labels: {ph2_labels}")
    # print(f"Length of these arrays: {len(ph2_imgs)}, {len(ph2_labels)}")

    # Number of classes
    nr_classes = len(np.unique(ph2_labels))


    return ph2_imgs, ph2_labels, nr_classes



# Create a Dataset Class
class PH2Dataset(Dataset):
    def __init__(self, ph2_imgs, ph2_labels, base_data_path, transform=None):
        """
        Args:
            base_data_path (string): Data directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # Init variables
        self.images_names, self.images_labels = ph2_imgs, ph2_labels
        self.base_data_path = base_data_path
        self.transform = transform


        return 


    # Method: __len__
    def __len__(self):
        return len(self.images_names)



    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        # Get images
        img_name = self.images_names[idx]
        
        # Remove start and end spaces
        img_name = img_name.strip()
        # img_name.replace(" ", "")

        # Open image with PIL
        image = Image.open(os.path.join(self.base_data_path, "images", img_name, f"{img_name}_Dermoscopic_Image", f"{img_name}.bmp"))

        # Get labels
        label = self.images_labels[idx]

        # Apply transformation
        if self.transform:
            image = self.transform(image)


        return image, label



# Uncomment these lines below if you want to test these classes
# Get images and labels
# imgs, labels, nr_classes = map_images_and_labels(data_dir=data_dir)
# print(nr_classes)

# Create torchvision transforms for the Dataset class
# transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# Create a PH2Dataset object
# dataset = PH2Dataset(ph2_imgs=imgs, ph2_labels=labels, base_data_path=data_dir, transform=transforms)

# Create a Dataloader object
# loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

# Go through loader and check if everything is OK
# for batch_idx, (images, labels) in enumerate(loader):
    # print(batch_idx, images, labels)
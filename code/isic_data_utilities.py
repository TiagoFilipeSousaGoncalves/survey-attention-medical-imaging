# Imports
import numpy as np
import pandas as pd
import os
from PIL import Image

# Sklearn Imports
from sklearn.model_selection import train_test_split

# PyTorch Imports
import torch
from torch.utils.data import Dataset



# Create a Dataset Class
class ISIC2020Dataset(Dataset):
    def __init__(self, base_data_path, csv_path, split, random_seed=42, transform=None):
        """
        Args:
            base_data_path (string): Data directory.
            csv_path (string): Path for pickle with annotations.
            split (string): "train", "val", "test" splits.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        # Assure we have the right string in the split argument
        assert split in ["train", "val", "test"], "Please provide a valid split (i.e., 'train', 'val' or 'test')"

        # Aux variables to obtain the correct data splits
        # Read CSV file with label information       
        csv_df = pd.read_csv(csv_path)
        print(f"The dataframe has: {len(csv_df)} records.")
        
        # Get the IDs of the Patients
        patient_ids = csv_df.copy()["patient_id"]
        
        # Get the unique patient ids
        unique_patient_ids = np.unique(patient_ids.values)


        # Split into train, validation and test according to the IDs of the Patients
        # First we split into train and test
        train_ids, test_ids, _, _ = train_test_split(unique_patient_ids, np.zeros_like(unique_patient_ids), test_size=0.30, random_state=random_seed)
        train_ids, val_ids, _, _ = train_test_split(train_ids, np.zeros_like(train_ids), test_size=0.20, random_state=random_seed)


        # Now, we get the data
        if split == "train":
            # Get the right sampled dataframe
            self.dataframe = csv_df.copy()[csv_df.copy()["patient_id"]==train_ids]
            
            # Get the image names
            self.image_names = self.dataframe.copy()["image_name"].values

            # Get the image labels
            self.image_labels = self.dataframe.copy()["target"].values


        elif split == "val":
            # Get the right sampled dataframe
            self.dataframe = csv_df.copy()[csv_df.copy()["patient_id"]==val_ids]
            
            # Get the image names
            self.image_names = self.dataframe.copy()["image_name"].values

            # Get the image labels
            self.image_labels = self.dataframe.copy()["target"].values
        

        else:
            # Get the right sampled dataframe
            self.dataframe = csv_df.copy()[csv_df.copy()["patient_id"]==test_ids]
            
            # Get the image names
            self.image_names = self.dataframe.copy()["image_name"].values

            # Get the image labels
            self.image_labels = self.dataframe.copy()["target"].values


        # Init variables
        self.base_data_path = base_data_path
        imgs_in_folder = os.listdir(self.base_data_path)
        imgs_in_folder = [i for i in imgs_in_folder if not i.startswith(".")]
        print(f"The folder has: {len(imgs_in_folder)} files.")

        self.transform = transform


        return 


    # Method: __len__
    def __len__(self):
        return len(self.image_names)



    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        # Get images
        img_name = self.image_names[idx]
        image = Image.open(os.path.join(self.base_data_path, f"{img_name}.jpg"))

        # Get labels
        label = self.image_labels[idx]

        # Apply transformation
        if self.transform:
            image = self.transform(image)


        return image, label
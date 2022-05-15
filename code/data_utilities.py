# Imports
import os
import _pickle as cPickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image

# Sklearn Imports
from sklearn.model_selection import train_test_split

# PyTorch Imports
import torch
from torch.utils.data import Dataset



# General
# Function: Resize images
def resize_images(datapath, newpath, newheight=512):
    
    # Create new directories (if necessary)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    

    # Go through data directory and generate new (resized) images
    for f in tqdm(os.listdir(datapath)):
        if(f.endswith(".jpg") or f.endswith('.png')):
            img = Image.open(os.path.join(datapath, f))
            w, h = img.size
            ratio = w / h
            new_w = int(np.ceil(newheight * ratio))
            new_img = img.resize((new_w, newheight), Image.ANTIALIAS)
            new_img.save(os.path.join(newpath, f))


    return



# MIMIC-CXR
# MIMIC-CXR: Get labels and paths from pickle
def mimic_map_images_and_labels(base_data_path, pickle_path):
    # Open pickle file
    with open(pickle_path, "rb") as fp:
        pickle_data = cPickle.load(fp)

    # Split Images and Labels
    images_path = list()
    labels = list()

    # Go through pickle file
    for path, clf in zip(pickle_data[:, 0], pickle_data[:, 1]):
        images_path.append(os.path.join(base_data_path, path+".jpg"))
        labels.append(int(clf))
    

    # Assign variables to class variables
    images_paths = images_path
    images_labels = labels
	
	# Nr of Classes
    nr_classes = len(np.unique(images_labels))


    return images_paths, images_labels, nr_classes



# MIMIC-CXR: Dataset Class
class MIMICXRDataset(Dataset):
    def __init__(self, base_data_path, pickle_path, random_seed=42, resized=None, low_data_regimen=None, perc_train=None, transform=None):
        """
        Args:
            base_data_path (string): Data directory.
            pickle_path (string): Path for pickle with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # Init variables
        images_paths, images_labels, _ = mimic_map_images_and_labels(base_data_path, pickle_path)

        # Activate low data regimen training
        if low_data_regimen:
            assert perc_train > 0.0 and perc_train <= 0.50, f"Invalid perc_train '{perc_train}'. Please be sure that perc_train > 0 and perc_train <= 50"


            # Get the data percentage
            images_paths, _, images_labels, _ = train_test_split(images_paths, images_labels, train_size=perc_train, stratify=images_labels, random_state=random_seed)

            print(f"Low data regimen.\n% of train data: {perc_train}")


        # Attribute variables
        self.images_paths = images_paths
        self.images_labels = images_labels
        self.transform = transform


        return 


    # Method: __len__
    def __len__(self):
        return len(self.images_paths)



    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        # Get images
        img_name = self.images_paths[idx]
        image = Image.open(img_name)

        # Get labels
        label = self.images_labels[idx]

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label



# ISIC2020
# ISIC2020: Get data paths
def isic_get_data_paths(base_data_path, resized=None):
    
    # Build data directories
    data_dir = os.path.join(base_data_path, 'jpeg', 'train_resized') if resized else os.path.join(base_data_path, 'jpeg', 'train')
    csv_fpath = os.path.join(base_data_path, 'train.csv')


    # Get number of classes
    nr_classes = 2


    return data_dir, csv_fpath, nr_classes



# ISIC2020: Dataset Class
class ISIC2020Dataset(Dataset):
    def __init__(self, base_data_path, csv_path, split, random_seed=42, resized=None, low_data_regimen=None, perc_train=None, transform=None):
        """
        Args:
            base_data_path (string): Data directory.
            csv_path (string): Path for pickle with annotations.
            split (string): "train", "val", "test" splits.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        # Assure we have the right string in the split argument
        assert split in ["Train", "Validation", "Test"], "Please provide a valid split (i.e., 'Train', 'Validation' or 'Test')"

        # Aux variables to obtain the correct data splits
        # Read CSV file with label information       
        csv_df = pd.read_csv(csv_path)
        # print(f"The dataframe has: {len(csv_df)} records.")
        
        # Get the IDs of the Patients
        patient_ids = csv_df.copy()["patient_id"]
        
        # Get the unique patient ids
        unique_patient_ids = np.unique(patient_ids.values)


        # Split into train, validation and test according to the IDs of the Patients
        # First we split into train and test (60%, 20%, 20%)
        train_ids, test_ids, _, _ = train_test_split(unique_patient_ids, np.zeros_like(unique_patient_ids), test_size=0.20, random_state=random_seed)
        train_ids, val_ids, _, _ = train_test_split(train_ids, np.zeros_like(train_ids), test_size=0.25, random_state=random_seed)


        # Now, we get the data
        if split == "Train":
            # Get the right sampled dataframe
            tr_pids_mask = csv_df.copy().patient_id.isin(train_ids)
            self.dataframe = csv_df.copy()[tr_pids_mask]
            
            # Get the image names
            image_names = self.dataframe.copy()["image_name"].values

            # Get the image labels
            images_labels = self.dataframe.copy()["target"].values


            # Activate low data regimen training
            if low_data_regimen:
                assert perc_train > 0.0 and perc_train <= 0.50, f"Invalid perc_train '{perc_train}'. Please be sure that perc_train > 0 and perc_train <= 50"


                # Get the data percentage
                image_names, _, images_labels, _ = train_test_split(image_names, images_labels, train_size=perc_train, stratify=images_labels, random_state=random_seed)

                print(f"Low data regimen.\n% of train data: {perc_train}")
            


            # Attribute variables object variables
            self.image_names = image_names
            self.images_labels = images_labels


            # Information print
            print(f"The {split} split has {len(self.image_names)} images")


        elif split == "Validation":
            # Get the right sampled dataframe
            val_pids_mask = csv_df.copy().patient_id.isin(val_ids)
            self.dataframe = csv_df.copy()[val_pids_mask]
            
            # Get the image names
            self.image_names = self.dataframe.copy()["image_name"].values

            # Get the image labels
            self.images_labels = self.dataframe.copy()["target"].values

            # Information print
            print(f"The {split} split has {len(self.image_names)} images")
        

        else:
            # Get the right sampled dataframe
            test_pids_mask = csv_df.copy().patient_id.isin(test_ids)
            self.dataframe = csv_df.copy()[test_pids_mask]
            
            # Get the image names
            self.image_names = self.dataframe.copy()["image_name"].values

            # Get the image labels
            self.images_labels = self.dataframe.copy()["target"].values

            # Information print
            print(f"The {split} split has {len(self.image_names)} images")


        # Init variables
        self.base_data_path = base_data_path
        # imgs_in_folder = os.listdir(self.base_data_path)
        # imgs_in_folder = [i for i in imgs_in_folder if not i.startswith(".")]
        # print(f"The folder has: {len(imgs_in_folder)} files.")

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
        label = self.images_labels[idx]

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label



# APTOS2019
# Function: Get labels and paths
def aptos_map_images_and_labels(base_path, split='Train', resized=None, low_data_regimen=None, perc_train=None):

    assert split in ["Train", "Validation", "Test"], f"Invalid split '{split}'. Please choose from ['Train', 'Validation', 'Test']."


    df = pd.read_csv(os.path.join(base_path, 'train.csv'))
    
    if resized:
        df["id_code"] = df["id_code"].apply(lambda x: os.path.join(base_path, "train_resized", x + '.png'))
    
    else:
        df["id_code"] = df["id_code"].apply(lambda x: os.path.join(base_path, "train_images", x + '.png'))
    
    # Convert to binary classification
    df["diagnosis"] = df["diagnosis"].apply(lambda x: 1 if x > 0 else 0)
    nr_classes = len(np.unique(df["diagnosis"]))
    # print(nr_classes)


    # Regular train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(df["id_code"].values, df["diagnosis"].values, train_size=0.85, stratify=df["diagnosis"], random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.75, stratify=y_train, random_state=42)


    if low_data_regimen:
        assert perc_train > 0.0 and perc_train <= 0.50, f" Invalid perc_train '{perc_train}'. Please be sure that perc_train > 0 and perc_train <= 50"


        # Get the data percentage
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=perc_train, stratify=y_train, random_state=42)

        print(f"Low data regimen.\n% of train data: {perc_train}")



    # Get splits
    if split == "Train":
        return X_train, y_train, nr_classes
    
    elif split == "Validation":
        return X_val, y_val, nr_classes
    
    elif split == "Test":
        return X_test, y_test, nr_classes



# Class: Dataset Class
class APTOSDataset(Dataset):
    def __init__(self, base_data_path, split='Train', resized=None, low_data_regimen=None, perc_train=None, transform=None):
        """
        Args:
            base_data_path (string): Data directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # Init variables
        self.images_paths, self.images_labels, _ = aptos_map_images_and_labels(base_data_path, split=split, resized=resized, low_data_regimen=low_data_regimen, perc_train=perc_train)
        self.transform = transform



    # Method: __len__
    def __len__(self):
        return len(self.images_paths)


    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        # Get images
        img_name = self.images_paths[idx]
        image = Image.open(img_name)

        # Get labels
        label = self.images_labels[idx]

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label

# Imports
import os
import _pickle as cPickle
import numpy as np
import pandas as pd
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

# Sklearn Imports
from sklearn.model_selection import train_test_split

# PyTorch Imports
import torch
from torch.utils.data import Dataset



# CBIS-DDSM
# CBIS-DDSM: Get images and labels from directory files
def cbis_map_images_and_labels(dir):
    # Images
    dir_files = os.listdir(dir)
    dir_imgs = [i for i in dir_files if i.split('.')[1]=='png']
    dir_imgs.sort()

    # Labels
    dir_labels_txt = [i.split('.')[0]+'case.txt' for i in dir_imgs]
    

    # Create a Numpy array to append file names and labels
    imgs_labels = np.zeros(shape=(len(dir_imgs), 2), dtype=object)

    # Go through images and labels
    idx = 0
    for image, label in zip(dir_imgs, dir_labels_txt):
        # Debug print
        # print(f"Image file: {image} | Label file: {label}")

        # Append image (Column 0)
        imgs_labels[idx, 0] = image
        
        # Append label (Column 1)
        # Read temp _label
        _label = np.genfromtxt(
            fname=os.path.join(dir, label),
            dtype=str
        )

        # Debug print
        # print(f"_label: {_label}")
        
        # Append to the Numpy Array
        imgs_labels[idx, 1] = str(_label)

        # Debug print
        # print(f"Image file: {imgs_labels[idx, 0]} | Label: {imgs_labels[idx, 1]}")


        # Update index
        idx += 1
    

    # Create labels dictionary to map strings into numbers
    _labels_unique = np.unique(imgs_labels[:, 1])

    # Nr of Classes
    nr_classes = len(_labels_unique)

    # Create labels dictionary
    labels_dict = dict()
    
    for idx, _label in enumerate(_labels_unique):
        labels_dict[_label] = idx


    return imgs_labels, labels_dict, nr_classes



# CBIS-DDSM: Dataset Class
class CBISDataset(Dataset):
    def __init__(self, base_data_path, transform=None):
        """
        Args:
            base_data_path (string): Data directory.
            pickle_path (string): Path for pickle with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            feature_extractor (callable, optional): feature extractor for ViT
        """

        # Init variables
        self.base_data_path = base_data_path
        imgs_labels, self.labels_dict, self.nr_classes = cbis_map_images_and_labels(dir=base_data_path)
        self.images_paths, self.images_labels = imgs_labels[:, 0], imgs_labels[:, 1]
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
        
        # Open image with PIL
        image = Image.open(os.path.join(self.base_data_path, img_name))
        
        # Perform transformations with Numpy Array
        image = np.asarray(image)
        image = np.reshape(image, newshape=(image.shape[0], image.shape[1], 1))
        image = np.concatenate((image, image, image), axis=2)

        # Load image with PIL
        image = Image.fromarray(image)

        # Get labels
        label = self.labels_dict[self.images_labels[idx]]

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label



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
    def __init__(self, base_data_path, pickle_path, transform=None):
        """
        Args:
            base_data_path (string): Data directory.
            pickle_path (string): Path for pickle with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # Init variables
        self.images_paths, self.images_labels, _ = mimic_map_images_and_labels(base_data_path, pickle_path)
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
# ISIC2020: Dataset Class
class ISIC2020Dataset(Dataset):
    def __init__(self, base_data_path, csv_path, split, random_seed=42, transform=None, feature_extractor=None):
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
            self.image_names = self.dataframe.copy()["image_name"].values

            # Get the image labels
            self.image_labels = self.dataframe.copy()["target"].values

            # Information print
            print(f"The {split} split has {len(self.image_names)} images")


        elif split == "Validation":
            # Get the right sampled dataframe
            val_pids_mask = csv_df.copy().patient_id.isin(val_ids)
            self.dataframe = csv_df.copy()[val_pids_mask]
            
            # Get the image names
            self.image_names = self.dataframe.copy()["image_name"].values

            # Get the image labels
            self.image_labels = self.dataframe.copy()["target"].values

            # Information print
            print(f"The {split} split has {len(self.image_names)} images")
        

        else:
            # Get the right sampled dataframe
            test_pids_mask = csv_df.copy().patient_id.isin(test_ids)
            self.dataframe = csv_df.copy()[test_pids_mask]
            
            # Get the image names
            self.image_names = self.dataframe.copy()["image_name"].values

            # Get the image labels
            self.image_labels = self.dataframe.copy()["target"].values

            # Information print
            print(f"The {split} split has {len(self.image_names)} images")


        # Init variables
        self.base_data_path = base_data_path
        # imgs_in_folder = os.listdir(self.base_data_path)
        # imgs_in_folder = [i for i in imgs_in_folder if not i.startswith(".")]
        # print(f"The folder has: {len(imgs_in_folder)} files.")

        self.transform = transform
        self.feature_extractor = feature_extractor

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

        if(self.feature_extractor):
            image = self.feature_extractor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        return image, label



# PH2
# Dictionary: Legend for Clinical Diagnosis
clinical_diagnosis_labels_dict = {
    0:"Common Nevus",
    1:"Atypical Nevus",
    2:"Melanoma"
}


# Dictionary: Legends for Asymmetry
asymmetry_labels_dict = {
    0:"Fully Symmetric",
    1:"Symetric in 1 axe",
    2:"Fully Asymmetric"
}


# Dictionary: Legends for Pigment Network, Dots/Globules, Streaks, Regression Areas, and Blue-Whitish Veil
pigment_labels_dict = {
    "A":"Absent",
    "AT":"Atypical",
    "P":"Present",
    "T":"Typical"
}


# Dictionary: Legends for Colours
colours_labels_dict = {
    1:"White",
    2:"Red",
    3:"Light-Brown",
    4:"Dark-Brown",
    5:"Blue-Gray",
    6:"Black"
}



# Function: Get images and labels from directory files
def ph2_map_images_and_labels(data_dir):

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



# PH2: Dataset Class
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



# APTOS2019: Get labels and paths

def aptos_map_images_and_labels(base_path, split='train'):
    df = pd.read_csv(os.path.join(base_path, 'train.csv'))
    df["id_code"] = df["id_code"].apply(lambda x: os.path.join(base_path, "train_images", x + '.png'))
    
    # convert to binary classification
    df["diagnosis"] = df["diagnosis"].apply(lambda x: 1 if x > 0 else 0)
    stats = np.unique(df["diagnosis"], return_counts=True)
    #print(stats)

    X_train, X_test, y_train, y_test = train_test_split(df["id_code"].values, df["diagnosis"].values, train_size=0.85, stratify=df["diagnosis"], random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.75, stratify=y_train, random_state=42)

    if(split == "train"):
        return X_train, y_train
    elif(split == "val"):
        return X_val, y_val
    elif(split == "test"):
        return X_test, y_test
    else:
        print("Invalid split. Please choose from [train, val, test]")
        quit()



# APTOS2019: Dataset Class
class APTOSDataset(Dataset):
    def __init__(self, base_data_path, split='train', transform=None):
        """
        Args:
            base_data_path (string): Data directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # Init variables
        self.images_paths, self.images_labels = aptos_map_images_and_labels(base_data_path, split=split)
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
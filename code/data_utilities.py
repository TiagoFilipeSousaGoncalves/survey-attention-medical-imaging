# Imports
import os
import _pickle as cPickle
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

# PyTorch Imports
import torch
from torch.utils.data import Dataset




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
            pickle_path (string): Path for pickle with annotations.
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
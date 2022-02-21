# Imports
import os
import argparse
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

# Sklearn Import
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision


# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)


# Project Imports
from model_utilities_baseline import VGG16, DenseNet121, ResNet50
from model_utilities_se import SEResNet50, SEVGG16, SEDenseNet121
from model_utilities_cbam import CBAMResNet50, CBAMVGG16, CBAMDenseNet121
from transformers import ViTFeatureExtractor, ViTForImageClassification
from data_utilities import cbis_map_images_and_labels, mimic_map_images_and_labels, ph2_map_images_and_labels, CBISDataset, MIMICXRDataset, PH2Dataset, ISIC2020Dataset



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data set
parser.add_argument('--dataset', type=str, required=True, choices=["CBISDDSM", "ISIC2020", "MIMICCXR", "PH2"], help="Data set: CBISDDSM, ISIC2020, MIMICCXR, PH2")

# Data split
parser.add_argument('--split', type=str, required=True, choices=["Train", "Validation", "Test"], help="Data split: Train, Validation or Test")

# Model
parser.add_argument('--model', type=str, required=True, choices=["DenseNet121", "ResNet50", "VGG16", "SEDenseNet121", "SEResNet50", "SEVGG16", "CBAMDenseNet121", "CBAMResNet50", "CBAMVGG16", "ViT"], help='Model Name: DenseNet121, ResNet50, VGG16, SEDenseNet121, SEResNet50, SEVGG16, CBAMDenseNet121, CBAMResNet50, CBAMVGG16, ViT')

# Batch size
parser.add_argument('--batchsize', type=int, default=4, help="Batch-size for training and validation")

# Image size
parser.add_argument('--imgsize', type=int, default=224, help="Size of the image after transforms")

# Resize
parser.add_argument('--resize', type=str, choices=["direct_resize", "resizeshortest_randomcrop"], default="direct_resize", help="Resize data transformation")

# Output directory
parser.add_argument("--outdir", type=str, default="results", help="Output directory")

# Number of workers
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")

# Number of layers (ViT)
parser.add_argument("--nr_layers", type=int, default=12, help="Number of hidden layers (only for ViT)")


# Parse the arguments
args = parser.parse_args()



# Dataset
dataset = args.dataset

# Data split
data_split = args.split

# Results Directory
outdir = args.outdir

# Number of workers (threads)
workers = args.num_workers

# Batch size
BATCH_SIZE = args.batchsize

# Image size (after transforms)
IMG_SIZE = args.imgsize

# Number of layers of the Visual Transformer
nr_layers = args.nr_layers

# Resize (data transforms)
resize_opt = args.resize

# Results directory
outdir = args.outdir


# Weights directory
weights_dir = os.path.join(outdir, "weights")



# CBISDDSM
if dataset == "CBISDDSM":
    # Directories
    data_dir = "/ctm-hdd-pool01/tgoncalv/datasets/CBISPreprocDataset"
    # data_dir = "/BARRACUDA8T/DATASETS/CBIS_DDSM/"

    # Data splits
    if data_split == "Train":
        eval_dir = os.path.join(data_dir, "train")
    
    elif data_split == "Validation":
        eval_dir = os.path.join(data_dir, "val")
    
    elif data_split == "Test":
        eval_dir = os.path.join(data_dir, "test")
 

    imgs_labels, labels_dict, nr_classes = cbis_map_images_and_labels(dir=eval_dir)


# MIMICXR
elif dataset == "MIMICCXR":
    # Directories
    data_dir = "/ctm-hdd-pool01/wjsilva19/MedIA"
    # data_dir = "/BARRACUDA8T/DATASETS/MIMIC_CXR_Pleural_Subset/"

    # Data splits
    if data_split == "Train":
        eval_dir = os.path.join(data_dir, "Train_images_AP_resized")
    
    elif data_split == "Validation":
        eval_dir = os.path.join(data_dir, "Val_images_AP_resized")
    
    elif data_split == "Test":
        eval_dir = os.path.join(data_dir, "Test_images_AP_resized")


    _, _, nr_classes = mimic_map_images_and_labels(base_data_path=eval_dir, pickle_path=os.path.join(eval_dir, "Annotations.pickle"))


# ISIC2020
elif dataset == "ISIC2020":
    # Directories
    data_dir = "/ctm-hdd-pool01/tgoncalv/datasets/ISIC2020/jpeg/train"
    csv_fpath = "/ctm-hdd-pool01/tgoncalv/datasets/ISIC2020/train.csv"
    # data_dir = "/BARRACUDA8T/DATASETS/ISIC2020/train"
    # csv_fpath = "/BARRACUDA8T/DATASETS/ISIC2020/train.csv"


# PH2
elif dataset == "PH2":
    # Directories
    data_dir = "/ctm-hdd-pool01/tgoncalv/datasets/PH2Dataset"

    # Data split
    # Get all the images, labels, and number of classes of PH2 Dataset
    ph2_imgs, ph2_labels, nr_classes = ph2_map_images_and_labels(data_dir)

    # Remove class 1
    ph2_imgs = ph2_imgs[ph2_labels!=1]
    ph2_labels = ph2_labels[ph2_labels!=1]


    # Split into train, validation and test (60%, 20%, 20%)
    # Train and Test
    ph2_imgs_train, ph2_imgs_test, ph2_labels_train, ph2_labels_test = train_test_split(ph2_imgs, ph2_labels, test_size=0.20, random_state=random_seed)
    # Train and Validation
    ph2_imgs_train, ph2_imgs_val, ph2_labels_train, ph2_labels_val = train_test_split(ph2_imgs_train, ph2_labels_train, test_size=0.25, random_state=random_seed)


    # Data splits
    if data_split == "Train":
        ph2_imgs_eval = ph2_imgs_train.copy()
        ph2_labels_eval = ph2_labels_train.copy()
    
    elif data_split == "Validation":
        ph2_imgs_eval = ph2_imgs_val.copy()
        ph2_labels_eval = ph2_labels_val.copy()

    elif data_split == "Test":
        ph2_imgs_eval = ph2_imgs_test.copy()
        ph2_labels_eval = ph2_labels_test.copy()



# Choose GPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Input Data Dimensions
img_nr_channels = 3
img_height = IMG_SIZE
img_width = IMG_SIZE


# Get the right model from the CLI
model = args.model
feature_extractor = None


# VGG-16
if model == "VGG16":
    model = VGG16(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)
    model_name = "vgg16"


# DenseNet-121
elif model == "DenseNet121":
    model = DenseNet121(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)
    model_name = "densenet121"


# ResNet50
elif model == "ResNet50":
    model = ResNet50(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)
    model_name = "resnet50"


# SEResNet50
elif model == "SEResNet50":
    model = SEResNet50(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)
    model_name = "seresnet50"


# SEVGG16
elif model == "SEVGG16":
    model = SEVGG16(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)
    model_name = "sevgg16"


# SEDenseNet121
elif model == "SEDenseNet121":
    model = SEDenseNet121(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)
    model_name = "sedensenet121"


# CBAMResNet50
elif model == "CBAMResNet50":
    model = CBAMResNet50(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)
    model_name = "cbamresnet50"


# CBAMVGG16
elif model == "CBAMVGG16":
    model = CBAMVGG16(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)
    model_name = "cbamvgg16"


# CBAMDenseNet121
elif model == "CBAMDenseNet121":
    model = CBAMDenseNet121(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)
    model_name = "cbamdensenet121"

# ViT
elif model == "ViT":
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=nr_classes, ignore_mismatched_sizes=True, num_hidden_layers=nr_layers, image_size=IMG_SIZE)
    model_name = "vit"
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')



# Load model weights
checkpoint = torch.load(os.path.join(weights_dir, f"{model_name}_{dataset.lower()}_best.pt"), map_location=DEVICE)

# We need to add an exception to prevent some errors from the attention mechanisms that were already trained
# Case without any error
try:
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

# Case related to CBAM blocks
except:

    # Debug print
    print("Fixing key values with old trained CBAM models")

    # Get missing keys
    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False) 

    if len(missing) == len(unexpected):
        
        # Method to remap the new state_dict keys (https://gist.github.com/the-bass/0bf8aaa302f9ba0d26798b11e4dd73e3)
        state_dict = checkpoint['model_state_dict']
        
        # New state dict
        new_state_dict = OrderedDict()

        for key, value in state_dict.items():
            if key in unexpected:
                new_state_dict[missing[unexpected.index(key)]] = value
            else:
                new_state_dict[key] = value


    # Now we try to load the new state_dict
    model.load_state_dict(new_state_dict, strict=True)
    print("Success!")



# Move model to device
model = model.to(DEVICE)

# Put model in evaluation mode
model.eval()


# Validation
# Transforms
eval_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMG_SIZE if resize_opt == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.RandomCrop(IMG_SIZE if resize_opt == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=feature_extractor.image_mean if feature_extractor else MEAN, std=feature_extractor.image_std if feature_extractor else STD)
])

# Datasets
# CBISDDSM
if dataset == "CBISDDSM":
    eval_set = CBISDataset(base_data_path=eval_dir, transform=eval_transforms)


# MIMCCXR
elif dataset == "MIMICCXR":
    eval_set = MIMICXRDataset(base_data_path=eval_dir, pickle_path=os.path.join(eval_dir, "Annotations.pickle"), transform=eval_transforms)


# ISIC2020
elif dataset == "ISIC2020":
    eval_set = ISIC2020Dataset(base_data_path=data_dir, csv_path=csv_fpath, split=data_split, random_seed=random_seed, transform=eval_transforms, feature_extractor=feature_extractor)


# PH2
elif dataset == "PH2":
    eval_set = PH2Dataset(ph2_imgs=ph2_imgs_eval, ph2_labels=ph2_labels_eval, base_data_path=data_dir, transform=eval_transforms)



# Dataloaders
eval_loader = DataLoader(dataset=eval_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=workers)


# Loss function
LOSS = torch.nn.CrossEntropyLoss(reduction="sum")


# Test model
print(f"Testing Step | Data Set: {dataset}")


# Running train loss
run_eval_loss = 0.0


# Deactivate gradients
with torch.no_grad():

    # Iterate through dataloader
    for images, labels in tqdm(eval_loader):
        y_eval_true = np.append(y_eval_true, labels.numpy(), axis=0)

        # Move data data anda model to GPU (or not)
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        # Forward pass: compute predicted outputs by passing inputs to the model
        if(isinstance(model, ViTForImageClassification)):
            out = model(pixel_values=images)
            logits = out.logits
        else:
            logits = model(images)
        
        # Compute the batch loss
        # Using CrossEntropy w/ Softmax
        loss = LOSS(logits, labels)
        
        # Update batch losses
        run_eval_loss += loss


        # Using Softmax Activation
        # Apply Softmax on Logits and get the argmax to get the predicted labels
        s_logits = torch.nn.Softmax(dim=1)(logits)
        s_logits = torch.argmax(s_logits, dim=1)
        y_eval_pred = torch.cat((y_eval_pred, s_logits))

    

    # Compute Average Validation Loss
    avg_eval_loss = run_eval_loss/len(eval_loader.dataset)

    # Compute Validation Accuracy
    y_eval_pred = y_eval_pred.cpu().detach().numpy()
    eval_acc = accuracy_score(y_true=y_eval_true, y_pred=y_eval_pred)


    # Print Statistics
    print(f"Model Name: {model_name}\t{data_split} Loss: {avg_eval_loss}\t{data_split} Accuracy: {eval_acc}")
    


# Finish statement
print("Finished.")

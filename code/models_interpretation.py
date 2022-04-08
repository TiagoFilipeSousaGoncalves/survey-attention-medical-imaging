# Imports
import os
import argparse
from collections import OrderedDict
import numpy as np

# Sklearn Import
from sklearn.model_selection import train_test_split

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision

# Project Imports
from data_utilities import cbis_map_images_and_labels, mimic_map_images_and_labels, ph2_map_images_and_labels, CBISDataset, MIMICXRDataset, PH2Dataset, ISIC2020Dataset, APTOSDataset
from model_utilities_baseline import VGG16, DenseNet121, ResNet50
from model_utilities_cbam import CBAMResNet50, CBAMVGG16, CBAMDenseNet121
from model_utilities_xai import generate_post_hoc_xmap
from model_utilities_se import SEResNet50, SEVGG16, SEDenseNet121
from model_utilities_xai_transformer import generate_attribution
from xai_utilities_explanation_generator import LRP
from transformers import ViTFeatureExtractor, ViTForImageClassification, DeiTFeatureExtractor, DeiTForImageClassification




# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data dir
parser.add_argument('--data_dir', type=str, default="data", help="Main data directory (e.g., 'data/')")

# Data set
parser.add_argument('--dataset', type=str, required=True, choices=["CBISDDSM", "ISIC2020", "MIMICCXR", "APTOS", "PH2"], help="Data set: CBISDDSM, ISIC2020, MIMICCXR, APTOS, PH2")

# Data split
parser.add_argument('--split', type=str, required=True, choices=["Train", "Validation", "Test"], help="Data split: Train, Validation or Test")

# Model
parser.add_argument('--model', type=str, required=True, choices=["DenseNet121", "ResNet50", "VGG16", "SEDenseNet121", "SEResNet50", "SEVGG16", "CBAMDenseNet121", "CBAMResNet50", "CBAMVGG16", "ViT", "DeiT"], help='Model Name: DenseNet121, ResNet50, VGG16, SEDenseNet121, SEResNet50, SEVGG16, CBAMDenseNet121, CBAMResNet50, CBAMVGG16, ViT, DeiT')

# Model checkpoint
parser.add_argument("--modelckpt", type=str, required=True, help="Directory where model is stored")

# Batch size
parser.add_argument('--batchsize', type=int, default=4, help="Batch-size for training and validation")

# Image size
parser.add_argument('--imgsize', type=int, default=224, help="Size of the image after transforms")

# Resize
parser.add_argument('--resize', type=str, choices=["direct_resize", "resizeshortest_randomcrop"], default="direct_resize", help="Resize data transformation")

# Number of workers
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")

# GPU ID
parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU")

# Number of layers (ViT)
parser.add_argument("--nr_layers", type=int, default=12, help="Number of hidden layers (only for ViT)")


# Parse the arguments
args = parser.parse_args()


# Data directory
data_dir = args.data_dir

# Dataset
dataset = args.dataset

# Data split
data_split = args.split

# Model Directory
modelckpt = args.modelckpt

# Number of workers (threads)
workers = args.num_workers

# GPU ID
gpu_id = args.gpu_id

# Batch size
BATCH_SIZE = args.batchsize

# Image size (after transforms)
IMG_SIZE = args.imgsize

# Number of layers of the Visual Transformer
nr_layers = args.nr_layers

# Resize (data transforms)
resize_opt = args.resize



# CBISDDSM
if dataset == "CBISDDSM":
    # Directories
    # data_dir = "/ctm-hdd-pool01/tgoncalv/datasets/CBISPreprocDataset"
    dataset_dir = os.path.join(data_dir, "CBISPreprocDataset")

    if data_split == "Train":
        eval_dir = os.path.join(dataset_dir, "train")
    
    elif data_split == "Validation":
        eval_dir = os.path.join(dataset_dir, "val")
    
    elif data_split == "Test":
        eval_dir = os.path.join(dataset_dir, "test")
    

    # Get labels and number of classes
    imgs_labels, labels_dict, nr_classes = cbis_map_images_and_labels(dir=eval_dir)


    # Weights directory
    weights_dir = os.path.join(modelckpt, "weights")

    # Post-hoc explanations
    xai_maps_dir = os.path.join(modelckpt, "xai_maps")
    if not(os.path.isdir(xai_maps_dir)):
        os.makedirs(xai_maps_dir)



# MIMICXR
elif dataset == "MIMICCXR":
    # Directories
    # data_dir = "/ctm-hdd-pool01/wjsilva19/MedIA"
    dataset_dir = os.path.join(data_dir, "MedIA")

    if data_split == "Train":    
        eval_dir = os.path.join(dataset_dir, "Train_images_AP_resized")
    
    elif data_split == "Validation":
        eval_dir = os.path.join(dataset_dir, "Val_images_AP_resized")
    
    elif data_split == "Test":
        eval_dir = os.path.join(dataset_dir, "Test_images_AP_resized")
    

    # Get labels and number of classes
    _, _, nr_classes = mimic_map_images_and_labels(base_data_path=eval_dir, pickle_path=os.path.join(eval_dir, "Annotations.pickle"))


    # Weights directory
    weights_dir = os.path.join(modelckpt, "weights")

    # Post-hoc explanations
    xai_maps_dir = os.path.join(modelckpt, "xai_maps")
    if not(os.path.isdir(xai_maps_dir)):
        os.makedirs(xai_maps_dir)



# APTOS
elif dataset == "APTOS":
    # Directories
    # data_dir = "/BARRACUDA8T/DATASETS/APTOS2019/"
    dataset_dir = os.path.join(data_dir, "APTOS2019")

    # Number of classes is added manually
    nr_classes = 2

    # Weights directory
    weights_dir = os.path.join(modelckpt, "weights")

    # Post-hoc explanations
    xai_maps_dir = os.path.join(modelckpt, "xai_maps")
    if not(os.path.isdir(xai_maps_dir)):
        os.makedirs(xai_maps_dir)



# ISIC2020
elif dataset == "ISIC2020":
    # Directories
    # data_dir = "/ctm-hdd-pool01/tgoncalv/datasets/ISIC2020/jpeg/train"
    # csv_fpath = "/ctm-hdd-pool01/tgoncalv/datasets/ISIC2020/train.csv"
    # data_dir = "/BARRACUDA8T/DATASETS/ISIC2020/train"
    # csv_fpath = "/BARRACUDA8T/DATASETS/ISIC2020/train.csv"
    # data_dir = "data/ISIC2020/jpeg/train"
    # csv_fpath = "data/ISIC2020/train.csv"

    dataset_dir = os.path.join(data_dir, "ISIC2020/jpeg/train")
    csv_fpath = os.path.join(data_dir, "ISIC2020/train.csv")

    # Number of classes is added manually
    nr_classes = 2

    # Weights directory
    weights_dir = os.path.join(modelckpt, "weights")

    # Post-hoc explanations
    xai_maps_dir = os.path.join(modelckpt, "xai_maps")
    if not(os.path.isdir(xai_maps_dir)):
        os.makedirs(xai_maps_dir)



# PH2
elif dataset == "PH2":
    # Directories
    # data_dir = "/ctm-hdd-pool01/tgoncalv/datasets/PH2Dataset"
    dataset_dir = os.path.join(data_dir, "PH2Dataset")


    # Data split
    # Get all the images, labels, and number of classes of PH2 Dataset
    ph2_imgs, ph2_labels, nr_classes = ph2_map_images_and_labels(dataset_dir)

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
    

    # Weights directory
    weights_dir = os.path.join(modelckpt, "weights")

    # Post-hoc explanations
    xai_maps_dir = os.path.join(modelckpt, "xai_maps")
    if not(os.path.isdir(xai_maps_dir)):
        os.makedirs(xai_maps_dir)



# Choose GPU
DEVICE = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"


# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Input Data Dimensions
img_nr_channels = 3
img_height = IMG_SIZE
img_width = IMG_SIZE


# Get the right model from the CLI
model = args.model
model_name = model.lower()
feature_extractor = None


# VGG-16
if model == "VGG16":
    model = VGG16(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)


# DenseNet-121
elif model == "DenseNet121":
    model = DenseNet121(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)


# ResNet50
elif model == "ResNet50":
    model = ResNet50(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)


# SEResNet50
elif model == "SEResNet50":
    model = SEResNet50(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)


# SEVGG16
elif model == "SEVGG16":
    model = SEVGG16(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)


# SEDenseNet121
elif model == "SEDenseNet121":
    model = SEDenseNet121(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)


# CBAMResNet50
elif model == "CBAMResNet50":
    model = CBAMResNet50(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)


# CBAMVGG16
elif model == "CBAMVGG16":
    model = CBAMVGG16(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)


# CBAMDenseNet121
elif model == "CBAMDenseNet121":
    model = CBAMDenseNet121(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)

# ViT
elif model == "ViT":
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=nr_classes, ignore_mismatched_sizes=True, num_hidden_layers=nr_layers, image_size=IMG_SIZE)
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# DeiT
elif model == "DeiT":
    model = DeiTForImageClassification.from_pretrained('facebook/deit-tiny-distilled-patch16-224', num_labels=nr_classes, ignore_mismatched_sizes=True, num_hidden_layers=nr_layers, image_size=IMG_SIZE)
    feature_extractor = DeiTFeatureExtractor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')



# Load model weights
model_file = os.path.join(weights_dir, f"{model_name}_{dataset.lower()}_best.pt")
checkpoint = torch.load(model_file, map_location=DEVICE)

# We need to add an exception to prevent some errors from the attention mechanisms that were already trained
# Case without any error
try:
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print("Loaded model from " + model_file)

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

# APTOS
elif dataset == "APTOS":
    eval_set = APTOSDataset(base_data_path=dataset_dir, split=data_split, transform=eval_transforms)

# ISIC2020
elif dataset == "ISIC2020":
    eval_set = ISIC2020Dataset(base_data_path=dataset_dir, csv_path=csv_fpath, split=data_split, random_seed=random_seed, transform=eval_transforms)


# PH2
elif dataset == "PH2":
    eval_set = PH2Dataset(ph2_imgs=ph2_imgs_eval, ph2_labels=ph2_labels_eval, base_data_path=data_dir, transform=eval_transforms)



# Dataloaders
eval_loader = DataLoader(dataset=eval_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=workers)



# Generate post-hoc explanation
print(f"Generating post-hoc explanation | Model: {model_name} | Dataset: {dataset}")


# Iterate through dataloader
for batch_idx, (images, labels) in enumerate(eval_loader):

    # Move data data anda model to GPU (or not)
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    model = model.to(DEVICE)

    # Forward pass: compute predicted outputs by passing inputs to the model
    if(isinstance(model, ViTForImageClassification) or isinstance(model, DeiTForImageClassification)):
        out = model(pixel_values=images)
        logits = out.logits
    else:
        logits = model(images)

    
    # Using Softmax Activation
    # Apply Softmax on Logits and get the argmax to get the predicted labels
    s_logits = torch.nn.Softmax(dim=1)(logits)
    s_logits = torch.argmax(s_logits, dim=1)

    # Get prediction
    prediction = s_logits[0].cpu().item()
    prediction = int(prediction)


    # Generate post-hoc explanation
    # For DeiT
    if(isinstance(model, ViTForImageClassification) or isinstance(model, DeiTForImageClassification)):
        
        # Create an attribution generator
        attribution_generator = LRP(model)
        original_image, original_label, xai_map = generate_attribution(image=images[0], attribution_generator=attribution_generator, ground_truth_label=labels[0], device=DEVICE, mean_array=MEAN, std_array=STD)

        # xAI maps saving directory
        xai_map_save_dir = os.path.join(xai_maps_dir, "lrp")
        if not(os.path.isdir(xai_map_save_dir)):
            os.makedirs(xai_map_save_dir)
        
        # Save image
        np.save(file=os.path.join(xai_map_save_dir, f"idx{batch_idx}_gt{original_label}_pred{prediction}.npy"), arr=xai_map, allow_pickle=True)



    # For the rest of the models
    else:
        for post_hoc_method in ["deeplift", "lrp"]:

            # Get original image and post-hoc explanation
            original_image, original_label, xai_map = generate_post_hoc_xmap(image=images[0], ground_truth_label=labels[0], model=model, post_hoc_method=post_hoc_method, device=DEVICE, mean_array=MEAN, std_array=STD)


            # Original images saving directory
            ori_img_save_dir = os.path.join(xai_maps_dir, "original-imgs")
            if not(os.path.isdir(ori_img_save_dir)):
                os.makedirs(ori_img_save_dir)

            # Save image
            np.save(file=os.path.join(ori_img_save_dir, f"idx{batch_idx}_gt{original_label}_pred{prediction}.npy"), arr=original_image, allow_pickle=True)


            # xAI maps saving directory
            xai_map_save_dir = os.path.join(xai_maps_dir, post_hoc_method)
            if not(os.path.isdir(xai_map_save_dir)):
                os.makedirs(xai_map_save_dir)
            
            # Save image
            np.save(file=os.path.join(xai_map_save_dir, f"idx{batch_idx}_gt{original_label}_pred{prediction}.npy"), arr=xai_map, allow_pickle=True)




# Finish statement
print("Finished.")

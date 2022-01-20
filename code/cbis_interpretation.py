# Imports
from email.policy import strict
import numpy as np
import os
import argparse

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
from model_utilities_xai import generate_post_hoc_xmap
from cbis_data_utilities import map_images_and_labels, CBISDataset



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the argument
parser.add_argument('--model', type=str, required=True, help='Model Name: VGG16, DenseNet121, ResNet50, SEResNet50, SEVGG16, SEDenseNet121, CBAMResNet50, CBAMVGG16, CBAMDenseNet121')

# Parse the argument
args = parser.parse_args()



# Directories
# data_dir = "/ctm-hdd-pool01/tgoncalv/datasets/CBISPreprocDataset"
data_dir = "data/CBISPreprocDataset"
test_dir = os.path.join(data_dir, "test")

# Results and Weights
weights_dir = os.path.join("results", "cbis", "weights")

# Post-hoc explanations
xai_maps_dir = os.path.join("results", "cbis", "xai_maps")
if not(os.path.isdir(xai_maps_dir)):
    os.makedirs(xai_maps_dir)



# Choose GPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Input Data Dimensions
img_nr_channels = 3
img_height = 224
img_width = 224

# Output Data Dimensions
imgs_labels, labels_dict, nr_classes = map_images_and_labels(dir=test_dir)


# Get the right model from the CLI
model = args.model 


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


else:
    raise ValueError(f"{model} is not a valid model name argument. Please provide a valid model name.")



# Hyper-parameters
BATCH_SIZE = 1


# Load model weights
missing, unexpected = model.load_state_dict(torch.load(os.path.join(weights_dir, f"{model_name}_cbis.pt"), map_location=DEVICE), strict=False)
print(len(missing), len(unexpected))
print(unexpected)
model.eval()


# Load data
# Test
# Transforms
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])

# Test Dataset
test_set = CBISDataset(base_data_path=test_dir, transform=test_transforms)

# Test Dataloader
test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)



# Generate post-hoc explanation
print(f"Generating post-hoc explanation for model: {model_name}")


# Iterate through dataloader
for batch_idx, (images, labels) in enumerate(test_loader):

    # Move data data anda model to GPU (or not)
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    model = model.to(DEVICE)

    # Forward pass: compute predicted outputs by passing inputs to the model
    logits = model(images)

    
    # Using Softmax Activation
    # Apply Softmax on Logits and get the argmax to get the predicted labels
    s_logits = torch.nn.Softmax(dim=1)(logits)
    s_logits = torch.argmax(s_logits, dim=1)

    # Get prediction
    prediction = s_logits[0].cpu().item()
    prediction = int(prediction)


    # Generate post-hoc explanation
    for post_hoc_method in ["deeplift", "lrp"]:

        # Get original image and post-hoc explanation
        original_image, original_label, xai_map = generate_post_hoc_xmap(image=images[0], ground_truth_label=labels[0], model=model, post_hoc_method=post_hoc_method, device=DEVICE, mean_array=MEAN, std_array=STD)


        # Original images saving directory
        ori_img_save_dir = os.path.join(xai_maps_dir, f"{model_name.lower()}", "original-imgs")
        if not(os.path.isdir(ori_img_save_dir)):
            os.makedirs(ori_img_save_dir)

        # Save image
        np.save(file=os.path.join(ori_img_save_dir, f"idx{batch_idx}_gt{original_label}_pred{prediction}.npy"), arr=original_image, allow_pickle=True)


        # xAI maps saving directory
        xai_map_save_dir = os.path.join(xai_maps_dir, f"{model_name.lower()}", post_hoc_method)
        if not(os.path.isdir(xai_map_save_dir)):
            os.makedirs(xai_map_save_dir)
        
        # Save image
        np.save(file=os.path.join(xai_map_save_dir, f"idx{batch_idx}_gt{original_label}_pred{prediction}.npy"), arr=xai_map, allow_pickle=True)




# Finish statement
print("Finished.")

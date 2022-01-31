# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# PyTorch Imports
import torch

# Captum Imports
from captum.attr import visualization as viz

# Project Imports
from model_utilities_xai import convert_figure



# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data set
parser.add_argument('--dataset', type=str, required=True, help="Data set: CBISDDSM, MIMICXR, ISIC2020")

# Model
parser.add_argument('--model', type=str, required=True, help='Model Name: VGG16, DenseNet121, ResNet50, SEResNet50, SEVGG16, SEDenseNet121, CBAMResNet50, CBAMVGG16, CBAMDenseNet121')


# Parse the argument
args = parser.parse_args()



# Get the data set and set the directories
dataset = args.dataset

# CBIS-DDSM
if dataset == "CBISDDSM":
    
    # Set the directory of the xAI maps
    xai_maps_dir = os.path.join("results", "cbis", "xai_maps")

    # Set the directory of the .PNG figures
    png_figs_dir = os.path.join(xai_maps_dir, "png")

    # Create .PNG directory, if needed
    if not(os.path.isdir(png_figs_dir)):
            os.makedirs(png_figs_dir)


# MIMIC-CXR
elif dataset == "MIMICXR": 
    pass


# ISIC2020
elif dataset == "ISIC2020":
    pass


else:
    pass



# Get the right model from the CLI
model = args.model 


# VGG-16
if model == "VGG16":
    model_name = "vgg16"


# DenseNet-121
elif model == "DenseNet121":
    model_name = "densenet121"


# ResNet50
elif model == "ResNet50":
    model_name = "resnet50"


# SEResNet50
elif model == "SEResNet50":
    model_name = "seresnet50"


# SEVGG16
elif model == "SEVGG16":
    model_name = "sevgg16"


# SEDenseNet121
elif model == "SEDenseNet121":
    model_name = "sedensenet121"


# CBAMResNet50
elif model == "CBAMResNet50":
    model_name = "cbamresnet50"


# CBAMVGG16
elif model == "CBAMVGG16":
    model_name = "cbamvgg16"


# CBAMDenseNet121
elif model == "CBAMDenseNet121":
    model_name = "cbamdensenet121"


else:
    raise ValueError(f"{model} is not a valid model name argument. Please provide a valid model name.")



# Create a list of sub-directories
sub_dirs = ["original-imgs", "deeplift", "lrp"]


# Get model's results directory
model_save_dir = os.path.join(xai_maps_dir, f"{model_name.lower()}")


# Get the files
attribute_flist = os.listdir(os.path.join(model_save_dir, sub_dirs[0]))
attribute_flist = [i for i in attribute_flist if not i.startswith('.')]
attribute_flist.sort()


# .PNG sub-dirs
for sub_dir_name in sub_dirs:
    png_sub_dir = os.path.join(png_figs_dir, model_name, sub_dir_name)
    if not(os.path.isdir(png_sub_dir)):
        os.makedirs(png_sub_dir)



# Debug print
print("Creating figures...")

# Loop through files
for fname in attribute_flist:

    # Original Image
    original_fname = os.path.join(model_save_dir, sub_dirs[0], fname)
    original_img = np.load(original_fname, allow_pickle=True)

    # Get figure
    figure, axis = viz.visualize_image_attr(None, original_img, method="original_image", use_pyplot=False)

    # Get the figure from memory
    convert_figure(figure)
    
    # Save figure
    plt.axis('off')
    plt.savefig(os.path.join(png_figs_dir, model_name, sub_dirs[0], fname.split('.')[0]+'.png'), bbox_inches='tight')
    plt.clf()
    # plt.show()
    plt.close()



    # Deeplift
    deeplift_fname = os.path.join(model_save_dir, sub_dirs[1], fname)
    deeplift_map = np.load(deeplift_fname, allow_pickle=True)

    # Get figure
    figure, axis = viz.visualize_image_attr(deeplift_map, original_img, method="blended_heat_map", sign="all", show_colorbar=False, use_pyplot=False)
    
    # Get the figure from memory
    convert_figure(figure)

    # Save figure
    plt.axis('off')
    plt.savefig(os.path.join(png_figs_dir, model_name, sub_dirs[1], fname.split('.')[0]+'.png'), bbox_inches='tight')
    plt.clf()
    # plt.show()
    plt.close()



    # LRP
    lrp_fname = os.path.join(model_save_dir, sub_dirs[2], fname)
    lrp_map = np.load(lrp_fname, allow_pickle=True)

    # Get figure
    figure, axis = viz.visualize_image_attr(lrp_map, original_img, method="blended_heat_map", sign="all", show_colorbar=False, use_pyplot=False)
    
    # Get the figure from memory
    convert_figure(figure)

    # Save figure
    plt.axis('off')
    plt.savefig(os.path.join(png_figs_dir, model_name, sub_dirs[2], fname.split('.')[0]+'.png'), bbox_inches='tight')
    plt.clf()
    # plt.show()
    plt.close()



print("Finished")
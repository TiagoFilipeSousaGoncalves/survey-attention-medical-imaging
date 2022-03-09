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
# Model checkpoint
parser.add_argument("--modelckpt", type=str, required=True, help="Directory where model is stored")


# Parse the argument
args = parser.parse_args()


# Checkpoint
modelckpt = args.modelckpt


    
# Set the directory of the xAI maps
xai_maps_dir = os.path.join(modelckpt, "xai_maps")

# Set the directory of the .PNG figures
png_figs_dir = os.path.join(modelckpt, "xai_maps_png")

# Create .PNG directory, if needed
if not(os.path.isdir(png_figs_dir)):
    os.makedirs(png_figs_dir)



# Create a list of sub-directories
sub_dirs = ["original-imgs", "deeplift", "lrp"]



# Get the files
attribute_flist = os.listdir(os.path.join(xai_maps_dir, sub_dirs[0]))
attribute_flist = [i for i in attribute_flist if not i.startswith('.')]
attribute_flist.sort()


# .PNG sub-dirs
for sub_dir_name in sub_dirs:
    png_sub_dir = os.path.join(png_figs_dir, sub_dir_name)
    if not(os.path.isdir(png_sub_dir)):
        os.makedirs(png_sub_dir)



# Debug print
print(f"Creating figures from: {modelckpt}")

# Loop through files
for fname in attribute_flist:
    

    # Try to generate the final images of the attributes
    try:

        # Original Image
        original_fname = os.path.join(xai_maps_dir, sub_dirs[0], fname)
        original_img = np.load(original_fname, allow_pickle=True)

        # Get figure
        figure, axis = viz.visualize_image_attr(None, original_img, method="original_image", use_pyplot=False)

        # Get the figure from memory
        convert_figure(figure)
        
        # Save figure
        plt.axis('off')
        plt.savefig(os.path.join(png_figs_dir, sub_dirs[0], fname.split('.')[0]+'.png'), bbox_inches='tight')
        plt.clf()
        # plt.show()
        plt.close()



        # Deeplift
        deeplift_fname = os.path.join(xai_maps_dir, sub_dirs[1], fname)
        deeplift_map = np.load(deeplift_fname, allow_pickle=True)
        # print(deeplift_map.min(), deeplift_map.max(), deeplift_map.mean())

        # Get figure
        # figure, axis = viz.visualize_image_attr(deeplift_map, original_img, method="blended_heat_map", sign="all", show_colorbar=False, use_pyplot=False)
        figure, axis = viz.visualize_image_attr(deeplift_map, original_img, method="blended_heat_map", sign="all", cmap='bwr', show_colorbar=False, use_pyplot=False, alpha_overlay=0.7)
        
        # Get the figure from memory
        convert_figure(figure)

        # Save figure
        plt.axis('off')
        plt.savefig(os.path.join(png_figs_dir, sub_dirs[1], fname.split('.')[0]+'.png'), bbox_inches='tight')
        plt.clf()
        # plt.show()
        plt.close()



        # LRP
        lrp_fname = os.path.join(xai_maps_dir, sub_dirs[2], fname)
        lrp_map = np.load(lrp_fname, allow_pickle=True)

        # Get figure
        # figure, axis = viz.visualize_image_attr(lrp_map, original_img, method="blended_heat_map", sign="all", show_colorbar=False, use_pyplot=False)
        figure, axis = viz.visualize_image_attr(lrp_map, original_img, method="blended_heat_map", sign="all", cmap='bwr', show_colorbar=False, use_pyplot=False, alpha_overlay=0.7)
        
        # Get the figure from memory
        convert_figure(figure)

        # Save figure
        plt.axis('off')
        plt.savefig(os.path.join(png_figs_dir, sub_dirs[2], fname.split('.')[0]+'.png'), bbox_inches='tight')
        plt.clf()
        # plt.show()
        plt.close()
    

    # Pass if the image + attribute has bad quality
    except:
        pass



print("Finished")

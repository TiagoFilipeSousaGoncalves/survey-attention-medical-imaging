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
parser.add_argument("--modelckpt", type=str, required=True, help="Directory where model is stored.")

# Type of saliency maps
parser.add_argument("--saliency_maps", type=str, required=True, choices=["ALL", "DEEPLIFT", "LRP"], help="Saliency maps: ALL, DEEPLIFT, LRP.")

# Alpha overlay for saliency maps
parser.add_argument("--alpha_overlay", type=float, default=0.5, help="Alpha parameter for overlayed saliency maps.")


# Parse the argument
args = parser.parse_args()


# Checkpoint
modelckpt = args.modelckpt

# Saliency maps
saliency_maps = args.saliency_maps
# print(saliency_maps)

# Alpha overlay
alpha_overlay = args.alpha_overlay



# Set the directory of the xAI maps
xai_maps_dir = os.path.join(modelckpt, "xai_maps")

# Set the directory of the .PNG figures
png_figs_dir = os.path.join(modelckpt, "xai_maps_png")

# Create .PNG directory, if needed
if not(os.path.isdir(png_figs_dir)):
    os.makedirs(png_figs_dir)



# Create a list of sub-directories
if saliency_maps == "ALL":
    sub_dirs = ["original-imgs", "deeplift", "lrp"]

elif saliency_maps == "DEEPLIFT":
    sub_dirs = ["original-imgs", "deeplift"]

elif saliency_maps == "LRP":
    sub_dirs = ["original-imgs", "lrp"]



# Debug print
print(f"Creating figures from: {modelckpt}")

# Loop through files
# First, we create the .PNG sub-dirs
for sub_dir_name in sub_dirs:
    png_sub_dir = os.path.join(png_figs_dir, sub_dir_name)
    if not(os.path.isdir(png_sub_dir)):
        os.makedirs(png_sub_dir)
    

    # Then, we get the files
    attribute_flist = os.listdir(os.path.join(xai_maps_dir, sub_dir_name))
    attribute_flist = [i for i in attribute_flist if not i.startswith('.')]
    attribute_flist.sort()

    for fname in attribute_flist:
        

        # Try to generate the final images of the attributes
        try:
            
            if sub_dir_name == "original-imgs":
                # Original Image
                original_fname = os.path.join(xai_maps_dir, sub_dir_name, fname)
                original_img = np.load(original_fname, allow_pickle=True)

                # Get figure
                figure, axis = viz.visualize_image_attr(None, original_img, method="original_image", use_pyplot=False)

                # Get the figure from memory
                convert_figure(figure)
                
                # Save figure
                plt.axis('off')
                plt.savefig(os.path.join(png_figs_dir, sub_dir_name, fname.split('.')[0]+'.png'), bbox_inches='tight')
                plt.clf()
                # plt.show()
                plt.close()



            # Deeplift
            elif sub_dir_name == "deeplift":
                # Original Image
                original_fname = os.path.join(xai_maps_dir, "original-imgs", fname)
                original_img = np.load(original_fname, allow_pickle=True)

                # print("Generating DeepLIFT saliency maps images.")
                deeplift_fname = os.path.join(xai_maps_dir, sub_dir_name, fname)
                deeplift_map = np.load(deeplift_fname, allow_pickle=True)
                # print(deeplift_map.min(), deeplift_map.max(), deeplift_map.mean())

                # Get figure
                # figure, axis = viz.visualize_image_attr(deeplift_map, original_img, method="blended_heat_map", sign="all", show_colorbar=False, use_pyplot=False)
                figure, axis = viz.visualize_image_attr(deeplift_map, original_img, method="blended_heat_map", sign="all", cmap='bwr', show_colorbar=False, use_pyplot=False, alpha_overlay=alpha_overlay)
                
                # Get the figure from memory
                convert_figure(figure)

                # Save figure
                plt.axis('off')
                plt.savefig(os.path.join(png_figs_dir, sub_dir_name, fname.split('.')[0]+'.png'), bbox_inches='tight')
                plt.clf()
                # plt.show()
                plt.close()



            # LRP
            elif sub_dir_name == "lrp":
                # Original Image
                original_fname = os.path.join(xai_maps_dir, "original-imgs", fname)
                original_img = np.load(original_fname, allow_pickle=True)

                # print("Generating LRP saliency maps images.")
                lrp_fname = os.path.join(xai_maps_dir, sub_dir_name, fname)
                lrp_map = np.load(lrp_fname, allow_pickle=True)

                if len(lrp_map.shape) == 2:
                    lrp_map = np.reshape(lrp_map, (lrp_map.shape[0], lrp_map.shape[1], 1))

                # Get figure
                # figure, axis = viz.visualize_image_attr(lrp_map, original_img, method="blended_heat_map", sign="all", show_colorbar=False, use_pyplot=False)
                figure, axis = viz.visualize_image_attr(lrp_map, original_img, method="blended_heat_map", sign="all", cmap='bwr', show_colorbar=False, use_pyplot=False, alpha_overlay=alpha_overlay)
                
                # Get the figure from memory
                convert_figure(figure)

                # Save figure
                plt.axis('off')
                plt.savefig(os.path.join(png_figs_dir, sub_dir_name, fname.split('.')[0]+'.png'), bbox_inches='tight')
                plt.clf()
                # plt.show()
                plt.close()
        

        # Pass if the image + attribute has bad quality
        except:
            pass



print("Finished")

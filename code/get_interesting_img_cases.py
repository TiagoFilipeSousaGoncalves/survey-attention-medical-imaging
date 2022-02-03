# Imports
import os
import argparse
import re


# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data set
parser.add_argument('--dataset', type=str, required=True, help="Data set: CBISDDSM, MIMICXR, ISIC2020")

# Pattern
parser.add_argument('--pattern', type=str, required=True, help="Possible patterns: gt0_pred0, gt1_pred1, gt0_pred1, gt1_pred0")



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



# MIMIC-CXR
elif dataset == "MIMICXR": 
    pass


# ISIC2020
elif dataset == "ISIC2020":
    pass


else:
    pass



# Get the list of directories
sub_dirs = [d for d in os.listdir(png_figs_dir) if not d.startswith('.')]
sub_dirs.sort()
# print(f"Sub-directories in this directory: {sub_dirs}")


# Intersection 
intersection = list()


# Go through all the subdirectories
for idx, sub_d in enumerate(sub_dirs):

    # Get images in this sub_dir
    tmp = [i for i in os.listdir(os.path.join(png_figs_dir, sub_d, "original-imgs")) if not i.startswith('.')]
    tmp.sort()

    if idx == 0:
        intersection = tmp.copy()
    
    else:
        intersection = list(set(intersection) & set(tmp))



# Now check pattern
pattern = args.pattern


# Final list of files
final = list()
for fname in intersection:
    
    # Find the pattern in file
    if len(re.findall(pattern, fname)) > 0:
        final.append(fname)


# Save list of files into a txt file
final.sort()
with open(f"files_{pattern}.txt", "a+") as t:
    for f in final:
        t.write(f"{f} \n")

# Imports
import os
import argparse
import re


# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data set
parser.add_argument('--dataset', type=str, required=True, choices=["CBISDDSM", "ISIC2020", "MIMICCXR", "APTOS", "PH2"], help="Data set: CBISDDSM, ISIC2020, MIMICCXR, APTOS, PH2")

# Pattern
parser.add_argument('--pattern', type=str, required=True, help="Possible patterns: gt0_pred0, gt1_pred1, gt0_pred1, gt1_pred0")



# Parse the argument
args = parser.parse_args()



# Get the data set and set the directories
dataset = args.dataset



# Modelcheckpoints list
modelckpt_list = [
    'results/aptos/densenet121/2022-03-02_20-18-23',
    'results/aptos/resnet50/2022-03-02_17-57-24',
    'results/aptos/sedensenet121/2022-03-02_21-16-28',
    'results/aptos/seresnet50/2022-03-02_18-46-07',
    'results/aptos/cbamdensenet121/2022-03-02_22-17-26',
    'results/aptos/cbamresnet50/2022-03-02_19-41-49'
    ]



# Intersection 
intersection = list()


# Go through the modelckpt list of files
for idx, modelckpt in enumerate(modelckpt_list):
    
    # Set the directory of the .PNG figures
    png_figs_dir = os.path.join(modelckpt, "xai_maps_png")


    # Get images in this sub_dir
    tmp = [i for i in os.listdir(os.path.join(png_figs_dir, "original-imgs")) if not i.startswith('.')]
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
with open(f"files_{dataset}_{pattern}.txt", "a+") as t:
    for f in final:
        t.write(f"{f} \n")

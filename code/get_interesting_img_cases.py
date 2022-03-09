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
if dataset == "APTOS":
    modelckpt_list = [

    ]


elif dataset == "MIMICCXR":
    modelckpt_list = [
        "results/mimiccxr/densenet121/2022-03-02_14-52-30",
        "results/mimiccxr/sedensenet121/2022-03-04_10-27-00",
        "results/mimiccxr/cbamdensenet121/2022-03-05_13-17-36",
        "results/mimiccxr/resnet50/2022-03-02_14-53-12",
        "results/mimiccxr/seresnet50/2022-03-04_09-03-01",
        "results/mimiccxr/cbamresnet50/2022-03-08_16-16-53"

    ]


elif dataset == "ISIC2020":
    modelckpt_list = [
    'results/isic2020/densenet121/2022-03-01_18-19-07',
    'results/isic2020/sedensenet121/2022-03-02_01-12-08',
    'results/isic2020/cbamdensenet121/2022-03-02_08-19-52',
    'results/isic2020/resnet50/2022-02-28_22-58-32',
    'results/isic2020/seresnet50/2022-03-01_04-29-33',
    'results/isic2020/cbamresnet50/2022-03-01_10-57-05'
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

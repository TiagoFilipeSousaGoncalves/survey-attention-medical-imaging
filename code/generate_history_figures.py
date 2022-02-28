# Imports
from cProfile import label
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt



# Fix Random Seeds
random_seed = 42
np.random.seed(random_seed)



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Model
parser.add_argument('--model', type=str, required=True, choices=["DenseNet121", "ResNet50", "VGG16", "SEDenseNet121", "SEResNet50", "SEVGG16", "CBAMDenseNet121", "CBAMResNet50", "CBAMVGG16", "ViT", "DeiT"], help='Model Name: DenseNet121, ResNet50, VGG16, SEDenseNet121, SEResNet50, SEVGG16, CBAMDenseNet121, CBAMResNet50, CBAMVGG16, ViT, DeiT')

# Model checkpoint
parser.add_argument("--modelckpt", type=str, required=True, help="Directory where model is stored")


# Parse the arguments
args = parser.parse_args()


# Data directory
# Model
model = args.model

# Model Directory
modelckpt = args.modelckpt



# Directories
# History directory
history_dir = os.path.join(modelckpt, "history")



# Train and Validation losses
tr_losses = np.load(file=os.path.join(history_dir, f"{model.lower()}_tr_losses.npy"), allow_pickle=True)
val_losses = np.load(file=os.path.join(history_dir, f"{model.lower()}_val_losses.npy"), allow_pickle=True)


# Plot configuration
plt.title(f"{model} Loss Values")
plt.plot(tr_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.legend(loc='best')
plt.savefig(fname=os.path.join(history_dir, f"{model.lower()}_losses.png"), pad_inches='tight')
plt.clf()



# Train and Validation Accuracy
tr_metrics = np.load(file=os.path.join(history_dir, f"{model.lower()}_tr_metrics.npy"), allow_pickle=True)
val_metrics = np.load(file=os.path.join(history_dir, f"{model.lower()}_val_metrics.npy"), allow_pickle=True)


# Plot configuration
plt.title(f"{model} Accuracy Values")
plt.plot(tr_metrics, label="Train")
plt.plot(val_metrics, label="Validation")
plt.legend(loc='best')
plt.savefig(fname=os.path.join(history_dir, f"{model.lower()}_accuracy.png"), pad_inches='tight')
plt.clf()

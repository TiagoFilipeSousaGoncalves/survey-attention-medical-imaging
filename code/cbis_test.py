# Imports
import numpy as np
import _pickle as cPickle
import os
from PIL import Image
import argparse

# Sklearn Import
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# PyTorch Imports
import torch
from torch._C import device
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
from cbis_data_utilities import map_images_and_labels, CBISDataset



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data set
parser.add_argument('--dataset', type=str, required=True, help="Data set: CBISDDSM, ISIC2020, MIMICXR")

# Model
parser.add_argument('--model', type=str, required=True, help='Model Name: DenseNet121, ResNet50, VGG16, SEDenseNet121, SEResNet50, SEVGG16, CBAMDenseNet121, CBAMResNet50, CBAMVGG16')

# Data split
parser.add_argument('--split', type=str, required=True, help='Data split: Train, Validation, Test')

# Parse the argument
args = parser.parse_args()



# Directories
# data_dir = "/ctm-hdd-pool01/tgoncalv/datasets/CBISPreprocDataset"
data_dir = "data/CBISPreprocDataset"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")


# Results and Weights
weights_dir = os.path.join("results", "cbis", "weights")



# Choose GPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Input Data Dimensions
img_nr_channels = 3
img_height = 224
img_width = 224


# Get the data split
data_split = args.split

if data_split == "Train":
    curr_dir = train_dir


elif data_split == "Validation":
    curr_dir = val_dir


elif data_split == "Test":
    curr_dir = test_dir


else:
    raise ValueError(f"{data_split} is not a valid split name argument. Please provide a valid split name.")



# Output Data Dimensions
imgs_labels, labels_dict, nr_classes = map_images_and_labels(dir=curr_dir)



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
BATCH_SIZE = 32
LOSS = torch.nn.CrossEntropyLoss()


# Load model weights
model.load_state_dict(torch.load(os.path.join(weights_dir, f"{model_name}_cbis.pt"), map_location=DEVICE))
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
test_set = CBISDataset(base_data_path=curr_dir, transform=test_transforms)

# Test Dataloader
test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)



# Test model
print(f"Testing Phase | Data Set: {args.dataset} | Data Split:{data_split}")


# Initialise lists to compute scores
y_test_true = list()
y_test_pred = list()


# Running train loss
run_test_loss = 0.0


# Deactivate gradients
with torch.no_grad():

    # Iterate through dataloader
    for batch_idx, (images, labels) in enumerate(test_loader):

        # Move data data anda model to GPU (or not)
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        model = model.to(DEVICE)

        # Forward pass: compute predicted outputs by passing inputs to the model
        logits = model(images)
        
        # Compute the batch loss
        # Using CrossEntropy w/ Softmax
        loss = LOSS(logits, labels)
        
        # Update batch losses
        run_test_loss += (loss.item() * images.size(0))

        # Concatenate lists
        y_test_true += list(labels.cpu().detach().numpy())
        
        # Using Softmax Activation
        # Apply Softmax on Logits and get the argmax to get the predicted labels
        s_logits = torch.nn.Softmax(dim=1)(logits)
        s_logits = torch.argmax(s_logits, dim=1)
        y_test_pred += list(s_logits.cpu().detach().numpy())

    

    # Compute Average Train Loss
    avg_test_loss = run_test_loss/len(test_loader.dataset)

    # Compute Training Accuracy
    test_acc = accuracy_score(y_true=y_test_true, y_pred=y_test_pred)
    # val_recall = recall_score(y_true=y_val_true, y_pred=y_val_pred)
    # val_precision = precision_score(y_true=y_val_true, y_pred=y_val_pred)
    # val_f1 = f1_score(y_true=y_val_true, y_pred=y_val_pred)

    # Print Statistics
    print(f"{model_name}\tTest Loss: {avg_test_loss}\tTest Accuracy: {test_acc}")
    # print(f"Validation Loss: {avg_val_loss}\tValidation Accuracy: {val_acc}\tValidation Recall: {val_recall}\tValidation Precision: {val_precision}\tValidation F1-Score: {val_f1}")


# Finish statement
print("Finished.")
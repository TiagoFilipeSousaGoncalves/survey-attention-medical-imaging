# Imports
import numpy as np
import os

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision


# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)


# Project Imports
from isic_model_utilities import VGG16, DenseNet121, ResNet50
from isic_data_utilities import ISIC2020Dataset


# Directories
data_dir = "/ctm-hdd-pool01/tgoncalv/datasets/ISIC2020/jpeg/train"
csv_fpath = "/ctm-hdd-pool01/tgoncalv/datasets/ISIC2020/train.csv"

# Results and Weights
weights_dir = os.path.join("results", "isic", "weights")


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
nr_classes = 2


# VGG-16
# model = VGG16(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)
# model_name = "vgg16"

# DenseNet-121
# model = DenseNet121(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)
# model_name = "densenet121"

# ResNet50
model = ResNet50(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)
model_name = "resnet50"


# Hyper-parameters
LOSS = torch.nn.CrossEntropyLoss()
BATCH_SIZE = 8


# Load model weights
model.load_state_dict(torch.load(os.path.join(weights_dir, f"{model_name}_isic.pt")))
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
test_set = ISIC2020Dataset(base_data_path=data_dir, csv_path=csv_fpath, split='test', random_seed=random_seed, transform=test_transforms)

# Test Dataloader
test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)



# Test model
print("Testing Phase...")


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
    # val_recall = recall_score(y_true=y_val_true, y_pred=y_val_pred, average="weighted")
    # val_precision = precision_score(y_true=y_val_true, y_pred=y_val_pred, average="weighted")
    # val_f1 = f1_score(y_true=y_val_true, y_pred=y_val_pred, average="weighted")

    # Print Statistics
    print(f"{model_name}\tValidation Loss: {avg_test_loss}\tValidation Accuracy: {test_acc}")
    # print(f"Validation Loss: {avg_val_loss}\tValidation Accuracy: {val_acc}\tValidation Recall: {val_recall}\tValidation Precision: {val_precision}\tValidation F1-Score: {val_f1}")


# Finish statement
print("Finished.")
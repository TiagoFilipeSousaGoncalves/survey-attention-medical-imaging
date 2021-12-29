# Imports
import numpy as np
import _pickle as cPickle
import os
from PIL import Image

# Sklearn Import
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
from model_utilities_baseline import VGG16, DenseNet121, ResNet50
from model_utilities_se import SEResNet50, SEVGG16, SEDenseNet121
from cbis_data_utilities import map_images_and_labels, CBISDataset


# Directories
data_dir = "/ctm-hdd-pool01/tgoncalv/datasets/CBISPreprocDataset"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

# Results and Weights
weights_dir = os.path.join("results", "cbis", "weights")
if not os.path.isdir(weights_dir):
    os.makedirs(weights_dir)


# History Files
history_dir = os.path.join("results", "cbis", "history")
if not os.path.isdir(history_dir):
    os.makedirs(history_dir)


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
imgs_labels, labels_dict, nr_classes = map_images_and_labels(dir=train_dir)


# Baseline Models
# VGG-16
# model = VGG16(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)
# model_name = "vgg16"

# DenseNet-121
# model = DenseNet121(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)
# model_name = "densenet121"

# ResNet50
# model = ResNet50(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)
# model_name = "resnet50"


# Squeeze-Excitation Models
# SEResNet50
# model = SEResNet50(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)
# model_name = "seresnet50"

# SEVGG16
# model = SEVGG16(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)
# model_name = "sevgg16"

# SEDenseNet121
# model = SEDenseNet121(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)
# model_name = "sedensenet121"


# CBAM Models
# TODO:


# Hyper-parameters
EPOCHS = 300
LOSS = torch.nn.CrossEntropyLoss()
LEARNING_RATE = 1e-4
OPTIMISER = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
BATCH_SIZE = 4


# Load data
# Train
# Transforms
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.1), scale=(0.95, 1.05), shear=0, resample=0, fillcolor=(0, 0, 0)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])

# Train Dataset
train_set = CBISDataset(base_data_path=train_dir, transform=train_transforms)

# Train Dataloader
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)


# Validation
# Transforms
val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])

# Validation Dataset
val_set = CBISDataset(base_data_path=val_dir, transform=val_transforms)

# Validation Dataloader
val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)



# Train model and save best weights on validation set
# Initialise min_train and min_val loss trackers
min_train_loss = np.inf
min_val_loss = np.inf

# Initialise losses arrays
train_losses = np.zeros((EPOCHS, ))
val_losses = np.zeros_like(train_losses)

# Initialise metrics arrays
train_metrics = np.zeros((EPOCHS, 4))
val_metrics = np.zeros_like(train_metrics)


# Go through the number of Epochs
for epoch in range(EPOCHS):
    # Epoch 
    print(f"Epoch: {epoch+1}")
    
    # Training Loop
    print("Training Phase")
    
    # Initialise lists to compute scores
    y_train_true = list()
    y_train_pred = list()


    # Running train loss
    run_train_loss = 0.0


    # Put model in training mode
    model.train()


    # Iterate through dataloader
    for batch_idx, (images, labels) in enumerate(train_loader):

        # Move data data anda model to GPU (or not)
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        model = model.to(DEVICE)


        # Find the loss and update the model parameters accordingly
        # Clear the gradients of all optimized variables
        OPTIMISER.zero_grad()


        # Forward pass: compute predicted outputs by passing inputs to the model
        logits = model(images)
        
        # Compute the batch loss
        # Using CrossEntropy w/ Softmax
        loss = LOSS(logits, labels)
        
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # Perform a single optimization step (parameter update)
        OPTIMISER.step()
        
        # Update batch losses
        run_train_loss += (loss.item() * images.size(0))

        # Concatenate lists
        y_train_true += list(labels.cpu().detach().numpy())
        
        # Using Softmax
        # Apply Softmax on Logits and get the argmax to get the predicted labels
        s_logits = torch.nn.Softmax(dim=1)(logits)
        s_logits = torch.argmax(s_logits, dim=1)
        y_train_pred += list(s_logits.cpu().detach().numpy())
    

    # Compute Average Train Loss
    avg_train_loss = run_train_loss/len(train_loader.dataset)

    # Compute Train Metrics
    train_acc = accuracy_score(y_true=y_train_true, y_pred=y_train_pred)
    # train_recall = recall_score(y_true=y_train_true, y_pred=y_train_pred)
    # train_precision = precision_score(y_true=y_train_true, y_pred=y_train_pred)
    # train_f1 = f1_score(y_true=y_train_true, y_pred=y_train_pred)

    # Print Statistics
    print(f"Train Loss: {avg_train_loss}\tTrain Accuracy: {train_acc}")
    # print(f"Train Loss: {avg_train_loss}\tTrain Accuracy: {train_acc}\tTrain Recall: {train_recall}\tTrain Precision: {train_precision}\tTrain F1-Score: {train_f1}")


    # Append values to the arrays
    # Train Loss
    train_losses[epoch] = avg_train_loss
    # Save it to directory
    fname = os.path.join(history_dir, f"{model_name}_tr_losses.npy")
    np.save(file=fname, arr=train_losses, allow_pickle=True)


    # Train Metrics
    # Acc
    train_metrics[epoch, 0] = train_acc
    # Recall
    # train_metrics[epoch, 1] = train_recall
    # Precision
    # train_metrics[epoch, 2] = train_precision
    # F1-Score
    # train_metrics[epoch, 3] = train_f1
    # Save it to directory
    fname = os.path.join(history_dir, f"{model_name}_tr_metrics.npy")
    np.save(file=fname, arr=train_metrics, allow_pickle=True)


    # Update Variables
    # Min Training Loss
    if avg_train_loss < min_train_loss:
        print(f"Train loss decreased from {min_train_loss} to {avg_train_loss}.")
        min_train_loss = avg_train_loss


    # Validation Loop
    print(f"Validation Phase")


    # Initialise lists to compute scores
    y_val_true = list()
    y_val_pred = list()


    # Running train loss
    run_val_loss = 0.0


    # Put model in evaluation mode
    model.eval()

    # Deactivate gradients
    with torch.no_grad():

        # Iterate through dataloader
        for batch_idx, (images, labels) in enumerate(val_loader):

            # Move data data anda model to GPU (or not)
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            model = model.to(DEVICE)

            # Forward pass: compute predicted outputs by passing inputs to the model
            logits = model(images)
            
            # Compute the batch loss
            # Using CrossEntropy w/ Softmax
            loss = LOSS(logits, labels)
            
            # Update batch losses
            run_val_loss += (loss.item() * images.size(0))

            # Concatenate lists
            y_val_true += list(labels.cpu().detach().numpy())
            
            # Using Softmax Activation
            # Apply Softmax on Logits and get the argmax to get the predicted labels
            s_logits = torch.nn.Softmax(dim=1)(logits)
            s_logits = torch.argmax(s_logits, dim=1)
            y_val_pred += list(s_logits.cpu().detach().numpy())

        

        # Compute Average Train Loss
        avg_val_loss = run_val_loss/len(val_loader.dataset)

        # Compute Training Accuracy
        val_acc = accuracy_score(y_true=y_val_true, y_pred=y_val_pred)
        # val_recall = recall_score(y_true=y_val_true, y_pred=y_val_pred)
        # val_precision = precision_score(y_true=y_val_true, y_pred=y_val_pred)
        # val_f1 = f1_score(y_true=y_val_true, y_pred=y_val_pred)

        # Print Statistics
        print(f"Validation Loss: {avg_val_loss}\tValidation Accuracy: {val_acc}")
        # print(f"Validation Loss: {avg_val_loss}\tValidation Accuracy: {val_acc}\tValidation Recall: {val_recall}\tValidation Precision: {val_precision}\tValidation F1-Score: {val_f1}")

        # Append values to the arrays
        # Train Loss
        val_losses[epoch] = avg_val_loss
        # Save it to directory
        fname = os.path.join(history_dir, f"{model_name}_val_losses.npy")
        np.save(file=fname, arr=val_losses, allow_pickle=True)


        # Train Metrics
        # Acc
        val_metrics[epoch, 0] = val_acc
        # Recall
        # val_metrics[epoch, 1] = val_recall
        # Precision
        # val_metrics[epoch, 2] = val_precision
        # F1-Score
        # val_metrics[epoch, 3] = val_f1
        # Save it to directory
        fname = os.path.join(history_dir, f"{model_name}_val_metrics.npy")
        np.save(file=fname, arr=val_metrics, allow_pickle=True)

        # Update Variables
        # Min validation loss and save if validation loss decreases
        if avg_val_loss < min_val_loss:
            print(f"Validation loss decreased from {min_val_loss} to {avg_val_loss}.")
            min_val_loss = avg_val_loss

            print("Saving best model on validation...")

            # Save checkpoint
            model_path = os.path.join(weights_dir, f"{model_name}_cbis.pt")
            torch.save(model.state_dict(), model_path)

            print(f"Successfully saved at: {model_path}")


# Finish statement
print("Finished.")
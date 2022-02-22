# Imports
import os
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from torchinfo import summary

# Sklearn Imports
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter


# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)


# Project Imports
from model_utilities_baseline import VGG16, DenseNet121, ResNet50
from model_utilities_se import SEResNet50, SEVGG16, SEDenseNet121
from model_utilities_cbam import CBAMResNet50, CBAMVGG16, CBAMDenseNet121
from transformers import ViTFeatureExtractor, ViTForImageClassification
from data_utilities import cbis_map_images_and_labels, mimic_map_images_and_labels, ph2_map_images_and_labels, CBISDataset, MIMICXRDataset, PH2Dataset, ISIC2020Dataset



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data set
parser.add_argument('--dataset', type=str, required=True, choices=["CBISDDSM", "ISIC2020", "MIMICCXR", "PH2"], help="Data set: CBISDDSM, ISIC2020, MIMICCXR, PH2")

# Model
parser.add_argument('--model', type=str, required=True, choices=["DenseNet121", "ResNet50", "VGG16", "SEDenseNet121", "SEResNet50", "SEVGG16", "CBAMDenseNet121", "CBAMResNet50", "CBAMVGG16", "ViT"], help='Model Name: DenseNet121, ResNet50, VGG16, SEDenseNet121, SEResNet50, SEVGG16, CBAMDenseNet121, CBAMResNet50, CBAMVGG16, ViT')

# Batch size
parser.add_argument('--batchsize', type=int, default=4, help="Batch-size for training and validation")

# Image size
parser.add_argument('--imgsize', type=int, default=224, help="Size of the image after transforms")

# Resize
parser.add_argument('--resize', type=str, choices=["direct_resize", "resizeshortest_randomcrop"], default="direct_resize", help="Resize data transformation")

# Number of epochs
parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs")

# Learning rate
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")

# Output directory
parser.add_argument("--outdir", type=str, default="results", help="Output directory")

# Number of workers
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")

# Save frequency
parser.add_argument("--save_freq", type=int, default=10, help="Frequency (in number of epochs) to save the model")

# Resume training
parser.add_argument("--resume", action="store_true", help="Resume training")
parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint from which to resume training")

# Number of layers (ViT)
parser.add_argument("--nr_layers", type=int, default=12, help="Number of hidden layers (only for ViT)")


# Parse the arguments
args = parser.parse_args()


# Resume training
if args.resume:
    assert args.ckpt is not None, "Please specify the model checkpoint when resume is True"

resume = args.resume

# Training checkpoint
ckpt = args.ckpt


# Dataset
dataset = args.dataset

# Results Directory
outdir = args.outdir

# Number of workers (threads)
workers = args.num_workers

# Number of training epochs
EPOCHS = args.epochs

# Learning rate
LEARNING_RATE = args.lr

# Batch size
BATCH_SIZE = args.batchsize

# Image size (after transforms)
IMG_SIZE = args.imgsize

# Save frquency
save_freq = args.save_freq

# Number of layers of the Visual Transformer
nr_layers = args.nr_layers

# Resize (data transforms)
resize_opt = args.resize



# Timestamp (to save results)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Create results directory
outdir = os.path.join(outdir, dataset.lower(), timestamp)
if not os.path.isdir(outdir):
    os.makedirs(outdir)


# Save training parameters
with open(os.path.join(outdir, "train_params.txt"), "w") as f:
    f.write(str(args))



# CBISDDSM
if dataset == "CBISDDSM":
    # Directories
    data_dir = "/ctm-hdd-pool01/tgoncalv/datasets/CBISPreprocDataset"
    #data_dir = "/BARRACUDA8T/DATASETS/CBIS_DDSM/"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
 
    imgs_labels, labels_dict, nr_classes = cbis_map_images_and_labels(dir=train_dir)


# MIMICXR
elif dataset == "MIMICCXR":
    # Directories
    data_dir = "/ctm-hdd-pool01/wjsilva19/MedIA"
    #data_dir = "/BARRACUDA8T/DATASETS/MIMIC_CXR_Pleural_Subset/"
    train_dir = os.path.join(data_dir, "Train_images_AP_resized")
    val_dir = os.path.join(data_dir, "Val_images_AP_resized")
    test_dir = os.path.join(data_dir, "Test_images_AP_resized")

    _, _, nr_classes = mimic_map_images_and_labels(base_data_path=train_dir, pickle_path=os.path.join(train_dir, "Annotations.pickle"))


# ISIC2020
elif dataset == "ISIC2020":
    # Directories
    data_dir = "/ctm-hdd-pool01/tgoncalv/datasets/ISIC2020/jpeg/train"
    # data_dir = "/BARRACUDA8T/DATASETS/ISIC2020/train"
    csv_fpath = "/ctm-hdd-pool01/tgoncalv/datasets/ISIC2020/train.csv"
    # csv_fpath = "/BARRACUDA8T/DATASETS/ISIC2020/train.csv"


# PH2
elif dataset == "PH2":
    # Directories
    data_dir = "/ctm-hdd-pool01/tgoncalv/datasets/PH2Dataset"

    # Data split
    # Get all the images, labels, and number of classes of PH2 Dataset
    ph2_imgs, ph2_labels, nr_classes = ph2_map_images_and_labels(data_dir)

    # Remove class 1
    ph2_imgs = ph2_imgs[ph2_labels!=1]
    ph2_labels = ph2_labels[ph2_labels!=1]


    # Split into train, validation and test (60%, 20%, 20%)
    # Train and Test
    ph2_imgs_train, ph2_imgs_test, ph2_labels_train, ph2_labels_test = train_test_split(ph2_imgs, ph2_labels, test_size=0.20, random_state=random_seed)
    # Train and Validation
    ph2_imgs_train, ph2_imgs_val, ph2_labels_train, ph2_labels_val = train_test_split(ph2_imgs_train, ph2_labels_train, test_size=0.25, random_state=random_seed)



# Results and Weights
weights_dir = os.path.join(outdir, "weights")
if not os.path.isdir(weights_dir):
    os.makedirs(weights_dir)


# History Files
history_dir = os.path.join(outdir, "history")
if not os.path.isdir(history_dir):
    os.makedirs(history_dir)


# Tensorboard
tbwritter = SummaryWriter(log_dir=os.path.join(outdir, "tensorboard"), flush_secs=30)


# Choose GPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Input Data Dimensions
img_nr_channels = 3
img_height = IMG_SIZE
img_width = IMG_SIZE


# Get the right model from the CLI
model = args.model
feature_extractor = None


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

# ViT
elif model == "ViT":
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=nr_classes, ignore_mismatched_sizes=True, num_hidden_layers=nr_layers, image_size=IMG_SIZE)
    model_name = "vit"
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')



# Move movel to device (CPU vs GPU)
model = model.to(DEVICE)


# Get model summary
model_summary = summary(model, (1, 3, IMG_SIZE, IMG_SIZE), device=DEVICE)
with open(os.path.join(outdir, "model_summary.txt"), 'w') as f:
    f.write(str(model_summary))


# Hyper-parameters
LOSS = torch.nn.CrossEntropyLoss(reduction="sum")
OPTIMISER = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Resume training from given checkpoint
init_epoch = 0
if resume:
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    OPTIMISER.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Resuming from {ckpt} at epoch {epoch}")


# Load data
# Train
# Transforms
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMG_SIZE if resize_opt == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.RandomCrop(IMG_SIZE if resize_opt == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.1), scale=(0.95, 1.05), shear=0, resample=0, fillcolor=(0, 0, 0)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=feature_extractor.image_mean if feature_extractor else MEAN, std=feature_extractor.image_std if feature_extractor else STD)
])

# Validation
# Transforms
val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMG_SIZE if resize_opt == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.RandomCrop(IMG_SIZE if resize_opt == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=feature_extractor.image_mean if feature_extractor else MEAN, std=feature_extractor.image_std if feature_extractor else STD)
])

# Datasets
# CBISDDSM
if dataset == "CBISDDSM":
    train_set = CBISDataset(base_data_path=train_dir, transform=train_transforms)
    val_set = CBISDataset(base_data_path=val_dir, transform=val_transforms)


# MIMCCXR
elif dataset == "MIMICCXR":
    train_set = MIMICXRDataset(base_data_path=train_dir, pickle_path=os.path.join(train_dir, "Annotations.pickle"), transform=train_transforms)
    val_set = MIMICXRDataset(base_data_path=val_dir, pickle_path=os.path.join(val_dir, "Annotations.pickle"), transform=val_transforms)


# ISIC2020
elif dataset == "ISIC2020":
    train_set = ISIC2020Dataset(base_data_path=data_dir, csv_path=csv_fpath, split='Train', random_seed=random_seed, transform=train_transforms, feature_extractor=feature_extractor)
    val_set = ISIC2020Dataset(base_data_path=data_dir, csv_path=csv_fpath, split='Validation', random_seed=random_seed, transform=val_transforms, feature_extractor=feature_extractor)


# PH2
elif dataset == "PH2":
    train_set = PH2Dataset(ph2_imgs=ph2_imgs_train, ph2_labels=ph2_labels_train, base_data_path=data_dir, transform=train_transforms)
    val_set = PH2Dataset(ph2_imgs=ph2_imgs_val, ph2_labels=ph2_labels_val, base_data_path=data_dir, transform=val_transforms)



# Dataloaders
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=workers)
val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=workers)


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
    y_train_true = np.empty((0), int)
    y_train_pred = torch.empty(0, dtype=torch.int32, device=DEVICE)


    # Running train loss
    run_train_loss = torch.tensor(0, dtype=torch.float64, device=DEVICE)


    # Put model in training mode
    model.train()

    # Iterate through dataloader
    for images, labels in tqdm(train_loader):
        # Concatenate lists
        y_train_true = np.append(y_train_true, labels.numpy(), axis=0)

        # Move data and model to GPU (or not)
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        # Find the loss and update the model parameters accordingly
        # Clear the gradients of all optimized variables
        OPTIMISER.zero_grad(set_to_none=True)


        # Forward pass: compute predicted outputs by passing inputs to the model
        if(isinstance(model, ViTForImageClassification)):
            out = model(pixel_values=images)
            logits = out.logits
        else:
            logits = model(images)

        
        # Compute the batch loss
        # Using CrossEntropy w/ Softmax
        loss = LOSS(logits, labels)

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # Perform a single optimization step (parameter update)
        OPTIMISER.step()
        
        # Update batch losses
        run_train_loss += loss

        # Using Softmax
        # Apply Softmax on Logits and get the argmax to get the predicted labels
        s_logits = torch.nn.Softmax(dim=1)(logits)
        s_logits = torch.argmax(s_logits, dim=1)
        y_train_pred = torch.cat((y_train_pred, s_logits))


    # Compute Average Train Loss
    avg_train_loss = run_train_loss/len(train_loader.dataset)
    

    # Compute Train Metrics
    y_train_pred = y_train_pred.cpu().detach().numpy()
    train_acc = accuracy_score(y_true=y_train_true, y_pred=y_train_pred)
    train_recall = recall_score(y_true=y_train_true, y_pred=y_train_pred, average='micro')
    train_precision = precision_score(y_true=y_train_true, y_pred=y_train_pred, average='micro')
    train_f1 = f1_score(y_true=y_train_true, y_pred=y_train_pred, average='micro')

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
    train_metrics[epoch, 1] = train_recall
    # Precision
    train_metrics[epoch, 2] = train_precision
    # F1-Score
    train_metrics[epoch, 3] = train_f1
    # Save it to directory
    fname = os.path.join(history_dir, f"{model_name}_tr_metrics.npy")
    np.save(file=fname, arr=train_metrics, allow_pickle=True)

    # Plot to Tensorboard
    tbwritter.add_scalar("loss/train", avg_train_loss, global_step=epoch)
    tbwritter.add_scalar("acc/train", train_acc, global_step=epoch)
    tbwritter.add_scalar("rec/train", train_recall, global_step=epoch)
    tbwritter.add_scalar("prec/train", train_precision, global_step=epoch)
    tbwritter.add_scalar("f1/train", train_f1, global_step=epoch)

    # Update Variables
    # Min Training Loss
    if avg_train_loss < min_train_loss:
        print(f"Train loss decreased from {min_train_loss} to {avg_train_loss}.")
        min_train_loss = avg_train_loss


    # Validation Loop
    print("Validation Phase")


    # Initialise lists to compute scores
    y_val_true = np.empty((0), int)
    y_val_pred = torch.empty(0, dtype=torch.int32, device=DEVICE)

    # Running train loss
    run_val_loss = 0.0

    # Put model in evaluation mode
    model.eval()

    # Deactivate gradients
    with torch.no_grad():

        # Iterate through dataloader
        for images, labels in tqdm(val_loader):
            y_val_true = np.append(y_val_true, labels.numpy(), axis=0)

            # Move data data anda model to GPU (or not)
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            # Forward pass: compute predicted outputs by passing inputs to the model
            if(isinstance(model, ViTForImageClassification)):
                out = model(pixel_values=images)
                logits = out.logits
            else:
                logits = model(images)
            
            # Compute the batch loss
            # Using CrossEntropy w/ Softmax
            loss = LOSS(logits, labels)
            
            # Update batch losses
            run_val_loss += loss


            # Using Softmax Activation
            # Apply Softmax on Logits and get the argmax to get the predicted labels
            s_logits = torch.nn.Softmax(dim=1)(logits)
            s_logits = torch.argmax(s_logits, dim=1)
            y_val_pred = torch.cat((y_val_pred, s_logits))

        

        # Compute Average Validation Loss
        avg_val_loss = run_val_loss/len(val_loader.dataset)

        # Compute Validation Accuracy
        y_val_pred = y_val_pred.cpu().detach().numpy()
        val_acc = accuracy_score(y_true=y_val_true, y_pred=y_val_pred)
        val_recall = recall_score(y_true=y_val_true, y_pred=y_val_pred, average='micro')
        val_precision = precision_score(y_true=y_val_true, y_pred=y_val_pred, average='micro')
        val_f1 = f1_score(y_true=y_val_true, y_pred=y_val_pred, average='micro')

        # Print Statistics
        print(f"Validation Loss: {avg_val_loss}\tValidation Accuracy: {val_acc}")
        # print(f"Validation Loss: {avg_val_loss}\tValidation Accuracy: {val_acc}\tValidation Recall: {val_recall}\tValidation Precision: {val_precision}\tValidation F1-Score: {val_f1}")

        # Append values to the arrays
        # Validation Loss
        val_losses[epoch] = avg_val_loss
        # Save it to directory
        fname = os.path.join(history_dir, f"{model_name}_val_losses.npy")
        np.save(file=fname, arr=val_losses, allow_pickle=True)


        # Train Metrics
        # Acc
        val_metrics[epoch, 0] = val_acc
        # Recall
        val_metrics[epoch, 1] = val_recall
        # Precision
        val_metrics[epoch, 2] = val_precision
        # F1-Score
        val_metrics[epoch, 3] = val_f1
        # Save it to directory
        fname = os.path.join(history_dir, f"{model_name}_val_metrics.npy")
        np.save(file=fname, arr=val_metrics, allow_pickle=True)

        # Plot to Tensorboard
        tbwritter.add_scalar("loss/val", avg_val_loss, global_step=epoch)
        tbwritter.add_scalar("acc/val", val_acc, global_step=epoch)
        tbwritter.add_scalar("rec/val", val_recall, global_step=epoch)
        tbwritter.add_scalar("prec/val", val_precision, global_step=epoch)
        tbwritter.add_scalar("f1/val", val_f1, global_step=epoch)

        # Update Variables
        # Min validation loss and save if validation loss decreases
        if avg_val_loss < min_val_loss:
            print(f"Validation loss decreased from {min_val_loss} to {avg_val_loss}.")
            min_val_loss = avg_val_loss

            print("Saving best model on validation...")

            # Save checkpoint
            model_path = os.path.join(weights_dir, f"{model_name}_{dataset.lower()}_best.pt")
            
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': OPTIMISER.state_dict(),
                'loss': avg_train_loss,
            }
            torch.save(save_dict, model_path)

            print(f"Successfully saved at: {model_path}")


        # Checkpoint loop/condition
        if epoch % save_freq == 0 and epoch > 0:

            # Save checkpoint
            model_path = os.path.join(weights_dir, f"{model_name}_{dataset.lower()}_{epoch:04}.pt")

            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': OPTIMISER.state_dict(),
                'loss': avg_train_loss,
            }
            torch.save(save_dict, model_path)


# Finish statement
print("Finished.")

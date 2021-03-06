# Imports
import os
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from torchinfo import summary

# Sklearn Imports
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

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
from model_utilities_baseline import DenseNet121, ResNet50
from model_utilities_se import SEDenseNet121, SEResNet50
from model_utilities_cbam import CBAMDenseNet121, CBAMResNet50
from data_utilities import APTOSDataset, ISIC2020Dataset, MIMICXRDataset
from transformers import DeiTFeatureExtractor
from transformer_explainability_utils.ViT_LRP import deit_tiny_patch16_224 as DeiT_Tiny


# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--data_dir', type=str, required=True, help="Directory of the data set.")

# Data set
parser.add_argument('--dataset', type=str, required=True, choices=["APTOS2019", "ISIC2020", "MIMICCXR"], help="Data set: APTOS2019, ISIC2020, MIMICCXR.")

# Model
parser.add_argument('--model', type=str, required=True, choices=["DenseNet121", "ResNet50", "SEDenseNet121", "SEResNet50", "CBAMDenseNet121", "CBAMResNet50", "DeiT-T-LRP"], help='Model Name: DenseNet121, ResNet50, SEDenseNet121, SEResNet50, CBAMDenseNet121, CBAMResNet50, DeiT-T-LRP.')

# Low Data Regimen
parser.add_argument('--low_data_regimen', action="store_true", help="Activate the low data regimen training.")
parser.add_argument('--perc_train', type=float, default=1, help="Percentage of training data to be used during training.")

# Batch size
parser.add_argument('--batchsize', type=int, default=4, help="Batch-size for training and validation")

# Image size
parser.add_argument('--imgsize', type=int, default=224, help="Size of the image after transforms")

# Resize
parser.add_argument('--resize', type=str, choices=["direct_resize", "resizeshortest_randomcrop"], default="direct_resize", help="Resize data transformation")

# Class Weights
parser.add_argument("--classweights", action="store_true", help="Weight loss with class imbalance")

# Number of epochs
parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs")

# Learning rate
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")

# Output directory
parser.add_argument("--outdir", type=str, default="results", help="Output directory")

# Number of workers
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")

# GPU ID
parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU")

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


# Data directory
data_dir = args.data_dir

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
model = args.model
model_name = model.lower()

# Low data regimen
low_data_regimen = args.low_data_regimen
perc_train = args.perc_train



# Timestamp (to save results)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
outdir = os.path.join(outdir, dataset.lower(), model_name, timestamp)
if not os.path.isdir(outdir):
    os.makedirs(outdir)


# Save training parameters
with open(os.path.join(outdir, "train_params.txt"), "w") as f:
    f.write(str(args))



# Load data
# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Input Data Dimensions
img_nr_channels = 3
img_height = IMG_SIZE
img_width = IMG_SIZE

# Feature extractor (for Transformers)
feature_extractor = None


# Train Transforms
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMG_SIZE if resize_opt == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.RandomCrop(IMG_SIZE if resize_opt == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.1), scale=(0.95, 1.05), shear=0, resample=0, fillcolor=(0, 0, 0)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=feature_extractor.image_mean if feature_extractor else MEAN, std=feature_extractor.image_std if feature_extractor else STD)
])


# Validation Transforms
val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMG_SIZE if resize_opt == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.RandomCrop(IMG_SIZE if resize_opt == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=feature_extractor.image_mean if feature_extractor else MEAN, std=feature_extractor.image_std if feature_extractor else STD)
])


# APTOS2019
if dataset == "APTOS":
    # Datasets
    train_set = APTOSDataset(base_data_path=data_dir, split="Train", resized=True, low_data_regimen=low_data_regimen, perc_train=perc_train, transform=train_transforms)
    val_set = APTOSDataset(base_data_path=data_dir, split="Validation", resized=True, low_data_regimen=low_data_regimen, perc_train=perc_train, transform=val_transforms)


# ISIC2020
elif dataset == "ISIC2020":
    # Datasets
    train_set = ISIC2020Dataset(base_data_path=data_dir, split='Train', random_seed=random_seed, resized=True, low_data_regimen=low_data_regimen, perc_train=perc_train, transform=train_transforms)
    val_set = ISIC2020Dataset(base_data_path=data_dir, split='Validation', random_seed=random_seed, resized=True, low_data_regimen=low_data_regimen, perc_train=perc_train, transform=val_transforms)


# MIMICXR
elif dataset == "MIMICCXR":
    # Directories
    train_dir = os.path.join(data_dir, "Train_images_AP_resized")
    val_dir = os.path.join(data_dir, "Val_images_AP_resized")
    test_dir = os.path.join(data_dir, "Test_images_AP_resized")

    # Datasets
    train_set = MIMICXRDataset(base_data_path=train_dir, pickle_path=os.path.join(train_dir, "Annotations.pickle"), resized=True, low_data_regimen=low_data_regimen, perc_train=perc_train, transform=train_transforms)
    val_set = MIMICXRDataset(base_data_path=val_dir, pickle_path=os.path.join(val_dir, "Annotations.pickle"), transform=val_transforms)



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
DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# Number of classes for models
nr_classes = train_set.nr_classes


# DenseNet121
if model == "DenseNet121":
    model = DenseNet121(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)

# ResNet50
elif model == "ResNet50":
    model = ResNet50(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)

# SEDenseNet121
elif model == "SEDenseNet121":
    model = SEDenseNet121(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)

# SEResNet50
elif model == "SEResNet50":
    model = SEResNet50(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)

# CBAMDenseNet121
elif model == "CBAMDenseNet121":
    model = CBAMDenseNet121(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)

# CBAMResNet50
elif model == "CBAMResNet50":
    model = CBAMResNet50(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)

# DeiT-Tiny (compatible with LRP)
elif model == "DeiT-T-LRP":
    model = DeiT_Tiny(pretrained=True, num_classes=nr_classes, input_size=(3, IMG_SIZE, IMG_SIZE), url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth")
    feature_extractor = DeiTFeatureExtractor.from_pretrained("facebook/deit-tiny-patch16-224")



# Put model into DEVICE (CPU or GPU)
model = model.to(DEVICE)


# Get model summary
try:
    model_summary = summary(model, (1, 3, IMG_SIZE, IMG_SIZE), device=DEVICE)

except:
    model_summary = str(model)


# Write into file
with open(os.path.join(outdir, "model_summary.txt"), 'w') as f:
    f.write(str(model_summary))



# Class weights for loss
if args.classweights:
    classes = np.array(range(nr_classes))
    cw = compute_class_weight('balanced', classes=classes, y=np.array(train_set.images_labels))
    cw = torch.from_numpy(cw).float().to(DEVICE)
    print(f"Using class weights {cw}")
else:
    cw = None



# Hyper-parameters
LOSS = torch.nn.CrossEntropyLoss(reduction="sum", weight=cw)
VAL_LOSS = torch.nn.CrossEntropyLoss(reduction="sum")
OPTIMISER = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)



# Resume training from given checkpoint
if resume:
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    OPTIMISER.load_state_dict(checkpoint['optimizer_state_dict'])
    init_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from {ckpt} at epoch {init_epoch}")
else:
    init_epoch = 0


# Dataloaders
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=workers)
val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=workers)



# Train model and save best weights on validation set
# Initialise min_train and min_val loss trackers
min_train_loss = np.inf
min_val_loss = np.inf

# Initialise losses arrays
train_losses = np.zeros((EPOCHS, ))
val_losses = np.zeros_like(train_losses)

# Initialise metrics arrays
train_metrics = np.zeros((EPOCHS, 5))
val_metrics = np.zeros_like(train_metrics)

# Go through the number of Epochs
for epoch in range(init_epoch, EPOCHS):
    # Epoch 
    print(f"Epoch: {epoch+1}")
    
    # Training Loop
    print("Training Phase")
    
    # Initialise lists to compute scores
    y_train_true = np.empty((0), int)
    y_train_pred = torch.empty(0, dtype=torch.int32, device=DEVICE)
    y_train_scores = torch.empty(0, dtype=torch.float, device=DEVICE) # save scores after softmax for roc auc


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
        logits = model(images)

        
        # Compute the batch loss
        # Using CrossEntropy w/ Softmax
        loss = LOSS(logits, labels)

        # Update batch losses
        run_train_loss += loss

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # Perform a single optimization step (parameter update)
        OPTIMISER.step()

        # Using Softmax
        # Apply Softmax on Logits and get the argmax to get the predicted labels
        s_logits = torch.nn.Softmax(dim=1)(logits)
        y_train_scores = torch.cat((y_train_scores, s_logits))
        s_logits = torch.argmax(s_logits, dim=1)
        y_train_pred = torch.cat((y_train_pred, s_logits))


    # Compute Average Train Loss
    avg_train_loss = run_train_loss/len(train_loader.dataset)
    

    # Compute Train Metrics
    y_train_pred = y_train_pred.cpu().detach().numpy()
    y_train_scores = y_train_scores.cpu().detach().numpy()
    train_acc = accuracy_score(y_true=y_train_true, y_pred=y_train_pred)
    train_recall = recall_score(y_true=y_train_true, y_pred=y_train_pred, average='micro')
    train_precision = precision_score(y_true=y_train_true, y_pred=y_train_pred, average='micro')
    train_f1 = f1_score(y_true=y_train_true, y_pred=y_train_pred, average='micro')
    train_auc = roc_auc_score(y_true=y_train_true, y_score=y_train_scores[:, 1], average='micro')

    # Print Statistics
    print(f"Train Loss: {avg_train_loss}\tTrain Accuracy: {train_acc}")


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
    # ROC AUC
    train_metrics[epoch, 4] = train_auc

    # Save it to directory
    fname = os.path.join(history_dir, f"{model_name}_tr_metrics.npy")
    np.save(file=fname, arr=train_metrics, allow_pickle=True)

    # Plot to Tensorboard
    tbwritter.add_scalar("loss/train", avg_train_loss, global_step=epoch)
    tbwritter.add_scalar("acc/train", train_acc, global_step=epoch)
    tbwritter.add_scalar("rec/train", train_recall, global_step=epoch)
    tbwritter.add_scalar("prec/train", train_precision, global_step=epoch)
    tbwritter.add_scalar("f1/train", train_f1, global_step=epoch)
    tbwritter.add_scalar("auc/train", train_auc, global_step=epoch)

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
    y_val_scores = torch.empty(0, dtype=torch.float, device=DEVICE) # save scores after softmax for roc auc

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
            logits = model(images)
            
            # Compute the batch loss
            # Using CrossEntropy w/ Softmax
            loss = VAL_LOSS(logits, labels)
            
            # Update batch losses
            run_val_loss += loss


            # Using Softmax Activation
            # Apply Softmax on Logits and get the argmax to get the predicted labels
            s_logits = torch.nn.Softmax(dim=1)(logits)                        
            y_val_scores = torch.cat((y_val_scores, s_logits))
            s_logits = torch.argmax(s_logits, dim=1)
            y_val_pred = torch.cat((y_val_pred, s_logits))

        

        # Compute Average Validation Loss
        avg_val_loss = run_val_loss/len(val_loader.dataset)

        # Compute Validation Accuracy
        y_val_pred = y_val_pred.cpu().detach().numpy()
        y_val_scores = y_val_scores.cpu().detach().numpy()
        val_acc = accuracy_score(y_true=y_val_true, y_pred=y_val_pred)
        val_recall = recall_score(y_true=y_val_true, y_pred=y_val_pred, average='micro')
        val_precision = precision_score(y_true=y_val_true, y_pred=y_val_pred, average='micro')
        val_f1 = f1_score(y_true=y_val_true, y_pred=y_val_pred, average='micro')
        val_auc = roc_auc_score(y_true=y_val_true, y_score=y_val_scores[:, 1], average='micro')

        # Print Statistics
        print(f"Validation Loss: {avg_val_loss}\tValidation Accuracy: {val_acc}")

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
        # ROC AUC
        val_metrics[epoch, 4] = val_auc

        # Save it to directory
        fname = os.path.join(history_dir, f"{model_name}_val_metrics.npy")
        np.save(file=fname, arr=val_metrics, allow_pickle=True)

        # Plot to Tensorboard
        tbwritter.add_scalar("loss/val", avg_val_loss, global_step=epoch)
        tbwritter.add_scalar("acc/val", val_acc, global_step=epoch)
        tbwritter.add_scalar("rec/val", val_recall, global_step=epoch)
        tbwritter.add_scalar("prec/val", val_precision, global_step=epoch)
        tbwritter.add_scalar("f1/val", val_f1, global_step=epoch)
        tbwritter.add_scalar("auc/val", val_auc, global_step=epoch)

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

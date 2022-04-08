# Source: https://github.com/hila-chefer/Transformer-Explainability

# Imports
import torch
import numpy as np
from numpy import *
import cv2

# Custom Imports
from model_utilities_xai import unnormalize



# Function: Generate attribution
def generate_attribution(image, attribution_generator, ground_truth_label, **kwargs):
    
    # Get original image
    original_image = np.transpose(image.cpu().detach().numpy(), (1, 2, 0))
    original_image = unnormalize(original_image, mean_array=kwargs["mean_array"], std_array=kwargs["std_array"])


    # Get label
    label = ground_truth_label.cpu().item()
    label = int(label)


    # Input to the xAI models
    input_img = image.unsqueeze(0)
    input_img.requires_grad = True


    # Put model in evaluation mode
    device = kwargs["device"]

    transformer_attribution = attribution_generator.generate_LRP(image.unsqueeze(0).to(device), method="transformer_attribution", index=label).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).to(device).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    # image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    # image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    
    
    return original_image, label, transformer_attribution



# Function: Show CAM on image
# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    

    return cam



# Function: Generate visualization
def generate_visualization(image_transformer_attribution, transformer_attribution):
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)


    return vis

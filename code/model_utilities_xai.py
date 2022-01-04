# Imports
import numpy as np

# PyTorch Imports
import torch

# Captum Imports
from captum.attr import DeepLift, LRP


# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)



# Function: A generic function that will be used for calling attribute on attribution algorithm defined in input
def attribute_image_features(model, algorithm, img_input, gt_label, **kwargs):
    model.zero_grad()
    tensor_attributions = algorithm.attribute(img_input, target=gt_label, **kwargs)


    return tensor_attributions



# Function: Function to unnormalize images
def unnormalize(image, mean_array, std_array):

    # Create a copy
    unnormalized_img = image.copy()

    # Get channels
    _, _, channels = unnormalized_img.shape


    for c in range(channels):
        unnormalized_img[:, :, c] = image[:, :, c] * std_array[c] + mean_array[c]


    return unnormalized_img



# Function: Generate post-hoc explanations
def generate_post_hoc_xmap(image, ground_truth_label, model, post_hoc_method, **kwargs):

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
    model.to(kwargs["device"])
    model.eval()


    # Select xAI method
    # DeepLift
    if post_hoc_method == "deeplift":

        # Create DeepLift framework
        xai_model = DeepLift(model)

        # Generate xAI post-hoc model
        xai_map = attribute_image_features(model, xai_model, input_img, label, abs=False)
        xai_map = np.transpose(xai_map.squeeze(0).cpu().detach().numpy(), (1, 2, 0))


    # LRP
    elif post_hoc_method == "lrp":

        # Create LRP framework
        xai_model = LRP(model)

        # Generate xAI post-hoc model
        xai_map = attribute_image_features(model, xai_model, input_img, label, abs=False)
        xai_map = np.transpose(xai_map.squeeze(0).cpu().detach().numpy(), (1, 2, 0))


    return original_image, label, xai_map

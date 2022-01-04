# Imports
import numpy as np
from collections import defaultdict

# PyTorch Imports
import torch
import torch.nn as nn

# Captum Imports
from captum.attr import DeepLift, LRP
from captum.attr._utils.custom_modules import Addition_Module
from captum.attr._utils.lrp_rules import EpsilonRule


# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)


# Class: CustomLRP
class CustomLRP(LRP):
    def _check_and_attach_rules(self) -> None:
        SUPPORTED_NON_LINEAR_LAYERS = [nn.ReLU, nn.Dropout, nn.Tanh, nn.Sigmoid]
        SUPPORTED_LAYERS_WITH_RULES = {
            nn.MaxPool1d: EpsilonRule,
            nn.MaxPool2d: EpsilonRule,
            nn.MaxPool3d: EpsilonRule,
            nn.Conv2d: EpsilonRule,
            nn.AvgPool2d: EpsilonRule,
            nn.AdaptiveAvgPool2d: EpsilonRule,
            nn.Linear: EpsilonRule,
            nn.BatchNorm2d: EpsilonRule,
            Addition_Module: EpsilonRule,
            }

        for layer in self.layers:
            if hasattr(layer, "rule"):
                layer.activations = {}  # type: ignore
                layer.rule.relevance_input = defaultdict(list)  # type: ignore
                layer.rule.relevance_output = {}  # type: ignore
                pass
            elif type(layer) in SUPPORTED_LAYERS_WITH_RULES.keys():
                layer.activations = {}  # type: ignore
                layer.rule = SUPPORTED_LAYERS_WITH_RULES[type(layer)]()  # type: ignore
                layer.rule.relevance_input = defaultdict(list)  # type: ignore
                layer.rule.relevance_output = {}  # type: ignore
            elif type(layer) in SUPPORTED_NON_LINEAR_LAYERS:
                layer.rule = None  # type: ignore
            else:
                raise TypeError(
                    (
                        f"Module of type {type(layer)} has no rule defined and no"
                        "default rule exists for this module type. Please, set a rule"
                        "explicitly for this module and assure that it is appropriate"
                        "for this type of layer."
                    )
                )



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


    # LRP
    elif post_hoc_method == "lrp":
        # Create LRP framework
        xai_model = CustomLRP(model)



    # Generate xAI post-hoc model
    xai_map = attribute_image_features(model, xai_model, input_img, label)
    xai_map = np.transpose(xai_map.squeeze(0).cpu().detach().numpy(), (1, 2, 0))


    return original_image, label, xai_map

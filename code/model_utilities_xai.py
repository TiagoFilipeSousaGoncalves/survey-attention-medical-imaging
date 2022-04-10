# Imports
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2

# PyTorch Imports
import torch
import torch.nn as nn

# Captum Imports
from captum.attr import DeepLift, LRP
from captum.attr._utils.custom_modules import Addition_Module
from captum.attr._utils.lrp_rules import EpsilonRule, PropagationRule

# Transformer xAI Imports
from transformer_explainability_utils.ViT_explanation_generator import LRP as DeiT_LRP

# Project Imports
from model_utilities_cbam import ChannelPool


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
            ChannelPool: EpsilonRule,
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
    

    def _check_rules(self) -> None:
        for module in self.model.modules():
            if hasattr(module, "rule"):
                if (
                    not isinstance(module.rule, PropagationRule)
                    and module.rule is not None
                ):
                    raise TypeError(
                        (
                            f"Please select propagation rules inherited from class "
                            f"PropagationRule for module: {module}"
                        )
                    )


    def _register_forward_hooks(self) -> None:
        SUPPORTED_NON_LINEAR_LAYERS = [nn.ReLU, nn.Dropout, nn.Tanh, nn.Sigmoid]

        for layer in self.layers:
            if type(layer) in SUPPORTED_NON_LINEAR_LAYERS:
                backward_handle = layer.register_backward_hook(
                    PropagationRule.backward_hook_activation
                )
                self.backward_handles.append(backward_handle)
            else:
                forward_handle = layer.register_forward_hook(
                    layer.rule.forward_hook  # type: ignore
                )
                self.forward_handles.append(forward_handle)
                if self.verbose:
                    print(f"Applied {layer.rule} on layer {layer}")


    def _register_weight_hooks(self) -> None:
        for layer in self.layers:
            if layer.rule is not None:
                forward_handle = layer.register_forward_hook(
                    layer.rule.forward_hook_weights  # type: ignore
                )
                self.forward_handles.append(forward_handle)


    def _register_pre_hooks(self) -> None:
        for layer in self.layers:
            if layer.rule is not None:
                forward_handle = layer.register_forward_pre_hook(
                    layer.rule.forward_pre_hook_activations  # type: ignore
                )
                self.forward_handles.append(forward_handle)



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



# Helper funtion to get figures to be shown after Captum VIZ
# https://stackoverflow.com/questions/49503869/attributeerror-while-trying-to-load-the-pickled-matplotlib-figure
def convert_figure(fig):

    # create a dummy figure and use its manager to display "fig"  
    dummy = plt.figure(figsize=(6,6))
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)



# Source: https://github.com/hila-chefer/Transformer-Explainability/blob/main/DeiT_example.ipynb
# We split the original functions into one to generate attributes and another to generate visualizations
# Function: Generate Transformer attribution array
def gen_transformer_att(image, ground_truth_label=None, attribution_generator=DeiT_LRP, device='cpu', **kwargs):

    # Get original image
    original_image = np.transpose(image.cpu().detach().numpy(), (1, 2, 0))
    original_image = unnormalize(original_image, mean_array=kwargs["mean_array"], std_array=kwargs["std_array"])


    # Get label
    label = ground_truth_label.cpu().item()
    label = int(label)


    # Input to the xAI models
    input_img = image.unsqueeze(0)
    input_img.requires_grad = True

    transformer_attribution = attribution_generator.generate_LRP(input_img.to(device), method="transformer_attribution", index=label).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).to(device).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    

    return original_image, label, transformer_attribution
    


# Function: Create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    
    
    return cam



# Function: Generate Transformer attributions visualization
def gen_transformer_att_vis(original_image, transformer_attribution):
    vis = show_cam_on_image(original_image, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    
    
    return vis

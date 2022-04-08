# Source: https://github.com/hila-chefer/Transformer-Explainability

# Imports
import torch
import numpy as np
from numpy import *
import cv2

# Custom Imports
from model_utilities_xai import unnormalize



# Function: Compute rollout between attention layers
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True) for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    
    
    return joint_attention



# Class: Transformer LRP
class LRP:
    def __init__(self, model, device):
        self.device = device
        self.model = model
        self.model.eval()

    def generate_LRP(self, input, index=None, method="transformer_attribution", is_ablation=False, start_layer=0):
        # output = self.model(input)
        output = self.model(pixel_values=input)
        output = output.logits
        
        kwargs = {"alpha": 1}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        # one_hot = torch.sum(one_hot.cuda() * output)
        one_hot = torch.sum(one_hot.to(self.device) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)


        return self.model.relprop(torch.tensor(one_hot_vector).to(self.device), method=method, is_ablation=is_ablation, start_layer=start_layer, **kwargs)



# Class: Baselines
class Baselines:
    def __init__(self, model, device):
        self.device = device
        self.model = model
        self.model.eval()


    def generate_cam_attn(self, input, index=None):
        # output = self.model(input.cuda(), register_hook=True)
        output = self.model(input, register_hook=True)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        # one_hot = torch.sum(one_hot.cuda() * output)
        one_hot = torch.sum(one_hot.to(self.device) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        #################### attn
        grad = self.model.blocks[-1].attn.get_attn_gradients()
        cam = self.model.blocks[-1].attn.get_attention_map()
        cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())


        return cam
        #################### attn


    def generate_rollout(self, input, start_layer=0):
        self.model(input)
        blocks = self.model.blocks
        all_layer_attentions = []
        
        for blk in blocks:
            attn_heads = blk.attn.get_attention_map()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
        
        rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        
        
        return rollout[:,0, 1:]



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
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)


    return vis

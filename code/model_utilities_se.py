# Imports
from typing import Type, Any, Callable, Union, List, Optional, Dict, cast

# PyTorch Imports
import torch
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet, VGG



# ResNet-50 Functions and Classes
# Helper Function (from: https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet18)
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> torch.nn.Conv2d:
    
    """3x3 convolution with padding"""

    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


# Helper Function (from: https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet18)
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> torch.nn.Conv2d:
    
    """1x1 convolution"""
    
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



# Squeeze-Excitation Layer (from: https://github.com/moskomule/senet.pytorch)
class SELayer(torch.nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        
        # Average Pooling Layer
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        
        # FC Layer
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channel // reduction, channel, bias=False),
            torch.nn.Sigmoid()
        )


    # Method: forward
    def forward(self, x):
        
        b, c, _, _ = x.size()
        
        y = self.avg_pool(x).view(b, c)
        
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y.expand_as(x)



# SE Bottleneck Layer for ResNet-50 (from: https://github.com/moskomule/senet.pytorch)
class SEBottleneck(torch.nn.Module):

    # Object attribute
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, *, reduction=16):
        super(SEBottleneck, self).__init__()
        
        # Init variables
        # self.expansion = 4

        # Conv + BN 1
        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        
        # Conv + BN 2
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        
        # Conv + BN 3
        self.conv3 = torch.nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(planes * 4)
        
        # ReLU
        self.relu = torch.nn.ReLU(inplace=True)
        
        # Squeeze-Excitation Block
        self.se = SELayer(planes * 4, reduction)
        
        # Downsample
        self.downsample = downsample
        
        # Stride
        self.stride = stride


    # Method: forward
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



# Model: SEResNet50 (adapted from: https://github.com/moskomule/senet.pytorch)
class SEResNet50(torch.nn.Module):
    def __init__(self, channels, height, width, nr_classes):
        super(SEResNet50, self).__init__()

        # Init variables
        self.channels = channels
        self.height = height
        self.width = width
        self.nr_classes = nr_classes
        self.weights_url = "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"


        # Init modules
        # Get model
        model = ResNet(SEBottleneck, [3, 4, 6, 3])
        model.avgpool = torch.nn.AdaptiveAvgPool2d(1)

        
        # Load pretrained weights
        model.load_state_dict(load_state_dict_from_url(self.weights_url))


        # Create our new models
        # We start by converting this model into a new one withouht the FC layers
        self.se_resnet50 = torch.nn.Sequential(*(list(model.children())[:-1]))

        # FC-Layers
        # Compute in_features
        _in_features = torch.rand(1, self.channels, self.height, self.width)
        _in_features = self.se_resnet50(_in_features)
        # print(_in_features.shape)
        _in_features = _in_features.size(0) * _in_features.size(1) * _in_features.size(2) * _in_features.size(3)

        # Create FC1 Layer for classification
        self.fc1 = torch.nn.Linear(in_features=_in_features, out_features=self.nr_classes)


        return
    

    def forward(self, inputs):
        # Compute SE features
        features = self.se_resnet50(inputs)

        # Reshape features
        features = torch.reshape(features, (features.size(0), -1))

        # FC1-Layer
        outputs = self.fc1(features)

        return outputs




# VGG-16 Functions and Classes
# Helper Function: Create VGG layers with SE Layer Blocks (adapted from: https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html#vgg16)
def make_layers_se(cfg: List[Union[str, int]], batch_norm: bool = False) -> torch.nn.Sequential:
    layers: List[torch.nn.Module] = []
    in_channels = 3
    
    for v in cfg:
        if v == 'M':
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
        
        else:
            v = cast(int, v)
            conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

            # Create an SE Layer
            se_layer = SELayer(channel=v)
            
            if batch_norm:
                layers += [conv2d, torch.nn.BatchNorm2d(v), torch.nn.ReLU(inplace=True)]
            
            else:
                # layers += [conv2d, torch.nn.ReLU(inplace=True)]
                layers += [conv2d, se_layer]
            
            in_channels = v

    
    return torch.nn.Sequential(*layers)



# Model: SEVGG-16 (adapted from: https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html#vgg16)
class SEVGG16(torch.nn.Module):
    def __init__(self, channels, height, width, nr_classes, pretrained=False):
        super(SEVGG16, self).__init__()

        # Init variables
        self.channels = channels
        self.height = height
        self.width = width
        self.nr_classes = nr_classes
        self.weights_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        self.cfgs: Dict[str, List[Union[str, int]]] = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
            }


        # Init modules
        # Build SEVGG16 to extract features
        if pretrained:
            kwargs = dict()
            kwargs['init_weights'] = False

            model = VGG(make_layers_se(self.cfgs['D'], batch_norm=False), **kwargs)

            state_dict = load_state_dict_from_url(self.weights_url, progress=True)
            
            # To prevent problems with keys: strict=False
            model.load_state_dict(state_dict, strict=False)
        

        else:
            model = VGG(make_layers_se(self.cfgs['D'], batch_norm=False))


        # Get the features of the model with the SE Layer
        self.se_vgg16 = model.features


        # FC-Layers
        # Compute in_features
        _in_features = torch.rand(1, self.channels, self.height, self.width)
        _in_features = self.se_vgg16(_in_features)
        _in_features = _in_features.size(0) * _in_features.size(1) * _in_features.size(2) * _in_features.size(3)

        # Create FC1 Layer for classification
        self.fc1 = torch.nn.Linear(in_features=_in_features, out_features=self.nr_classes)


        return
    

    def forward(self, inputs):
        # Compute Backbone features
        features = self.se_vgg16(inputs)

        # Reshape features
        features = torch.reshape(features, (features.size(0), -1))

        # FC1-Layer
        outputs = self.fc1(features)


        return outputs
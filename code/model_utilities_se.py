# Imports
from collections import OrderedDict
import re
from typing import Union, List, Dict, cast, Tuple

# PyTorch Imports
import torch
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet, VGG
import torch.utils.checkpoint as cp



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
            # layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]

            # Replace a MaxPool2d with an Attention Layer
            layers += [SELayer(channel=in_channels)]
        
        else:
            v = cast(int, v)
            conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

            # Create an SE Layer
            # se_layer = SELayer(channel=v)
            
            if batch_norm:
                layers += [conv2d, torch.nn.BatchNorm2d(v), torch.nn.ReLU(inplace=True)]
            
            else:
                layers += [conv2d, torch.nn.ReLU(inplace=True)]
                # layers += [conv2d, se_layer]
            
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
            model = VGG(make_layers_se(self.cfgs['D'], batch_norm=False), init_weights=False)


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



# DenseNet-121 Functions and Classes (adapted from: https://pytorch.org/vision/stable/_modules/torchvision/models/densenet.html#densenet121)
# Helper Class: Dense Layer Class
class _DenseLayer(torch.nn.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        memory_efficient: bool = False
    ) -> None:
        super(_DenseLayer, self).__init__()
        self.norm1: torch.nn.BatchNorm2d
        self.add_module('norm1', torch.nn.BatchNorm2d(num_input_features))
        self.relu1: torch.nn.ReLU
        self.add_module('relu1', torch.nn.ReLU(inplace=True))
        self.conv1: torch.nn.Conv2d
        self.add_module('conv1', torch.nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False))
        self.norm2: torch.nn.BatchNorm2d
        self.add_module('norm2', torch.nn.BatchNorm2d(bn_size * growth_rate))
        self.relu2: torch.nn.ReLU
        self.add_module('relu2', torch.nn.ReLU(inplace=True))
        self.conv2: torch.nn.Conv2d
        self.add_module('conv2', torch.nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[torch.Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[torch.Tensor]) -> torch.Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[torch.Tensor]) -> torch.Tensor:
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: F811
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = torch.nn.functional.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features



# Helper Class: Dense Block Class
class _DenseBlock(torch.nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False
    ) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: torch.Tensor) -> torch.Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)



# Helper Class: Transition Layer Class
class _Transition(torch.nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        self.add_module('norm', torch.nn.BatchNorm2d(num_input_features))
        self.add_module('relu', torch.nn.ReLU(inplace=True))
        self.add_module('conv', torch.nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', torch.nn.AvgPool2d(kernel_size=2, stride=2))



# Helper Class: SEDenseNet Class
class SEDenseNet(torch.nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
        memory_efficient: bool = False
    ) -> None:

        super(SEDenseNet, self).__init__()

        # First convolution
        self.features = torch.nn.Sequential(OrderedDict([
            ('conv0', torch.nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', torch.nn.BatchNorm2d(num_init_features)),
            ('relu0', torch.nn.ReLU(inplace=True)),
            ('pool0', torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            

            # Current number of features
            num_features = num_features + num_layers * growth_rate


            # We add SELayer between each Dense Block and Transition Block
            selayer = SELayer(channel=num_features)
            self.features.add_module('selayer%d' % (i + 1), selayer)


            # Each Transition Block
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', torch.nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = torch.nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if m != None:
                if isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, torch.nn.BatchNorm2d):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, torch.nn.Linear):
                    try:
                        torch.nn.init.constant_(m.bias, 0)
                    except:
                        pass


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = torch.nn.functional.relu(features, inplace=True)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out



# Model: SEDenseNet-121 (adapted from source "_densenet" and "densenet121")
class SEDenseNet121(torch.nn.Module):
    def __init__(self, channels, height, width, nr_classes, pretrained=False):
        super(SEDenseNet121, self).__init__()

        # Init variables
        self.channels = channels
        self.height = height
        self.width = width
        self.nr_classes = nr_classes
        self.weigths_url = 'https://download.pytorch.org/models/densenet121-a639ec97.pth'
        self.arch = 'densenet121'
        self.growth_rate = 32
        self.block_config = (6, 12, 24, 16)
        self.num_init_features = 64
        self.pretrained = pretrained
        self.progress = True


        # Create model
        model = SEDenseNet(self.growth_rate, self.block_config, self.num_init_features)


        # If pretrained (adapted from source "_load_state_dict function")
        if self.pretrained:
            # '.'s are no longer allowed in module names, but previous _DenseLayer
            # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
            # They are also in the checkpoints in model_urls. This pattern is used
            # to find such keys.
            pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            state_dict = load_state_dict_from_url(self.weigths_url, progress=self.progress)

            # Go through keys
            for key in list(state_dict.keys()):
                res = pattern.match(key)

                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            

            # Load state dict
            model.load_state_dict(state_dict)
        

        # Else (i.e., no pretraining)
        self.sedensenet121 = model.features
        
        
        # FC-Layers
        # Compute in_features
        _in_features = torch.rand(1, self.channels, self.height, self.width)
        _in_features = self.sedensenet121(_in_features)
        _in_features = _in_features.size(0) * _in_features.size(1) * _in_features.size(2) * _in_features.size(3)

        # Create FC1 Layer for classification
        self.fc1 = torch.nn.Linear(in_features=_in_features, out_features=self.nr_classes)


        return
    

    def forward(self, inputs):
        # Compute Backbone features
        features = self.sedensenet121(inputs)

        # Reshape features
        features = torch.reshape(features, (features.size(0), -1))

        # FC1-Layer
        outputs = self.fc1(features)


        return outputs
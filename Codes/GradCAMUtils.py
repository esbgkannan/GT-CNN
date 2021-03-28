import cv2
import torch

layer_finders = {}


def register_layer_finder(model_type):
    def register(func):
        layer_finders[model_type] = func
        return func
    return register


def visualize_cam(mask, img, alpha=1.0):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]

    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result


@register_layer_finder('1dcnn')
def find_1dcnn_layer(arch, target_layer_name):
    """Find alexnet layer to calculate GradCAM and GradCAM++
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_0'
            target_layer_name = 'classifier'
            target_layer_name = 'classifier_0'
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """

    hierarchy = target_layer_name.rsplit("_",1)
    

    if target_layer_name.rsplit("_",1)[0] == "conv_1":
        target_layer = arch.conv_1
    elif target_layer_name.rsplit("_",1)[0] == "conv_2":
        target_layer = arch.conv_2
    elif target_layer_name.rsplit("_",1)[0] == "conv_3":
        target_layer = arch.conv_3
    elif target_layer_name.rsplit("_",1)[0] == "maxpool_4":
        target_layer = arch.maxpool_4
        
    elif target_layer_name.rsplit("_",1)[0] == "maxpool_5":
        target_layer = arch.maxpool_5
        
    elif target_layer_name.rsplit("_",1)[0] == "maxpool_6":
        target_layer = arch.maxpool_6
        
    elif target_layer_name == "conv1":
        target_layer = arch.conv1
    elif target_layer_name == "conv2":
        target_layer = arch.conv2
    elif target_layer_name == "conv3":
        target_layer = arch.conv3
#     print(target_layer)
    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer

@register_layer_finder('ResNet')
def find_ResNet_layer(arch, target_layer_name):
    """Find alexnet layer to calculate GradCAM and GradCAM++
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_0'
            target_layer_name = 'classifier'
            target_layer_name = 'classifier_0'
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """

    hierarchy = target_layer_name.rsplit("_",1)
    

    if target_layer_name.rsplit("_",1)[0] == "layer1":
        target_layer = arch.layer1
    elif target_layer_name.rsplit("_",1)[0] == "layer2":
        target_layer = arch.layer2
    elif target_layer_name.rsplit("_",1)[0] == "layer3":
        target_layer = arch.layer3
    elif target_layer_name.rsplit("_",1)[0] == "layer4":
        target_layer = arch.layer4
        
#     print(target_layer)
    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer


@register_layer_finder('1dCNNAttention')
def find_CA_layer(arch, target_layer_name):
    """Find alexnet layer to calculate GradCAM and GradCAM++
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_0'
            target_layer_name = 'classifier'
            target_layer_name = 'classifier_0'
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """

    hierarchy = target_layer_name.rsplit("_",1)
    

    if target_layer_name == "layer1":
        return arch.layer1
    elif target_layer_name == "layer2":
        return arch.layer2
    elif target_layer_name == "layer3":
        return arch.layer3
        
    hierarchy = target_layer_name.rsplit("_",1)
    
    if target_layer_name.rsplit("_",1)[0] == "maxpool_4":
        target_layer = arch.maxpool_4
        
    elif target_layer_name.rsplit("_",1)[0] == "maxpool_5":
        target_layer = arch.maxpool_5
        
    elif target_layer_name.rsplit("_",1)[0] == "maxpool_6":
        target_layer = arch.maxpool_6
        
    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]        

    return target_layer


def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)


def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)

    def do(self, tensor):
        return normalize(tensor, self.mean, self.std)

    def undo(self, tensor):
        return denormalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
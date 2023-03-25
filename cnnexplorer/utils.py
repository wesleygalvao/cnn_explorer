"""Utility functions and classes for working with Pytorch modules"""

# TODO: A better documentation for all the functions and classes.

from torch import nn
import torch
from torch.nn.functional import interpolate
from torch.functional import F
from torch.nn.functional import avg_pool2d

import torchvision.transforms as T

from collections import OrderedDict
import functools
import scipy
from scipy import ndimage, misc
import pyprog

import numpy as np


class ActivationSampler(nn.Module):
    """Generates a hook for sampling a layer activation. Can be used as
    a function or as a module.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model from which layer information will be extracted.

    Example
    -------
    >>> model = nn.Sequential(nn.Conv2d(1, 1, 1), nn.ReLU())
    >>> input = torch.zeros((1, 1, 2, 2))
    >>>
    >>> layer_in_model = model[0]
    >>> sampler = ActivationSampler(layer_in_model)
    >>> output = model(input)
    >>> layer_activation = sampler()

    """

    def __init__(self, model: nn.Module):
        """Initialize the ActivationSampler object."""
        super().__init__()
        self.model_name = model.__class__.__name__
        self.activation = None
        model.register_forward_hook(self.get_hook())

    def forward(self, x=None):
        """Return the activation of the layer."""
        return self.activation

    def get_hook(self):
        """Return a hook for sampling the layer activation."""
        def hook(model, input, output):
            """Hook for sampling the layer activation."""
            self.activation = output
        return hook

    def extra_repr(self):
        """Return a string representation of the object."""
        return f'{self.model_name}'


def get_submodule_str(model, module):
    """Return a string representation of `module` in the form 'layer_name.sublayer_name...'
    """

    for name, curr_module in model.named_modules():
        if curr_module is module:
            module_name = name
            break

    return module_name


# Source: https://stackoverflow.com/a/31174427
def get_submodule(model, path: str, *default):
    """A function to get nested subobjects from model, given the nested attribute (submodule path) as string.  

    Parameters
    ----------
    
    path: 'attr1.attr2.etc'
    
    default: Optional default value, at any point in the path

    Returns 
    ----------
    model.attr1.attr2.etc
    :param path:
    :param model:
    """

    attrs = path.split('.')
    try:
        return functools.reduce(getattr, attrs, model)
    except AttributeError:
        if default:
            return default[0]
        raise


def get_output_shape(model, img_shape):
    input_img = torch.zeros(img_shape)[None, None]
    input_img = input_img.to(next(model.parameters()).device)
    output = model(input_img)
    return output[0].shape


def get_number_maps(model, layer):
    """Return the number of feature maps of a layer, given the model and its module
    """
    # Get Activations given a sub module
    act = ActivationSampler(layer)
    img = torch.zeros((1, 1, 2, 2)).to('cuda')
    # Pass img through model 
    model(img)
    n_maps = act.activation.shape[1]
    # Return number of feature maps
    return n_maps


# Discontinued 
# def get_number_maps_(model, module):
#     """Return the number of feature maps of a layer, given the model and its module
#     """
#     # Get sub model
#     sub_module = model_up_to(model, module)
#     # Get the output shape of sub_module
#     n_maps = get_output_shape(sub_module, (1, 1))
#     # Return number of feature maps
#     return n_maps[0]

def model_up_to(model, module):
    """Return a new model with all layers in model up to layer `module`."""

    split_module_str = get_submodule_str(model, module)
    split_modules_names = split_module_str.split('.')
    module = model
    splitted_model = []
    name_prefix = ''
    for idx, split_module_name in enumerate(split_modules_names):
        for child_module_name, child_module in module.named_children():
            if child_module_name == split_module_name:
                if idx == len(split_modules_names) - 1:
                    # If at last module
                    full_name = f'{name_prefix}{child_module_name}'
                    splitted_model.append((full_name, child_module))
                module = child_module
                name_prefix += split_module_name + '_'
                break
            else:
                full_name = f'{name_prefix}{child_module_name}'
                splitted_model.append((full_name, child_module))

    new_model = torch.nn.Sequential(OrderedDict(splitted_model))

    return new_model


def get_image_label(dataset, img_idx):
    """Return original the image, equalized image, ground truth 
    and file name, given a dataset and image index. 

    """
    # Get original image from dataset
    img_or, _ = dataset.get_item(img_idx)
    img_or = np.array(img_or)
    # Get equalized image and its label
    img_equalized, label = dataset[img_idx]
    # Get filename
    filename = dataset.img_file_paths[img_idx].stem

    return img_or, img_equalized, label, filename


def feature_maps_interp(feature_maps, mode_='linear', scale_factor_=2):
    """Feature maps interpolation"""

    if mode_ == 'linear':
        # Run interpolation on feature maps with chosen scale factor
        feature_maps_interpolated = F.interpolate(feature_maps,
                                                  scale_factor=scale_factor_,
                                                  mode=mode_)

        feature_maps_interpolated = feature_maps_interpolated.permute(0, 2, 1)

        feature_maps_interpolated = F.interpolate(feature_maps_interpolated,
                                                  scale_factor=scale_factor_,
                                                  mode=mode_)

        feature_maps_interpolated = feature_maps_interpolated.permute(0, 2, 1)

    if mode_ == 'bicubic':
        # Change tensor dimension to [batch_size == 1, channels, height, width]
        feature_maps = feature_maps[None]
        # Run interpolation on feature maps with chosen scale factor
        feature_maps_interpolated = F.interpolate(feature_maps,
                                                  scale_factor=scale_factor_,
                                                  mode=mode_)
        feature_maps_interpolated = torch.squeeze(feature_maps_interpolated)

    return feature_maps_interpolated


def image_sampling(image, kernel_size_):
    # Change tensor dimension to [batch_size == 1, channels, height, width]
    image = image[None]
    # Run interpolation on feature maps with chosen scale factor
    image_sampled = torch.nn.functional.avg_pool2d(image,
                                                   kernel_size=kernel_size_)
    image_sampled = torch.squeeze(image_sampled)

    return image_sampled


def binary_dilation(binary_image, iterations_level):
    """Dilation function over a binary image"""

    dilated_image = scipy.ndimage.binary_dilation(binary_image, iterations=iterations_level)
    dilated_image = dilated_image.astype(int)
    dilated_image = torch.from_numpy(dilated_image)

    return dilated_image


def feature_maps_masking(layer_feature_maps, mask):
    """
  Apply the dilation and masking over all the feature maps of ResUNet layer. 
  The dilation uses iterations_level to define the level of iterations.  
  """
    # Mask all feature maps of layer_feature_maps
    feature_maps_masked = layer_feature_maps * mask

    return feature_maps_masked


def crop_feature_maps(layers_fm_list, top=0, left=0, height=128, width=128):
    """Given a list of feature maps, it is cropped at specified location and output.
    Returns a list of cropped feature maps. 

    It uses torchvision.transforms.functional.crop() function by PyTorch. 

    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then cropped.

    Parameters
    ----------
        layers_fm_list (PIL Image or Tensor): A list of feature maps to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        List of PIL Image or Tensor: Cropped feature maps
    """
    cropped_fm_list = []
    for layer in layers_fm_list:
        cropped_fm_list.append(T.functional.crop(layer, top=top, left=left, height=height, width=width))

    return cropped_fm_list


def remove_prefix(text, prefix):
    '''Remove the prefix from the text, given a substring pattern
  '''
    return text[text.startswith(prefix) and len(prefix):]

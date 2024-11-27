# ====================================================================== #
# In this script we apply the "forward" measurement models as defined in 
# https://openreview.net/forum?id=OnD9zGAGT0k
# ====================================================================== #
import os
import torch
import matplotlib.pyplot as plt
import numpy as np 
import torch.nn.functional as F
import argparse
import yaml   
from data.blur_models.kernel_encoding.kernel_wizard import KernelWizard
from data.motionblur import Kernel

class RandomInpainting(object):
    """ 
    Implements the random-inpainting forward measurement model
    - y ~ N(Px, sigma**2 * I) if noise_model = "gaussian",
    - y ~ Poisson(Px) if noise_model = "poisson".
    
    Here P is the masking matrix given by randomly dropping 92% of pixels

    Parameters
    ----------
    - sigma: float = variance of Gaussian noise
    - noise_model: str = which model to implement "gaussian" | "poisson"
    """
    def __init__(self, noise_model="gaussian", sigma=1.):
        self.sigma = sigma
        if (noise_model != "gaussian") and (noise_model != "poisson"):
            print(f"Noise model {noise_model} not implemented! Use 'gaussian' or 'poisson'.")
            return ValueError 
        self.noise_model = noise_model

    def __call__(self, tensor):
        _, n, d = tensor.shape
        mask = torch.rand((n,d)) > 0.5
        x = tensor * mask
        if self.noise_model == "gaussian":
            return x + torch.randn(size=x.size())*self.sigma**2
        elif self.noise_model == "poisson":
            return torch.poisson(x) 
        else: 
            return None
        
    def __repr__(self):
        return self.__class__.__name__ + f"(mean={0}, std={1})"


class BoxInpainting(object):
    """ 
    Implements the box inpainting forward measurement model
    - y ~ N(y|Px, sigma**2 * I) if noise_model = "gaussian"
    - y ~ Poisson(Px; lamb) if noise_model = "poisson"
    
    TODO: Same box is generated for all images in the batch. 
          Change or keep it as it is. 
    """
    def __init__(self, noise_model="gaussian", sigma=1.):
        self.sigma = sigma
        self.noise_model = noise_model
        if (noise_model != "gaussian") and (noise_model != "poisson"):
            print(f"Noise model {noise_model} not implemented! Use 'gaussian' or 'poisson' ")
            return ValueError 

    def box(self, x):
        """Generate random coordinates for a 128x128 box that fits within the image"""
        _, h, w = x.shape

        max_x = h - 128 if h >= 128 else 0
        max_y = w - 128 if w >= 128 else 0
        
        x1 = torch.randint(0, max(1, max_x), (1,)).item()
        x2 = torch.randint(0, max(1, max_y), (1,)).item()
        
        box_h = min(128, h)
        box_w = min(128, w)
        
        return x1, x2, box_h, box_w

    def __call__(self, tensor):
        _, h, w = tensor.shape  # tensor shape is [channels, height, width]
        x = tensor
        x1, x2, box_h, box_w = self.box(x)
        
        if self.noise_model == "gaussian":
            x[:, x1:x1 + box_h, x2:x2 + box_w] = torch.randn((3, box_h, box_w)) * self.sigma**2
            return x
        elif self.noise_model == "poisson":
            x[:, x1:x1 + box_h, x2:x2 + box_w] = torch.poisson(x[:, x1:x1 + box_h, x2:x2 + box_w])
            return x  
        else: 
            return None


class LinearBlurring(object):
    """
    Implements the inpainting forward measurement model
    y ~ N(y| C^phi * x, sigma**2 * I) if self.noise_model = "gaussian"
    y ~ Poisson(C^phi * x) if self.noise_model = "poisson
    and C^phi is the convolution matrix of the given convolution kernel phi
    Parameters: 
    ----------
        sigma = variance of Gaussian
        phi = convolutional kernel
        C = block hankel matrix
    """

    def __init__(self, sigma=1., ):
        #self.sigma = sigma
        #self.phi = torch.convolution() ###  <- think about this 
        pass


class SuperResolution(object):

    def __init__(self, downscale_factor=0.25, upscale_factor=4, noise_model="gaussian", sigma=0.05):
        self.downscale_factor = downscale_factor
        self.upscale_factor   = upscale_factor
        if (noise_model != "gaussian") and (noise_model != "poisson"):
            print(f"Noise model {noise_model} not implemented! Use 'gaussian' or 'poisson'.")
            return ValueError 
        self.noise_model = noise_model
        self.sigma = sigma

    def __call__(self, tensor):
        # returns a downsampled image with added gaussian noise. I.e image from 256x256 -> 64x64 + gaussian noise
        downsampled_image = self.bicubic_downsample(tensor)
        noisy_downsample = self.add_gaussian_noise(downsampled_image, 64)
        return noisy_downsample
    
    def denormalize(self, tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean
    
    def bicubic_downsample(self, image):
        # batch_size, channels, height, width = images.shape
        image = image.unsqueeze(0)
        downsampled_image = torch.nn.functional.interpolate(image, scale_factor=self.downscale_factor, mode='bicubic', align_corners=True)
        downsampled_image = downsampled_image.squeeze(0)
        return downsampled_image
    
    def bicubic_upsample(self, image):
        image = image.unsqueeze(0)
        upsampled_image = torch.nn.functional.interpolate(image, scale_factor=self.upscale_factor, mode='bicubic', align_corners=True)
        upsampled_image = upsampled_image.squeeze(0)
        return upsampled_image
    
    def add_gaussian_noise(self, image, image_size):
        image += torch.randn((3, image_size, image_size)) * self.sigma
        return image
    

    def upsample_with_noise(self, low_res_img):
        _, h, w = low_res_img.shape  # tensor shape is [channels, height, width]
        x_low_res = low_res_img
        x_upscaled = torch.zeros((3, h*self.upscale_factor, w*self.upscale_factor))
        # print("Shape of upscaled image tensor: ", x_upscaled.shape)
        x_upscaled[:, ::self.upscale_factor, ::self.upscale_factor] = x_low_res
        mask = (x_upscaled == 0).float()  # Shape: (1, 3, 256, 256)

        if self.noise_model == "gaussian":
            noise = torch.randn_like(x_upscaled)  # Standard Gaussian noise
        elif self.noise_model == "poisson":
             noise = torch.poisson(x_upscaled) 
        else: 
            return None
        noise = torch.randn_like(x_upscaled)  # Standard Gaussian noise
        noise_intensity = 0.1  # Adjust the noise intensity as needed
        x_upscaled_with_noise = x_upscaled + noise * mask * noise_intensity

        # Step 6: Clamp the result to ensure pixel values remain valid (e.g., [0, 1])
        x_upscaled_with_noise = torch.clamp(x_upscaled_with_noise, 0, 255)

        return x_upscaled_with_noise, mask

class NonLinearBlurring(object):
    """
    Implements the non-linear blurring forward measurement model.
    y ~ N(y| F(x,k), sigma**2 * I) if self.noise_model = "gaussian"
    y ~ Poisson(F(x,k)) if self.noise_model = "poisson"
    F(x,k) is a external pretrained model from (see link)
        link: https://github.com/VinAIResearch/blur-kernel-space-exploring  
    """
    def __init__(self, noise_model="gaussian", sigma=1.):
        self.sigma = sigma

    def generate_blur(self, tensor):
        # NOTE: From https://github.com/VinAIResearch/blur-kernel-space-exploring/blob/main/generate_blur.py 
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yml_path = os.path.join(current_dir, "blur_models", "default.yml") 
        print(yml_path)
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        # Initializing mode
        with open(yml_path, "r") as f:
            opt = yaml.load(f, Loader=yaml.SafeLoader)["KernelWizard"]
            model_path = opt["pretrained"]
        model = KernelWizard(opt)
        model.eval()
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        with torch.no_grad():
            kernel = torch.randn((1, 512, 2, 2)) * 1.2
            # NOTE: The normalization transformations below was taken from DPS repo. 
            tensor = (tensor + 1.0) / 2.0  #[-1, 1] -> [0, 1]
            LQ_tensor = model.adaptKernel(tensor, kernel=kernel)
            LQ_tensor = (LQ_tensor * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]

        return LQ_tensor

    def __call__(self, tensor):
        blurred_img = self.generate_blur(tensor)
        x = blurred_img

        if self.noise_model == "gaussian":
            return x + torch.randn(size=x.size())*self.sigma**2
        elif self.noise_model == "poisson":
            return torch.poisson(x) 
        else: 
            return None
        
class GaussianBlur(object):
    """
    Implements the Gaussian convolution (Gaussian noise) forward measurement model.
    The Gaussian kernel is 61x61 and convolved with the ground truth image to produce 
    the measurement. 
    """
    def __init__(self, kernel_size, sigma):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def gaussian_kernel(self):
        size = self.kernel_size[0]
        x = torch.linspace(-(size // 2), size // 2, size)
        y = torch.linspace(-(size // 2), size // 2, size)
        x, y = torch.meshgrid(x, y, indexing='xy')
        kernel = torch.exp(-(x**2 + y**2) / (2 * self.sigma**2))
        kernel /= kernel.sum()  
        return kernel

    def __call__(self, tensor):
        kernel = self.gaussian_kernel()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        num_channels = tensor.size(0)
        kernel = kernel.repeat(num_channels, 1, 1, 1)
        blurred = F.conv2d(tensor, weight=kernel, padding=self.kernel_size[0] // 2, groups=3)
        return blurred
    
class MotionBlur(object):
    """
    Implements the motion blur forward measurement model. 
    The motion blur kernel is an external kernel from (see link)
        link: https://github.com/LeviBorodenko/motionblur/tree/master
    """
    def __init__(self, kernel_size, intensity) -> None:
        self.kernel_size = kernel_size
        self.intensity = intensity

    def __call__(self, tensor):
        kernel_matrix = Kernel(size=self.kernel_size, intensity=self.intensity).kernelMatrix
        kernel_tensor = torch.tensor(kernel_matrix, dtype=tensor.dtype)
        kernel_tensor = kernel_tensor.unsqueeze(0).unsqueeze(0)
        num_channels = tensor.size(0)
        kernel_tensor = kernel_tensor.repeat(num_channels, 1, 1, 1) 
        blurred = F.conv2d(tensor, weight=kernel_tensor, padding=self.kernel_size[0] // 2, groups=3)
        return blurred

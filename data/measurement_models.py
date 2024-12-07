# ====================================================================== #
# In this script we apply the "forward" measurement models as defined in 
# https://openreview.net/forum?id=OnD9zGAGT0k
# ====================================================================== #
import os
import torch
import torch.nn.functional as F
import yaml   
from data.blur_models.kernel_encoding.kernel_wizard import KernelWizard
from data.motionblur import Kernel
from guided_diffusion import dist_util

class noiser:
    """Could use this to make the code a bit cleaner"""
    def __init__(self, noise_model="gaussian", sigma: float = 0.05):
        self.noise_model = noise_model
        self.sigma = sigma
    
    def __call__(self, tensor):
        if self.noise_model == "gaussian":
            return tensor + torch.randn_like(tensor) * self.sigma
        elif self.noise_model == "poisson":
            return torch.poisson(tensor)
        else: 
            return tensor
        
# def noiser(tensor, noise_model="gaussian", sigma: float = 0.05):
#     if noise_model == "gaussian":
#         return tensor + torch.randn_like(tensor) * sigma
#     elif noise_model == "poisson":
#         return torch.poisson(tensor)
#     else: 
#         return tensor

class Identity(object):
    "Implements the identity function as forward measurement model"
    def __init__(self, noise_model="gaussian", sigma=.05):
        if (noise_model != "gaussian") and (noise_model != "poisson"):
            print(f"Noise model {noise_model} not implemented! Use 'gaussian' or 'poisson'.")
            return ValueError 
        self.noise_model = noise_model
        self.sigma = sigma

    def __call__(self, tensor):
        if self.noise_model == "gaussian":
            return tensor + torch.randn_like(tensor)*self.sigma
        elif self.noise_model == "poisson":
            return torch.poisson(tensor)
        else: 
            return tensor
  
    
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
    def __init__(self, noise_model="gaussian", sigma=.05, inpainting_noise_level=.92):
        self.mask = None
        self.sigma = sigma
        self.noise_level = inpainting_noise_level
        if noise_model not in ["gaussian", "poisson"]:
            raise ValueError(f"Noise model {noise_model} not implemented! Use 'gaussian' or 'poisson'.")
        self.noise_model = noise_model
        
    def __call__(self, tensor):
        device = tensor.device
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        b, c, h, w = tensor.shape
        if self.mask is None:
            mask = (torch.rand((b, 1, h, w), device=device) > self.noise_level)
            self.mask = mask.expand(-1, c, -1, -1)
        return tensor * self.mask.to(device)
    
    def forward_noise(self, tensor):
        tensor = tensor.squeeze(0) if len(tensor.shape) == 4 else tensor
        device = tensor.device
        tensor = self(tensor)
        tensor = self.noiser(tensor, device)
        if (len(tensor.shape) == 4) and (tensor.shape[0] == 1):
            tensor = tensor.squeeze(0)
        return tensor
        
    def noiser(self, tensor, device):
        if self.noise_model == "gaussian":
            noise = torch.randn(tensor.shape, device=device) * self.sigma
            result = tensor + noise
        else:
            masked = tensor
            result = torch.poisson(masked)
        return result
    
    def __repr__(self):
        return self.__class__.__name__


class BoxInpainting(object):
    """ 
    Implements the box inpainting forward measurement model
    - y ~ N(y|Px, sigma**2 * I) if noise_model = "gaussian"
    - y ~ Poisson(Px; lamb) if noise_model = "poisson"
    """
    def __init__(self, noise_model="gaussian", sigma=1.):
        self.sigma = sigma
        self.noise_model = noise_model
        self.x1 = None
        self.x2 = None
        self.box_h = None
        self.box_w = None
        self.box_values = False
        if (noise_model != "gaussian") and (noise_model != "poisson"):
            print(f"Noise model {noise_model} not implemented! Use 'gaussian' or 'poisson' ")
            return ValueError
        

    def box(self, x):
        """Generate random coordinates for a 128x128 box that fits within the image"""
        # Only generate box values the first time. Needs to have consistent
        # forward operation for all steps. 
        if self.box_values: 
            return
        
        _, _, h, w = x.shape

        max_x = h - 128 if h >= 128 else 0
        max_y = w - 128 if w >= 128 else 0
        
        self.x1 = torch.randint(0, max(1, max_x), (1,)).item()
        self.x2 = torch.randint(0, max(1, max_y), (1,)).item()
        
        self.box_h = min(128, h)
        self.box_w = min(128, w)

        self.box_values = True
        
        return

    def __call__(self, tensor):
        device = tensor.device
            
        # Handle batch dimension consistently
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
            
        b, c, h, w = tensor.shape

        x = tensor
        self.box(x)

        # Generate mask on the correct device
        mask = (torch.zeros((b, 1, 128, 128), device=device) > 0.5)
        mask = mask.expand(-1, c, -1, -1)
        
        x[:, :, self.x1:self.x1 + self.box_h, self.x2:self.x2 + self.box_w] \
            = x[:, :, self.x1:self.x1 + self.box_h, self.x2:self.x2 + self.box_w] \
            * mask

        return x.squeeze(0) if (len(x.shape) == 4) and (x.shape[0] == 1) else x
 
    def forward_noise(self, tensor):
        # Handle batch dimension consistently
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        tensor = self(tensor)
        tensor = self.noiser(tensor)
        return tensor.squeeze(0) if (len(tensor.shape) == 4) and (tensor.shape[0] == 1) else tensor

    def noiser(self, tensor):
        if self.noise_model == "gaussian":
            return tensor + torch.randn(size=tensor.size()) * self.sigma
        elif self.noise_model == "poisson":
            return torch.poisson(tensor) 
        else: 
            return None
    
    def __repr__(self):
        return self.__class__.__name__

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

        # Handle batch dimension consistently
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)

        if tensor.get_device() == 0 and dist_util.dev() == torch.device("mps"):
            tensor = tensor.to("cpu")
            downsampled_image = self.bicubic_downsample(tensor)
            upsampled_image   = self.bicubic_upsample(downsampled_image)
            upsampled_image = upsampled_image.to("mps")
        
        else: 
            downsampled_image = self.bicubic_downsample(tensor)
            upsampled_image   = self.bicubic_upsample(downsampled_image)

        # print("noisy_downsample from call before return: ", noisy_downsample.get_device())
        return upsampled_image.squeeze(0) if (len(upsampled_image.shape) == 4) and (upsampled_image.shape[0] == 1) else upsampled_image
    
    def bicubic_downsample(self, image):
        downsampled_image = torch.nn.functional.interpolate(image, scale_factor=self.downscale_factor, mode='bicubic', align_corners=True)            
        downsampled_image = downsampled_image.squeeze(0)
        # print("image shape: ", downsampled_image.shape)
        return downsampled_image
    
    def bicubic_upsample(self, image):
        image = image.unsqueeze(0)
        upsampled_image = torch.nn.functional.interpolate(image, scale_factor=self.upscale_factor, mode='bicubic', align_corners=True)
        upsampled_image = upsampled_image.squeeze(0)
        return upsampled_image

    def forward_noise(self, tensor):
        # Handle batch dimension consistently
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        tensor = self(tensor)
        tensor = self.noiser(tensor)
        return tensor
    
    def noiser(self, tensor):
        if self.noise_model == "gaussian":
            return tensor + torch.randn(size=tensor.size()) * self.sigma
        elif self.noise_model == "poisson":
            return torch.poisson(tensor) 
        else: 
            return None
    
    """
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
    """

    def __repr__(self):
        return self.__class__.__name__
        
class NonLinearBlurring(object):
    """
    Implements the non-linear blurring forward measurement model.
    y ~ N(y| F(x,k), sigma**2 * I) if self.noise_model = "gaussian"
    y ~ Poisson(F(x,k)) if self.noise_model = "poisson"
    F(x,k) is a external pretrained model from (see link)
        link: https://github.com/VinAIResearch/blur-kernel-space-exploring  
    """
    def __init__(self, noise_model="gaussian", sigma=.05):
        if noise_model not in ["gaussian", "poisson"]:
            raise ValueError(f"Noise model {noise_model} not implemented! Use 'gaussian' or 'poisson'.")
        self.sigma = sigma

    def generate_blur(self, tensor):
        # Handle batch dimension consistently
        if len(tensor.shape) == 3:
            x = tensor.unsqueeze(0)

        # NOTE: From https://github.com/VinAIResearch/blur-kernel-space-exploring/blob/main/generate_blur.py 
        device = tensor.device
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yml_path = os.path.join(current_dir, "blur_models", "default.yml") 
        #print(yml_path)
        #device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

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

        return LQ_tensor.squeeze(0) if (len(LQ_tensor.shape) == 4) and (LQ_tensor.shape[0] == 1) else LQ_tensor

    def __call__(self, tensor):
        blurred_img = self.generate_blur(tensor)
        return blurred_img
 
    def forward_noise(self, tensor):
        tensor = self(tensor)
        tensor = self.noiser(tensor)
        return tensor

    def noiser(self, tensor):
        if self.noise_model == "gaussian":
            return tensor + torch.randn(size=tensor.size()) * self.sigma
        elif self.noise_model == "poisson":
            return torch.poisson(tensor) 
        else: 
            return None
    
    def __repr__(self):
        return self.__class__.__name__

class GaussianBlur(object):
    """
    Implements the Gaussian convolution (Gaussian noise) forward measurement model.
    The Gaussian kernel is 61x61 and convolved with the ground truth image to produce 
    the measurement. 
    """
    def __init__(self, noise_model='gaussian', kernel_size=(61,61), sigma_in_conv=3, sigma=.05):
        if noise_model not in ["gaussian", "poisson"]:
            raise ValueError(f"Noise model {noise_model} not implemented! Use 'gaussian' or 'poisson'.")
        self.noise_model = noise_model
        self.kernel_size = kernel_size
        self.sigma_in_conv = sigma_in_conv
        self.sigma = sigma

    def gaussian_kernel(self, device=None):
        size = self.kernel_size[0]
        x = torch.linspace(-(size // 2), size // 2, size, device=device)
        y = torch.linspace(-(size // 2), size // 2, size, device=device)
        x, y = torch.meshgrid(x, y, indexing='xy')
        kernel = torch.exp(-(x**2 + y**2) / (2 * self.sigma_in_conv**2))
        kernel /= kernel.sum()
        return kernel

    def __call__(self, tensor):
        # Handle batch dimension consistently
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        device = tensor.device  
        kernel = self.gaussian_kernel(device=device)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        num_channels = tensor.size(1)
        kernel = kernel.repeat(num_channels, 1, 1, 1)
        blurred = F.conv2d(tensor, weight=kernel, padding=self.kernel_size[0] // 2, groups=num_channels)
        return blurred.squeeze(0) if (len(blurred.shape) == 4) and (blurred.shape[0] == 1) else blurred
    
    def forward_noise(self, tensor):
        tensor = self(tensor)
        tensor = self.noiser(tensor)
        return tensor

    def noiser(self, tensor):
        if self.noise_model == "gaussian":
            return tensor + torch.randn(size=tensor.size()) * self.sigma
        elif self.noise_model == "poisson":
            return torch.poisson(tensor) 
        else: 
            return None
    
    def __repr__(self):
        return self.__class__.__name__
    
class MotionBlur(object):
    """
    Implements the motion blur forward measurement model. 
    The motion blur kernel is an external kernel from (see link)
        link: https://github.com/LeviBorodenko/motionblur/tree/master
    """
    def __init__(self, noise_model="gaussian", kernel_size=(61,61), intensity=0.5, sigma=.05) -> None:
        if noise_model not in ["gaussian", "poisson"]:
            raise ValueError(f"Noise model {noise_model} not implemented! Use 'gaussian' or 'poisson'.")
        self.noise_model = noise_model
        self.kernel_size = kernel_size
        self.intensity = intensity
        self.sigma = sigma
        self.kernel_tensor = None

    def __call__(self, tensor):
        # Handle batch dimension consistently
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)

        if self.kernel_tensor is None:
            kernel_matrix = Kernel(size=self.kernel_size, intensity=self.intensity).kernelMatrix
            kernel_tensor = torch.tensor(kernel_matrix, dtype=tensor.dtype, device=tensor.device)
            kernel_tensor = kernel_tensor.unsqueeze(0).unsqueeze(0)
            num_channels = tensor.size(1)
            kernel_tensor = kernel_tensor.repeat(num_channels, 1, 1, 1)
            self.kernel_tensor = kernel_tensor

        self.kernel_tensor = self.kernel_tensor.to(tensor.device)
        num_channels = tensor.size(1)
        blurred = F.conv2d(tensor, weight=self.kernel_tensor, padding=self.kernel_size[0] // 2, groups=num_channels)

        # Return with original dimensions
        return blurred.squeeze(0) if len(tensor.shape) == 4 and tensor.shape[0] == 1 else blurred
    
    def forward_noise(self, tensor):
        tensor = self(tensor)
        tensor = self.noiser(tensor)
        return tensor
    
    def noiser(self, tensor):
        if self.noise_model == "gaussian":
            return tensor + torch.randn(size=tensor.size()) * self.sigma
        elif self.noise_model == "poisson":
            return torch.poisson(tensor) 
        else: 
            return None

    def __repr__(self):
        return self.__class__.__name__

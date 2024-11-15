# ====================================================================== #
# In this script we apply the "forward" measurement models as defined in 
# https://openreview.net/forum?id=OnD9zGAGT0k
# ====================================================================== #
import torch
import argparse
import torch
import yaml   
from blur_models.kernel_encoding.kernel_wizard import KernelWizard

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
            return x + torch.randn(size=x.size())*self.sigma
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
        if (noise_model != "gaussian") and (noise_model != "poisson"):
            print(f"Noise model {self.noise_model} not implemented! Use 'gaussian' or 'poisson' ")
            return ValueError 
        self.noise_model = noise_model

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
            x[:, x1:x1 + box_h, x2:x2 + box_w] = torch.randn((3, box_h, box_w)) * self.sigma
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


class NonLinearBlurring(object):

    def __init__(self, noise_model="gaussian", sigma=1.):
        self.sigma = sigma
        if (noise_model != "gaussian") and (noise_model != "poisson"):
            print(f"Noise model {noise_model} not implemented! Use 'gaussian' or 'poisson'.")
            return ValueError 
        self.noise_model = noise_model

    def generate_blur(self, tensor):
        # NOTE: From https://github.com/VinAIResearch/blur-kernel-space-exploring/blob/main/generate_blur.py  
        #device = torch.device("cuda")
        #parser = argparse.ArgumentParser(description="Kernel extractor testing")
        #parser.add_argument("--yml_path", action="store", help="yml path", type=str, required=True)
        #args = parser.parse_args()
        #yml_path = args.yml_path

        # NOTE: Might want to have yml path as input instead.
        # NOTE: default.yml set pretrained: pretrained/GOPRO_wVAE.pth
        #       pretrained only have kernel.pth (does this work?) 
        yml_path = "default.yml"

        # Initializing mode
        with open(yml_path, "r") as f:
            opt = yaml.load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        model = KernelWizard(opt)
        model.eval()
        model.load_state_dict(torch.load(model_path))
        #model = model.to(device)
        model = model.cuda()
        
        with torch.no_grad():
            # NOTE: kernel size (512, 2, 2)? 
            kernel = torch.randn((1, 512, 2, 2)).cuda() * 1.2
            LQ_tensor = model.adaptKernel(tensor, kernel)

        return LQ_tensor

    def __call__(self, tensor):
        blurred_img = self.generate_blur(tensor)
        x = blurred_img

        if self.noise_model == "gaussian":
            return x + torch.randn(size=x.size())*self.sigma
        elif self.noise_model == "poisson":
            return torch.poisson(x) 
        else: 
            return None
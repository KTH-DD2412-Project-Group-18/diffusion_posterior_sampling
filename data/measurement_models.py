# ====================================================================== #
# In this script we apply the "forward" measurement models as defined in 
# https://openreview.net/forum?id=OnD9zGAGT0k
# ====================================================================== #
import torch

class Inpainting(object):
    """ 
    Implements the inpainting forward measurement model
    y ~ N(y|Px, sigma**2 * I) if noise_model = "gaussian"
    y ~ Poisson(Px; lamb) if noise_model = "poisson"
    """
    def __init__(self, noise_model="gaussian", sigma=1.):
        self.sigma = sigma
        if noise_model != "gaussian" or noise_model != "poisson":
            print(f"Noise model {self.noise_model} not implemented! Use 'gaussian' or 'poisson' ")
            return ValueError 
        self.noise_model = noise_model

    def __call__(self, tensor):
        self.P = torch.randint(0,1,size=(0,1))
        meas = self.P * tensor
        if self.noise_model == "gaussian":
            return meas + torch.randn(size=meas.size())*self.std
        elif self.noise_model == "poisson":
            return torch.poisson(meas) 
        else: 
            return None
    
    def __repr__(self):
        return self.__class__.__name__ + f"(mean={0}, std={1})"

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
        self.sigma = sigma
        self.phi = torch.convolution() ###  <- think about this 
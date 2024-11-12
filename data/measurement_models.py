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
        batch_size, n, d = tensor.shape
        mask = torch.bernoulli(torch.ones_like(tensor)*0.08)
        x = tensor * mask
        if self.noise_model == "gaussian":
            return x + torch.randn(size=x.size())*self.std
        elif self.noise_model == "poisson":
            return torch.poisson(x) 
        else: 
            return None
    
    def __repr__(self):
        return self.__class__.__name__ + f"(mean={0}, std={1})"
    
class BoxInpainting(object):
    """ 
    Implements the box inpainting forward measurement model
    y ~ N(y|Px, sigma**2 * I) if noise_model = "gaussian"
    y ~ Poisson(Px; lamb) if noise_model = "poisson"
    TODO: Same box is generated for all images in the batch. 
          Change or keep it as it is. 
    """
    def __init__(self, noise_model="gaussian", sigma=1.):
        self.sigma = sigma
        if noise_model != "gaussian" and noise_model != "poisson":
            print(f"Noise model {self.noise_model} not implemented! Use 'gaussian' or 'poisson' ")
            return ValueError 
        self.noise_model = noise_model

    def __call__(self, tensor, batch_size):
        batch_size, channels, n, d = tensor.shape
        x = tensor
        coordinates = self.box(x, batch_size)
        x1, x2 = coordinates[0], coordinates[1]
        if self.noise_model == "gaussian":
            x[:, x1:x1 + 128, x2:x2 + 128] = torch.randn((batch_size, channels, 128, 128))*self.sigma
            return x
        elif self.noise_model == "poisson":
            x[:, :, x1:x1+128, x2:x2+128] = torch.poisson(x[:, :, x1:x1+128, x2:x2+128])
            return x  
        else: 
            return None
    
    def __repr__(self):
        return self.__class__.__name__ + f"(mean={0}, std={1})"
    
    def box(self, tensor, batchsize):
        """
        Sample a 128x128 box uniformly within 16 pixel margin.
        The sample consists of coordinates for the top left corner. 
        """
        n, d = tensor.shape[1], tensor.shape[2] # Might be wrong shapes (keep in mind)
        assert(n == d), "image is not square"
        box_interval_x = [16, n - 16 - 128]
        box_interval_y = [16, d - 16 - 128]
        p_x = torch.randint(box_interval_x[0], box_interval_x[1] + 1, (1,))
        p_y = torch.randint(box_interval_y[0], box_interval_y[1] + 1, (1,))
        
        return p_x.item(), p_y.item()


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



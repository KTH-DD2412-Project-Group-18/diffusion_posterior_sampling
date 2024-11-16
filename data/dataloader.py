import torch
from torchvision import datasets
from torchvision import transforms
from measurement_models import RandomInpainting, BoxInpainting, NonLinearBlurring, GaussianBlur, MotionBlur
import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    #measurement_model = RandomInpainting(noise_model="gaussian", sigma=0.05)
    #measurement_model = BoxInpainting(noise_model="gaussian", sigma=0.05)
    #measurement_model = NonLinearBlurring(noise_model="gaussian", sigma=0.05)
    #measurement_model, model = GaussianBlur(kernel_size=(61,61), sigma=3.0), 'Gaussian'
    measurement_model = MotionBlur((61, 61), 0.5)

    # -- 
    # We create an ImageFolder with our transformation according to our measurement_model
    # NOTE: I run from root so in absolute term the path is ../datasets/imagenet/val
    # NOTE: in imagenet/val there are a bunch of class-folders containing .JPEG files, this is what `ImageFolder`` wants!
    # -- 

    val_data = datasets.ImageFolder("./datasets/imagenet", 
                    transform= transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.ToTensor(),    
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        measurement_model
                    ])
                    ) 

    # -- 
    # Torch DataLoader handles the batch dimension
    # --
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=10,
        shuffle=False
    )

    # -- 
    # some visualization
    # --
    
    imgs, labels = next(iter(val_loader))
    print(imgs.shape, labels)

    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean

    img = denormalize(imgs[8])
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
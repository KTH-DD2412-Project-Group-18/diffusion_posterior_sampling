import torch
from torchvision import (datasets, transforms)
from measurement_models import RandomInpainting, BoxInpainting
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

if __name__ == "__main__":

    measurement_model = BoxInpainting(noise_model="gaussian", sigma=1.)

    # -- 
    # We create an ImageFolder with our transformation according to our measurement_model
    # NOTE: I run from root so in absolute term the path is ../datasets/imagenet/val
    # NOTE: in imagenet/val there are a bunch of class-folders containing .JPEG files, this is what `ImageFolder`` wants!
    # -- 
    val_data = datasets.ImageFolder("./datasets/imagenet/val", 
                      transform= transforms.Compose([
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

    img = denormalize(imgs[9])
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    plt.axis('off')
    fig.savefig("./fishman.png")

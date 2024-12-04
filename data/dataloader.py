import torch
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToPILImage

from measurement_models import RandomInpainting, BoxInpainting, NonLinearBlurring, GaussianBlur, MotionBlur, SuperResolution
import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    # measurement_model = BoxInpainting(noise_model="gaussian", sigma=1.)
    # measurement_model = SuperResolution(downscale_factor=0.25, upscale_factor=4, noise_model="gaussian", sigma=0.05)
    # measurement_model = RandomInpainting(noise_model="gaussian", sigma=0.05)
    # measurement_model = BoxInpainting(noise_model="gaussian", sigma=0.05)
    measurement_model = NonLinearBlurring(noise_model="gaussian", sigma=0.05)
    # measurement_model, model = GaussianBlur(kernel_size=(61,61), sigma=3.0), 'Gaussian'
    # measurement_model = MotionBlur((61, 61), 0.5)

    # -- 
    # We create an ImageFolder with our transformation according to our measurement_model
    # NOTE: I run from root so in absolute term the path is ../datasets/imagenet/val
    # NOTE: in imagenet/val there are a bunch of class-folders containing .JPEG files, this is what `ImageFolder`` wants!
    # -- 

    val_data = datasets.ImageFolder("./datasets/imagenet/val2", 
                      transform= transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                          measurement_model # Change model object here to change measurement type
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

    for batch_idx, data in enumerate(val_loader):
        print(f"Batch {batch_idx + 1}:")
        if isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                print(f" - Tensor {i}: Shape {item.shape}")
        else:
            print(f" - Data shape: {data.shape}")
        break  # Stop after the first batch for inspection

    # -- 
    # some visualization
    # --
    imgs, labels = next(iter(val_loader))
    print(imgs.shape, labels)

    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean

    ######################### Image after call
    print("imgs[9].shape: ", imgs[9].shape)
    img_original = denormalize(imgs[9])
    print("img shape: ", img_original.shape)
    image_np = img_original.permute(1, 2, 0).numpy()
    image_np = np.clip(image_np, 0, 1)
    print("img shape after permute and clip: ", image_np.shape)
    to_pil = ToPILImage()
    image = to_pil(image_np)
    image.save("image_from_call.png")  

    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()
    
    

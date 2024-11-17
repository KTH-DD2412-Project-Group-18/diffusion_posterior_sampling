import torch
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToPILImage

from measurement_models import RandomInpainting, BoxInpainting, SuperResolution
import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    measurement_model = BoxInpainting(noise_model="gaussian", sigma=1.)
    super_res_model = SuperResolution(downscale_factor=0.25, upscale_factor=4, noise_model="gaussian", sigma=0.05)

    # -- 
    # We create an ImageFolder with our transformation according to our measurement_model
    # NOTE: I run from root so in absolute term the path is ../datasets/imagenet/val
    # NOTE: in imagenet/val there are a bunch of class-folders containing .JPEG files, this is what `ImageFolder`` wants!
    # -- 
    val_data = datasets.ImageFolder("./datasets/imagenet/val", 
                      transform= transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                          super_res_model
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

    ######################### Original image
    print("imgs[9].shape: ", imgs[9].shape)
    img_original = denormalize(imgs[9])
    print("img shape: ", img_original.shape)
    to_pil = ToPILImage()
    image = to_pil(img_original)
    image.save("original_image.png")  


    ######################### Bicubic downsample image
    img_bicubic_downsampled = super_res_model.bicubic_downsample(img_original)
    img_bicubic_downsampled_np = img_bicubic_downsampled.permute(1, 2, 0).numpy()
    img_bicubic_downsampled_np = np.clip(img_bicubic_downsampled_np, 0, 1)

    print("img shape: ", img_bicubic_downsampled_np.shape)
    to_pil = ToPILImage()
    image = to_pil(img_bicubic_downsampled_np)
    image.save("downsampled_image.png") 

    plt.figure(figsize=(10, 10))
    plt.imshow(img_bicubic_downsampled_np)
    plt.axis('off')
    plt.show()

    ######################### Bicubic downsample + expand with noise

    print("Expanding image with noise")
    print("img_bicubic_downsampled shape: ", img_bicubic_downsampled.shape)
    img_expanded_with_noise, noise_mask = super_res_model.upsample_with_noise(img_bicubic_downsampled)
    print("img_expanded shape: ", img_expanded_with_noise.shape)
    img_expanded_with_noise = img_expanded_with_noise.permute(1, 2, 0).numpy()
    img_expanded_with_noise = np.clip(img_expanded_with_noise, 0, 1)

    print("Image after expanding with noise")
    plt.figure(figsize=(10, 10))
    plt.imshow(img_expanded_with_noise)
    plt.axis('off')
    plt.show()

    to_pil = ToPILImage()
    image = to_pil(img_expanded_with_noise)
    image.save("image_expanded_noise.png") 

    ######################### Bicubic downsample + bicubic upsample

    print("upsampling image with no noise")
    print("img_bicubic_downsample shape: ", img_bicubic_downsampled.shape)
    img_bicubic_upscaled_no_noise = super_res_model.bicubic_upsample(img_bicubic_downsampled)
    print("img_bicubic_upscaled shape: ", img_bicubic_upscaled_no_noise.shape)
    img_bicubic_upscaled_no_noise_np = img_bicubic_upscaled_no_noise.permute(1, 2, 0).numpy()
    img_bicubic_upscaled_no_noise_np = np.clip(img_bicubic_upscaled_no_noise_np, 0, 1)

    print("Image after upscaling without noise")
    plt.figure(figsize=(10, 10))
    plt.imshow(img_bicubic_upscaled_no_noise_np)
    plt.axis('off')
    plt.show()

    to_pil = ToPILImage()
    image = to_pil(img_bicubic_upscaled_no_noise_np)
    image.save("image_upscaled_no_noise.png") 

    ######################### Bicubic downsampling + gauss noise

    print(" downsample image with gaussian noise, sigma = 0.05")
    print("img shape: ", img_bicubic_downsampled.shape)
    measurement_img = super_res_model.add_gaussian_noise(img_bicubic_downsampled, 64)
    print("measurement_img shape: ", measurement_img.shape)
    measurement_img = measurement_img.permute(1, 2, 0).numpy()
    measurement_img = np.clip(measurement_img, 0, 1)

    print("Measurement image with gaussian noise")
    plt.figure(figsize=(10, 10))
    plt.imshow(measurement_img)
    plt.axis('off')
    plt.show()

    to_pil = ToPILImage()
    image = to_pil(measurement_img)
    image.save("measurement_img.png")  

    #########################  Bicubic downsample + gauss noise + bicubic upsample

    print(" upsampled measurement image")
    print("measurement_img shape: ", measurement_img.shape)
    measurement_img = super_res_model.add_gaussian_noise(img_bicubic_downsampled, 64)
    measurement_img_upscaled = super_res_model.bicubic_upsample(measurement_img)
    print("measurement_img_upsclaed shape: ", measurement_img_upscaled.shape)
    measurement_img_upscaled_np = measurement_img_upscaled.permute(1, 2, 0).numpy()
    measurement_img_upscaled_np = np.clip(measurement_img_upscaled_np, 0, 1)

    print("Upscaled measurement image with gaussian noise")
    plt.figure(figsize=(10, 10))
    plt.imshow(measurement_img_upscaled_np)
    plt.axis('off')
    plt.show()

    to_pil = ToPILImage()
    image = to_pil(measurement_img_upscaled_np)
    image.save("measurement_img_upscaled.png")  # Change extension for JPEG, e.g., "output_image.jpg"

    ######################### Input to reverse processs???????

    img_reverse_input = img_original - img_bicubic_upscaled_no_noise + measurement_img_upscaled

    img_reverse_input_np = img_reverse_input.permute(1, 2, 0).numpy()
    img_reverse_input_np = np.clip(img_reverse_input_np, 0, 1)

    print("Image reverse input ")
    plt.figure(figsize=(10, 10))
    plt.imshow(img_reverse_input_np)
    plt.axis('off')
    plt.show()

    to_pil = ToPILImage()
    image = to_pil(img_reverse_input_np)
    image.save("image_input_reverser.png")  # Change extension for JPEG, e.g., "output_image.jpg"

import torch
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToPILImage

from measurement_models import RandomInpainting, BoxInpainting, NonLinearBlurring, GaussianBlur, MotionBlur, SuperResolution
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

if __name__ == "__main__":

    # measurement_model = BoxInpainting(noise_model="gaussian", sigma=1.)
    # measurement_model = SuperResolution(downscale_factor=0.25, upscale_factor=4, noise_model="gaussian", sigma=0.05)
    # measurement_model = RandomInpainting(noise_model="gaussian", sigma=0.05)
    # measurement_model = BoxInpainting(noise_model="gaussian", sigma=0.05)
    # measurement_model = NonLinearBlurring(noise_model="gaussian", sigma=0.05)
    # measurement_model, model = GaussianBlur(kernel_size=(61,61), sigma=3.0), 'Gaussian'
    # measurement_model = MotionBlur((61, 61), 0.5)
    measurement_model = None

    # -- 
    # We create an ImageFolder with our transformation according to our measurement_model
    # NOTE: I run from root so in absolute term the path is ../datasets/imagenet/val
    # NOTE: in imagenet/val there are a bunch of class-folders containing .JPEG files, this is what `ImageFolder`` wants!
    # -- 

    if measurement_model == None:
        val_data = datasets.ImageFolder("./datasets/imagenet/val", 
                      transform= transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                      )
    else:
        val_data = datasets.ImageFolder("./datasets/imagenet/val", 
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
    print("imgs[9].shape: ", imgs[9].shape)

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

    print_magnitude = False

    if print_magnitude == False:
        # Print fourier transform
        print("gray")
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        
        print("compute dfft")
        # Compute the discrete Fourier Transform of the image
        fourier = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        print("fourier shape: ", fourier.shape)
        
        # Shift the zero-frequency component to the center of the spectrum
        fourier_shift = np.fft.fftshift(fourier)
        
        # calculate the magnitude of the Fourier Transform
        image_size = fourier_shift[0].shape[0]
        magnitude = 20*np.log(cv2.magnitude(fourier_shift[:,:,0],fourier_shift[:,:,1]))
        
        # Scale the magnitude for display
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        print("fourier magnitude: ", magnitude)
        print("magnitude shape: ", magnitude.shape)
        
        # Display the magnitude of the Fourier Transform
        cv2.imshow('Fourier Transform', magnitude)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:

        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        # calculating the discrete Fourier transform
        DFT = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        
        # reposition the zero-frequency component to the spectrum's middle
        shift = np.fft.fftshift(DFT)
        row, col = gray.shape
        center_row, center_col = row // 2, col // 2
        
        # create a mask with a centered square of 1s
        mask = np.zeros((row, col, 2), np.uint8)
        mask[center_row - 30:center_row + 30, center_col - 30:center_col + 30] = 1
        
        # put the mask and inverse DFT in place.
        fft_shift = shift * mask
        fft_ifft_shift = np.fft.ifftshift(fft_shift)
        imageThen = cv2.idft(fft_ifft_shift)
        
        # calculate the magnitude of the inverse DFT
        imageThen = cv2.magnitude(imageThen[:,:,0], imageThen[:,:,1])
        
        # visualize the original image and the magnitude spectrum
        plt.figure(figsize=(10,10))
        plt.subplot(121), plt.imshow(gray, cmap='gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(imageThen, cmap='gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()



    # to_pil = ToPILImage()
    # image = to_pil(image_np)
    # image.save("image_from_call.png")  

    # plt.figure(figsize=(10, 10))
    # plt.imshow(image_np)
    # plt.axis('off')
    # plt.show()
    
    


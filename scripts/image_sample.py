"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
Source: openAI

Modified by Alexander Gutell, Dan Vicente and Ludvig Skare in 2024
"""

import os
import time
from datetime import datetime
import argparse
import numpy as np
import torch as th
import torch.distributed as dist
from torchvision.transforms import ToPILImage

from torchvision import (datasets, 
                         transforms)
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    get_measurement_model,
    create_dps_diffusion
)
from PIL import Image
import matplotlib.pyplot as plt
from data.data_utils import SingleImageDataset

def denormalize_imagenet(tensor):
                device = tensor.device
                mean = th.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
                std = th.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
                return tensor * std + mean

def main():
    args = create_argparser().parse_args()
    rank = 0 # or rank = dist.get_rank()

    # dist_util.setup_dist()

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    dev = dist_util.dev()
    print(f"Using device = '{dev}'")

    model.to(dev)
    model.eval()

    t_start = datetime.now()

    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        if args.dps_update:
            # =========================================== #
            # Perform the DPS sampling as in algorithm 1/2
            # =========================================== #
            measurement_model = get_measurement_model(
                 name=args.measurement_model,
                 noise_model=args.noise_model,
                 sigma=args.sigma
            )

            logger.configure(dir=f"./output/{measurement_model}/{time.time()}")
            logger.log(f"Preparing dataset with {measurement_model} and creating dps sampler...")
            
            # Compute the forward measurement + noise
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            if args.single_image_data:
                dataset = SingleImageDataset(args.data_path, transform=transform)
            else:
                dataset = datasets.ImageFolder(args.data_path, transform=transform)

            dataloader = th.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size, 
                shuffle=False
            )

            imgs_clean, _ = next(iter(dataloader))
            img_clean = imgs_clean[-1].requires_grad_(False)
            img_noisy = measurement_model.forward_noise(img_clean.clone())

            # Save clean + noisy images for reference
            for prefix, img in [("clean", img_clean), ("noisy", img_noisy)]:
                if len(img.shape) == 4:
                    img_ = img.squeeze(0)
                else:
                    img_ = img
                img_ = denormalize_imagenet(img_)
                img_ = img_.permute(1,2,0).numpy()
                img_ = np.clip(img_, 0, 1)
                save_path = os.path.join(logger.get_dir(), f"{prefix}_meas.png")
                plt.imsave(save_path, img_)
                del img_

            # Create the DPS model with all necessary parameters
            dps_diffusion = create_dps_diffusion(
                measurement_model=measurement_model,
                measurement=img_noisy.to(dev),
                noise_model=args.noise_model,
                step_size=args.step_size,
                image_size=args.image_size,
                learn_sigma=args.learn_sigma,
                diffusion_steps=args.diffusion_steps,
                noise_schedule=args.noise_schedule,
                timestep_respacing=args.timestep_respacing,
                use_kl=args.use_kl,
                predict_xstart=args.predict_xstart,
                rescale_timesteps=args.rescale_timesteps,
                rescale_learned_sigmas=args.rescale_learned_sigmas
            )
            sample_fn = dps_diffusion.p_sample_loop
            logger.log(f"created DPS-diffusion model with step_size = {args.step_size}..")

            # denoise sampling using first image in batch (should be the same every time if we have shuffle=False)
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
            )

        else:
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size)
            )

        sample = (sample + 1) / 2

        #mean = th.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(sample.device)
        #std = th.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(sample.device)
        #sample = (sample - mean) / std

        sample = denormalize_imagenet(sample)
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        # if dist.get_world_size() > 1: ###
        if rank > 1:
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist_util.safe_all_gather(gathered_samples, sample)
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        else:
            all_images.append(sample.cpu().numpy())

        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    
    print("\nFinal array stats:")
    print(f"Shape: {arr.shape}")
    print(f"Range: [{arr.min()}, {arr.max()}]")
    print(f"Mean: {arr.mean():.3f}")
    
    if rank == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)
        try:
            if arr.shape[0] > 0:
                img = Image.fromarray(arr[0])
                img_path = os.path.join(logger.get_dir(), "sample_0.png")
                img.save(img_path)
                logger.log(f"Saved sample image to {img_path}")
        except Exception as e:
            logger.log(f"Failed to save sample image: {e}")

    # dist.barrier()
    logger.log("sampling complete")
    t_end = datetime.now()
    t_execution = (t_end - t_start)
    print(f"Elapsed time during sampling = {t_execution}")

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="./models/256x256_diffusion_uncond.pt",
        dps_update=True,
        measurement_model="BoxInpainting",
        noise_model="gaussian",
        sigma=.05,
        step_size=1.,
        data_path="./datasets/imagenet/val2", # Default = ImageNet validation
        sampling_batch_size=10,
        single_image_data=True
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
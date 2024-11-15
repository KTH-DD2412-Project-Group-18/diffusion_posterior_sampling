"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
Source: openAI
Modified by Dan Vicente in November 2024
"""

import os
from datetime import datetime

# Set CPU thread settings at (to limit CPU usage locally) 
# os.environ["OMP_NUM_THREADS"] = "2"  # OpenMP threads
# os.environ["MKL_NUM_THREADS"] = "2"  # MKL threads
# os.environ["NUMEXPR_NUM_THREADS"] = "2"  # NumExpr threads
# os.environ["VECLIB_MAXIMUM_THREADS"] = "2"  # Vector library threads (specific to Mac)

import argparse
import os
import sys
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

##  for images:
##  --meas_model "super-resolution", "inpainting", "linear-deblurring", "nonlinear-deblurring", "phase-retrieval"
## for audio: ??
##

def main():
    args = create_argparser().parse_args()
    
    meas_model = "gaussian" # NOTE: This needs to be added into args_to_dict probably
    
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log(f"sampling using DPS with measurement_model == {meas_model}")
    t_start = datetime.now()
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        print("Raw sample stats:")
        print(f"Shape: {sample.shape}")
        print(f"Device: {sample.device}")
        print(f"Range: [{sample.min().item():.3f}, {sample.max().item():.3f}]")
        print(f"Mean: {sample.mean().item():.3f}")

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        print("\nNormalized sample stats:")
        print(f"Shape: {sample.shape}")
        print(f"Range: [{sample.min().item()}, {sample.max().item()}]")
        print(f"Mean: {sample.float().mean().item():.3f}")

        if dist.get_world_size() > 1:
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist_util.safe_all_gather(gathered_samples, sample)
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        else:
            all_images.append(sample.cpu().numpy())

        if args.class_cond:
            if dist.get_world_size() > 1:
                gathered_labels = [th.zeros_like(classes) for _ in range(2)]
                dist_util.safe_all_gather(gathered_labels, classes)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            else:
                all_labels.append(classes.cpu().numpy())

        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    
    print("\nFinal array stats:")
    print(f"Shape: {arr.shape}")
    print(f"Range: [{arr.min()}, {arr.max()}]")
    print(f"Mean: {arr.mean():.3f}")
    
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
        try:
            from PIL import Image
            if arr.shape[0] > 0:
                img = Image.fromarray(arr[0])
                img_path = os.path.join(logger.get_dir(), "sample_0.png")
                img.save(img_path)
                logger.log(f"Saved sample image to {img_path}")
        except Exception as e:
            logger.log(f"Failed to save sample image: {e}")
            
    dist.barrier()
    logger.log("sampling complete")
    t_end = datetime.now()
    t_execution = t_end - t_start
    print(f"Elapsed time during sampling = {t_execution}")

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="./models/512x512_diffusion.pt",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()

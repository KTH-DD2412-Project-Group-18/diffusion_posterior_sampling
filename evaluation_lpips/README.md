
# Note: This is an edited readme from the project page below. See their github for more information

## Perceptual Similarity Metric and Dataset [[Project Page]](http://richzhang.github.io/PerceptualSimilarity/)

### Quick start

Run `pip install lpips`. The following Python code is all you need.

```python
import lpips
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

import torch
img0 = torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
img1 = torch.zeros(1,3,64,64)
d = loss_fn_alex(img0, img1)
```

### Installation
- Install PyTorch 1.0+ and torchvision fom http://pytorch.org

```bash
pip install -r requirements.txt
```
- Clone this repo:
```bash
git clone https://github.com/richzhang/PerceptualSimilarity
cd PerceptualSimilarity
```

## (1) Learned Perceptual Image Patch Similarity (LPIPS) metric

Evaluate the distance between image patches. **Higher means further/more different. Lower means more similar.**

### (A) Basic Usage

#### (A.I) Line commands

Example scripts to take the distance between 2 specific images, all corresponding pairs of images in 2 directories, or all pairs of images within a directory:

```
python lpips_2imgs.py -p0 imgs/ex_ref.png -p1 imgs/ex_p0.png --use_gpu
python lpips_2dirs.py -d0 imgs/ex_dir0 -d1 imgs/ex_dir1 -o imgs/example_dists.txt --use_gpu
python lpips_1dir_allpairs.py -d imgs/ex_dir_pair -o imgs/example_dists_pair.txt --use_gpu
```

**Some Options** By default in `model.initialize`:
- By default, `net='alex'`. Network `alex` is fastest, performs the best (as a forward metric), and is the default. For backpropping, `net='vgg'` loss is closer to the traditional "perceptual loss".
- By default, `lpips=True`. This adds a linear calibration on top of intermediate features in the net. Set this to `lpips=False` to equally weight all the features.

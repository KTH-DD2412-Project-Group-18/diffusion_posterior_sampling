import numpy as np
import torch as th
from .gaussian_diffusion import (GaussianDiffusion, )
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
from torchvision.transforms import ToPILImage

def denormalize_imagenet(tensor):
                mean = th.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
                std = th.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
                return tensor * std + mean

def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(self, model, *args, **kwargs): 
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )
    
    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class PoissonMseLoss(th.nn.Module):
    """
    A Poisson-normal approximation adjusted MSE-loss that supports autograd.
    """
    def __init__(self): 
        super(PoissonMseLoss, self).__init__()
    
    def forward(self, x, y):
        """ forward Poisson normalized loss (take absolute value of y (for "correct" mathematics)"""
        assert x.shape == y.shape, f"Shape missmatch between operator with shape {x.shape} and observation with shape {y.shape}"
        weights = 1. / (2 * th.abs(y).clamp(min=1e-6))
        diff = (x - y)
        loss = (weights * diff * diff).sum()
        return loss


class DiffusionPosteriorSampling(SpacedDiffusion):
    """
    A diffusion process that does the additional DPS sampling step as outlined in Chung et. al 
    
    Parameters
    ----------
    - use_timesteps: a collection (sequence or set) of timesteps from the original diffusion process to retain.
    - measurement_model: what measurement operator to use in posterior sampling,  "
    - noise_model: What additive noise model to use, "gaussian" or "poisson"
    - lr: factor to use in posterior sampling, zeta_i = step_size / ||y - A(x(x_0))||
    """
    def __init__(
            self,
            use_timesteps,
            measurement_model,
            measurement,
            noise_model, 
            step_size=1., 
            **kwargs
    ):
        super().__init__(use_timesteps, **kwargs)
        self.measurement_model = measurement_model
        if th.backends.mps.is_available():
            self.measurement = measurement.to("mps") # y = A(x) + n, where x is the clean sample
        elif th.cuda.is_available():
            self.measurement = measurement.to("cuda")
        else: 
            self.measurement = measurement
        self.noise_model = noise_model
        self.step_size = step_size
        if self.noise_model == "gaussian":
            self.measurement_loss = th.nn.MSELoss(reduction="sum")
        elif self.noise_model == "poisson":
            self.measurement_loss = PoissonMseLoss()
        else:
            print("only 'gaussian' and 'poisson' noise models!")
            return NotImplementedError

    def dps_update(self, 
                   x: th.Tensor, 
                   t: int, 
                   x0: th.Tensor, 
                   x_old: th.Tensor
    ) -> th.Tensor:
        """
        Computes the DPS-sampling step, a gradient update at
        We recompute eps and E[x0|x] to be able to track gradients
        Also, we detach the computational graph from the previous iteration since its not needed in backprorp
        """
        measurement = self.measurement
        if len(measurement.shape) != 4:
            measurement = measurement.unsqueeze(0)
        
        # == Compute recon-loss == #
        with th.set_grad_enabled(True):
            y_pred = self.measurement_model(x0)
            if len(y_pred.shape) != 4:
                y_pred = y_pred.unsqueeze(0)    
            loss = self.measurement_loss(y_pred, measurement)
            grad = th.autograd.grad(loss, x)[0]
        
        # === step 7 === #
        with th.no_grad():
            #print(f"step_size constant = {th.linalg.norm(th.abs(y_pred-self.measurement))} other constant = {th.sqrt(loss).item()}")
            zeta_i = self.step_size / th.linalg.norm(th.abs(y_pred - self.measurement)) if loss.item() > 0 else self.step_size
            x_new = x_old - zeta_i * grad
        
        # == prints for evaluating progress == #
        # print(f"loss = {loss.item()}")
        # print(f"zeta_i = {zeta_i}")
        # print(f"update magnitude = {th.norm(x_old - x_new)}")

        return x_new.detach()
    
    def p_sample(
            self, 
            model, 
            x, 
            t
    ) -> dict[str, th.Tensor]:
        """Override GaussianDiffusion p_sample with added DPS update step"""
        x_t = x.requires_grad_(True)

        # === step 3 & 4 === #
        with th.set_grad_enabled(True):
            # Here we generate the following values
            # - out["pred_xstart"] = E[x0|xt]
            # - out["mean"] = \hat mu_t 
            # - out["log_variance"] = log(sigma_t)
            # we use the latter two to generate the ancestral sample
            # and the first one is what we use in differentiation
            out = self.p_mean_variance(
                model,
                x_t,
                t
            )
        # === step 5 === #
        noise = th.randn_like(x_t).requires_grad_(False)

        # === step 6 === #
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        x0_hat = out["pred_xstart"]

        # DPS update step
        with th.set_grad_enabled(True):
            dps_sample = self.dps_update(
                x=x_t, 
                t=t,
                x0=x0_hat,
                x_old=sample  
            )

        return {
            "sample": dps_sample,
            "pred_xstart": x0_hat, 
            "mean": out["mean"]
        }
    
    def p_sample_loop_progressive(
        self,
        model,
        shape,
        device=None
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        
        # need to enable gradients from t = T to t = 1
        x_t = th.randn(*shape, device=device).requires_grad_(True) 
            
        indices = list(range(self.num_timesteps))[::-1]
        intermediate_indices = list(np.floor(np.linspace(min(indices), max(indices), len(indices)//4)).astype(int))

        for i in tqdm(indices, desc="Sampling", leave=False):
            t = th.tensor([i] * shape[0], device=device).requires_grad_(False)
            # we free the tensor from the computation graph
            # (each iteration we compute gradient only w.r.t curr sample)
            x_t = x_t.detach()
            
            # Also clear GPU cache (in case there is old garbage)
            if device.type == 'mps':
                th.mps.empty_cache()
            if device.type == "cuda":
                th.cuda.empty_cache()
            
            with th.set_grad_enabled(True):
                out = self.p_sample(
                    model,
                    x=x_t,
                    t=t,
                )
                yield out
            x_t = out["sample"]

            # if i in intermediate_indices:
            #     # save some intermediate images 
            #     img = out["sample"]
            #     #img = denormalize_imagenet(img)
            #     img = img[0] if len(img.shape) == 4 else img
            #     curr_time = time.time()
            #     to_pil = ToPILImage()
            #     image = to_pil(img.cpu())
            #     image.save(f"./intermediate_samples/sample_{i}_{curr_time}.png")

    def p_sample_loop(
        self,
        model,
        shape,
        device=None
    ) -> th.Tensor:
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            device=device
        ):
            final = sample
        return final["sample"]
    
class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)

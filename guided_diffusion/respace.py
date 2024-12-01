import numpy as np
import torch as th
from .gaussian_diffusion import (GaussianDiffusion, )

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
        assert x.shape == y.shape, f"Shape missmatch between operator with shape {x.shape} and observation with shape {y.shape}"
        lambda_matrix = th.diag(1. / (2*y))
        diff = (x - y)
        loss = diff.T @ lambda_matrix @ diff
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
        elif th.backends.cuda.is_available():
            self.measurement = measurement.to("cuda")
        else: 
            self.measurement = measurement
        self.noise_model = noise_model
        self.step_size = step_size
        if self.noise_model == "gaussian":
            self.measurement_loss = th.nn.MSELoss()
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
        
        with th.enable_grad():
            eps = self._predict_eps_from_xstart(x, t, x0)
            x0 = self._predict_xstart_from_eps(x, t, eps)
            x0 = x0[0] if len(x0.shape) == 4 else x0
            
            # Trying to scale A(x0) to be on the same scale as measurement
            y_meas = self.measurement_model(x0)
            y_min, y_max = y_meas.min(), y_meas.max()
            meas_min, meas_max = self.measurement.min(), self.measurement.max()
            y_meas = (y_meas - y_min) * (meas_max - meas_min) / (y_max - y_min) + meas_min
            
            loss = self.measurement_loss(y_meas, self.measurement)
            print(f"Loss value: {loss.item()}")
            loss.backward()
            grad = x.grad
            print(f"Gradient stats - min: {grad.min()}, max: {grad.max()}, mean: {grad.mean()}")
        
        with th.no_grad():
            zeta_i = (self.step_size / np.sqrt(loss.item()))
            print(f"zeta_i = {zeta_i} at t={t[0]}")
            x_new = x_old - zeta_i * grad
            print(f"Update magnitude: {(x_new - x_old).abs().mean()}")
        
        print(f"x0 range: [{x0.min()}, {x0.max()}]")
        print(f"y_meas range: [{y_meas.min()}, {y_meas.max()}]")
        print(f"measurement range: [{self.measurement.min()}, {self.measurement.max()}]")
        return x_new
    
    def p_sample(
        self,
        model,  
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
    
        with th.no_grad():
            out = super().p_sample(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs
            )
            x0_hat = out["pred_xstart"]
            x_mean = out["mean"]
            sample = out["sample"]
        
        x = x.detach().requires_grad_(True)
        dps_sample = self.dps_update(x=x, t=t, x0=x0_hat, x_old=sample)

        return {
            "sample": dps_sample,
            "pred_xstart": x0_hat, 
            "mean": x_mean
        }
    
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

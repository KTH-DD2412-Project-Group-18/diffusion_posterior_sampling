import numpy as np
import torch as th

from .gaussian_diffusion import GaussianDiffusion


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


# TODO: Need to modify this class to handle the different measurement_models
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

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

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
        lambda_matrix = th.diag(1./(2*y))
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
    def __init__(self,
                use_timesteps,
                measurement_model,
                noise_model, 
                step_size=1., 
                **kwargs
    ):
        super().__init__(use_timesteps, **kwargs)
        self.measurement_model = measurement_model
        self.noise_model = noise_model
        self.step_size = step_size
        if self.noise_model == "gaussian":
            self.loss = th.nn.MSELoss()
        elif self.noise_model == "poisson":
            self.loss = PoissonMseLoss()
        else:
            print("only 'gaussian' and 'poisson' noise models!")
            return NotImplementedError
    
    def compute_grad_and_value(self, x_hat, x_old, y):
        """ Compute the gradient and posterior sample"""
        out = self.measurement_operator(x_hat)
        loss = self.loss(out, y)
        loss.backward()
        with th.no_grad():
            grad = x_old.grad
            x_new = x_old - self.step_size * grad # x_{i-1} = x'_{i-1} - \zeta_i ||y-A(\hat{x})||
            x_new.grad.zero()
        return x_new, grad

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

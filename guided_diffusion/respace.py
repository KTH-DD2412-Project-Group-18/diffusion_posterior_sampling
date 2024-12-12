import numpy as np
import torch as th
from .gaussian_diffusion import (GaussianDiffusion, )
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
from dpm_solver.sampler import NoiseScheduleVP, model_wrapper, DPM_Solver

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
        assert x.shape == y.shape, f"Shape missmatch between operator with shape {x.shape} and observation with shape {y.shape}"
        lambda_matrix = th.diag(1. / (2*th.abs(y))) # take absolute value of y (for "correct" mathematics)
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
    

class DPMDiffusionPosteriorSampling(DPM_Solver):
    """
    A combined implementation of DPM-Solver with Diffusion Posterior Sampling.
    Extends DPM-Solver to include posterior sampling updates at each step.
    """
    def __init__(
        self,
        model_fn,
        noise_schedule,
        measurement_model,
        measurement,
        noise_model="gaussian",
        step_size=1.0,
        algorithm_type="dpmsolver++",
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.
    ):
        super().__init__(
            model_fn=model_fn,
            noise_schedule=noise_schedule,
            algorithm_type=algorithm_type,
            correcting_x0_fn=correcting_x0_fn,
            correcting_xt_fn=correcting_xt_fn,
            thresholding_max_val=thresholding_max_val
        )
        self.measurement_model = measurement_model
        self.measurement = measurement
        self.noise_model = noise_model
        self.step_size = step_size
        
        if noise_model == "gaussian":
            self.measurement_loss = th.nn.MSELoss(reduction="sum")
        elif noise_model == "poisson":
            self.measurement_loss = PoissonMseLoss()
        else:
            raise NotImplementedError("Only 'gaussian' and 'poisson' noise models supported")

    def dps_update(self, x, t, x0, x_old):
        """
        Compute the DPS sampling step (gradient update)
        """
        measurement = self.measurement
        if len(measurement.shape) != 4:
            measurement = measurement.unsqueeze(0)
        
        with th.set_grad_enabled(True):
            y_pred = self.measurement_model(x0)
            if len(y_pred.shape) != 4:
                y_pred = y_pred.unsqueeze(0)    
            loss = self.measurement_loss(y_pred, measurement)
            grad = th.autograd.grad(loss, x)[0]
        
        with th.no_grad():
            zeta_i = self.step_size / th.linalg.norm(th.abs(y_pred - self.measurement)) if loss.item() > 0 else self.step_size
            x_new = x_old - zeta_i * grad
        
        return x_new.detach()
    
    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order, solver_type='dpmsolver'):
            """
            Multistep DPM-Solver with the order `order` from time `t_prev_list[-1]` to time `t`.

            Args:
                x: A pytorch tensor. The initial value at time `s`.
                model_prev_list: A list of pytorch tensor. The previous computed model values.
                t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
                t: A pytorch tensor. The ending time, with the shape (1,).
                order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
                solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                    The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
            Returns:
                x_t: A pytorch tensor. The approximated solution at time `t`.
            """
            if order == 1:
                x_next = self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
            elif order == 2:
                x_next = self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
            elif order == 3:
                x_next = self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
            else:
                raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))
            x0_pred = self.data_prediction_fn(x, t_prev_list[-1])
            x_next = self.dps_update(x, t_prev_list[-1], x0_pred, x_next)
            return x_next.detach().requires_grad_(True)

    def sample(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, return_intermediate=False,
    ):
        """
        Compute the sample at time `t_end` by DPM-Solver, given the initial `x` at time `t_start`.

        =====================================================

        We support the following algorithms for both noise prediction model and data prediction model:
            - 'singlestep':
                Singlestep DPM-Solver (i.e. "DPM-Solver-fast" in the paper), which combines different orders of singlestep DPM-Solver. 
                We combine all the singlestep solvers with order <= `order` to use up all the function evaluations (steps).
                The total number of function evaluations (NFE) == `steps`.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    - If `order` == 1:
                        - Denote K = steps. We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - Denote K = (steps // 2) + (steps % 2). We take K intermediate time steps for sampling.
                        - If steps % 2 == 0, we use K steps of singlestep DPM-Solver-2.
                        - If steps % 2 == 1, we use (K - 1) steps of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                    - If `order` == 3:
                        - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                        - If steps % 3 == 0, we use (K - 2) steps of singlestep DPM-Solver-3, and 1 step of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 1, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 2, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of singlestep DPM-Solver-2.
            - 'multistep':
                Multistep DPM-Solver with the order of `order`. The total number of function evaluations (NFE) == `steps`.
                We initialize the first `order` values by lower order multistep solvers.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    Denote K = steps.
                    - If `order` == 1:
                        - We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - We firstly use 1 step of DPM-Solver-1, then use (K - 1) step of multistep DPM-Solver-2.
                    - If `order` == 3:
                        - We firstly use 1 step of DPM-Solver-1, then 1 step of multistep DPM-Solver-2, then (K - 2) step of multistep DPM-Solver-3.
            - 'singlestep_fixed':
                Fixed order singlestep DPM-Solver (i.e. DPM-Solver-1 or singlestep DPM-Solver-2 or singlestep DPM-Solver-3).
                We use singlestep DPM-Solver-`order` for `order`=1 or 2 or 3, with total [`steps` // `order`] * `order` NFE.
            - 'adaptive':
                Adaptive step size DPM-Solver (i.e. "DPM-Solver-12" and "DPM-Solver-23" in the paper).
                We ignore `steps` and use adaptive step size DPM-Solver with a higher order of `order`.
                You can adjust the absolute tolerance `atol` and the relative tolerance `rtol` to balance the computatation costs
                (NFE) and the sample quality.
                    - If `order` == 2, we use DPM-Solver-12 which combines DPM-Solver-1 and singlestep DPM-Solver-2.
                    - If `order` == 3, we use DPM-Solver-23 which combines singlestep DPM-Solver-2 and singlestep DPM-Solver-3.

        =====================================================

        Some advices for choosing the algorithm:
            - For **unconditional sampling** or **guided sampling with small guidance scale** by DPMs:
                Use singlestep DPM-Solver or DPM-Solver++ ("DPM-Solver-fast" in the paper) with `order = 3`.
                e.g., DPM-Solver:
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                            skip_type='time_uniform', method='singlestep')
                e.g., DPM-Solver++:
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                            skip_type='time_uniform', method='singlestep')
            - For **guided sampling with large guidance scale** by DPMs:
                Use multistep DPM-Solver with `algorithm_type="dpmsolver++"` and `order = 2`.
                e.g.
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=2,
                            skip_type='time_uniform', method='multistep')

        We support three types of `skip_type`:
            - 'logSNR': uniform logSNR for the time steps. **Recommended for low-resolutional images**
            - 'time_uniform': uniform time for the time steps. **Recommended for high-resolutional images**.
            - 'time_quadratic': quadratic time for the time steps.

        =====================================================
        Args:
            x: A pytorch tensor. The initial value at time `t_start`
                e.g. if `t_start` == T, then `x` is a sample from the standard normal distribution.
            steps: A `int`. The total number of function evaluations (NFE).
            t_start: A `float`. The starting time of the sampling.
                If `T` is None, we use self.noise_schedule.T (default is 1.0).
            t_end: A `float`. The ending time of the sampling.
                If `t_end` is None, we use 1. / self.noise_schedule.total_N.
                e.g. if total_N == 1000, we have `t_end` == 1e-3.
                For discrete-time DPMs:
                    - We recommend `t_end` == 1. / self.noise_schedule.total_N.
                For continuous-time DPMs:
                    - We recommend `t_end` == 1e-3 when `steps` <= 15; and `t_end` == 1e-4 when `steps` > 15.
            order: A `int`. The order of DPM-Solver.
            skip_type: A `str`. The type for the spacing of the time steps. 'time_uniform' or 'logSNR' or 'time_quadratic'.
            method: A `str`. The method for sampling. 'singlestep' or 'multistep' or 'singlestep_fixed' or 'adaptive'.
            denoise_to_zero: A `bool`. Whether to denoise to time 0 at the final step.
                Default is `False`. If `denoise_to_zero` is `True`, the total NFE is (`steps` + 1).

                This trick is firstly proposed by DDPM (https://arxiv.org/abs/2006.11239) and
                score_sde (https://arxiv.org/abs/2011.13456). Such trick can improve the FID
                for diffusion models sampling by diffusion SDEs for low-resolutional images
                (such as CIFAR-10). However, we observed that such trick does not matter for
                high-resolutional images. As it needs an additional NFE, we do not recommend
                it for high-resolutional images.
            lower_order_final: A `bool`. Whether to use lower order solvers at the final steps.
                Only valid for `method=multistep` and `steps < 15`. We empirically find that
                this trick is a key to stabilizing the sampling by DPM-Solver with very few steps
                (especially for steps <= 10). So we recommend to set it to be `True`.
            solver_type: A `str`. The taylor expansion type for the solver. `dpmsolver` or `taylor`. We recommend `dpmsolver`.
            atol: A `float`. The absolute tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
            rtol: A `float`. The relative tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
            return_intermediate: A `bool`. Whether to save the xt at each step.
                When set to `True`, method returns a tuple (x0, intermediates); when set to False, method returns only x0.
        Returns:
            x_end: A pytorch tensor. The approximated solution at time `t_end`.

        """
        if True:
            x = x.detach().requires_grad_(True)
            t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
            t_T = self.noise_schedule.T if t_start is None else t_start
            assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
            if return_intermediate:
                assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when saving intermediate values"
            if self.correcting_xt_fn is not None:
                assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when correcting_xt_fn is not None"
            device = x.device
            assert steps >= order
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
            assert timesteps.shape[0] - 1 == steps
        
        # Init the initial values.
        print("Hello!")
        
        step = 0
        t = timesteps[step]
        t_prev_list = [t]
        model_prev_list = [self.model_fn(x, t)]
        
        print("Made it here!")
        # Init the first `order` values by lower order multistep DPM-Solver.
        for step in range(1, order):
            x = x.detach().requires_grad_(True)
            t = timesteps[step]
            t_prev_list.append(t)
            model_prev_list.append(self.model_fn(x, t))
        print("I even made it here!")

        # Compute the remaining values by `order`-th order multistep DPM-Solver.
        for step in range(order, steps + 1):
            t = timesteps[step]
            # We only use lower order for steps < 10
            if lower_order_final and steps < 10:
                step_order = min(order, steps + 1 - step)
            else:
                step_order = order
            x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step_order, solver_type=solver_type)

            for i in range(order - 1):
                t_prev_list[i] = t_prev_list[i + 1]
                model_prev_list[i] = model_prev_list[i + 1]
            t_prev_list[-1] = t

            if step < steps:
                model_prev_list[-1] = self.model_fn(x, t)
        else:
            print("I won!")
            return x

    def multistep_dpm_solver_third_update(self, x, model_prev_list, t_prev_list, t, solver_type='dpmsolver'):
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        return x + model_prev_0

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
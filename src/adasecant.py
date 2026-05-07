import torch
from torch.optim.optimizer import Optimizer


class Adasecant(Optimizer):
    """
    Parameters
    ----------
    params : iterable of torch.nn.Parameter
        The parameters to optimize.
    decay : float
        Fixed rho for the gradient statistics.
    gamma_clip : float
        Upper bound on the variance reduction coefficient gamma. authors suggest 1.8 
    skip_nan_inf : bool
        If True, replace NaN/Inf entries in the gradient with 0.
    use_corrected_grad : bool
        If True, store the corrected gradient g_tilde as g_old for the
        next step's gamma computation.
    """

    def __init__(
        self,
        params,
        decay=0.95,
        gamma_clip=1.8,
        skip_nan_inf=False,
        use_corrected_grad=True,
    ):

        assert decay >= 0.0
        assert decay < 1.0

        self.damping = 1e-7
        self.upper_bound_tau = 1e8
        self.lower_bound_tau = 1.5

        defaults = dict(
            decay=decay,
            gamma_clip=gamma_clip,
            skip_nan_inf=skip_nan_inf,
            use_corrected_grad=use_corrected_grad,
        )
        super().__init__(params, defaults)

    # initialisation with 0
    def _init_state(self, p, group):
        state = self.state[p]
        eps = self.damping
        tau = 2.2

        # step counter
        state["step"] = 0

        # moving avgs of gradients, fixed decay rho
        state["mean_grad"] = torch.full_like(p.data, eps)
        state["mean_square_grad"] = torch.full_like(p.data, eps)

        # moving avgs of update step, adaptive decay 1/tau
        state["mean_dx"] = torch.zeros_like(p.data)
        state["mean_square_dx"] = torch.zeros_like(p.data)

        # moving avgs of curvature (alpha = g_t - g_{t-1}), adaptive decay 1/tau
        state["mean_curvature"] = torch.full_like(p.data, eps)
        state["mean_curvature_sqr"] = torch.full_like(p.data, eps)

        # moving avgs of gamma numerator/denominator, adaptive decay 1/tau
        state["gamma_num_sqr"] = torch.full_like(p.data, eps)
        state["gamma_den_sqr"] = torch.full_like(p.data, eps)

        # Covariance E[Delta * alpha] — adaptive decay 1/tau
        state["cov_num"] = torch.full_like(p.data, eps)

        state["tau"] = torch.full_like(p.data, (1.0 + eps) * tau)

        # old gradients
        state["g_old_plain"] = torch.full_like(p.data, eps)  # raw
        state["g_old"] = torch.full_like(p.data, eps)  # corrected (g~)

    # optimizer step
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = self.damping
            tau_min = self.lower_bound_tau
            tau_max = self.upper_bound_tau
            decay = group["decay"]
            gamma_clip = group["gamma_clip"]
            use_corrected_grad = group["use_corrected_grad"]
            skip_nan_inf = group["skip_nan_inf"]

            # gather and block-normalise gradients
            normalised_grads = {}
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.data

                # zero if nan/inf
                if skip_nan_inf:
                    bad = torch.isnan(g) | torch.isinf(g)
                    g = torch.where(bad, torch.zeros_like(g), g)

                # block-normalise
                g_norm = g.norm(2) + eps
                normalised_grads[p] = g / g_norm

            # per parameter update
            for p, g in normalised_grads.items():
                if len(self.state[p]) == 0:
                    self._init_state(p, group)
                state = self.state[p]

                step = state["step"]
                mean_grad = state["mean_grad"]
                mean_square_grad = state["mean_square_grad"]
                mean_dx = state["mean_dx"]
                mean_square_dx = state["mean_square_dx"]
                mean_curvature = state["mean_curvature"]
                mean_curvature_sqr = state["mean_curvature_sqr"]
                gamma_num_sqr = state["gamma_num_sqr"]
                gamma_den_sqr = state["gamma_den_sqr"]
                cov_num = state["cov_num"]
                tau = state["tau"]
                g_old_plain = state["g_old_plain"]
                g_old = state["g_old"]

                if step == 0:
                    msdx_eff = g * g
                    mdx_eff = g.clone()
                else:
                    msdx_eff = mean_square_dx
                    mdx_eff = mean_dx

                # E[g] - running avg of g
                new_mean_grad = decay * mean_grad + (1 - decay) * g
                new_mean_square_grad = decay * mean_square_grad + (1 - decay) * (g * g)

                # gamma
                obs_num = ((g_old - g) * (g_old - mean_grad)) ** 2
                obs_den = ((g_old - mean_grad) * (g - mean_grad)) ** 2

                inv_tau = 1.0 / tau
                new_gamma_num_sqr = (1 - inv_tau) * gamma_num_sqr + inv_tau * obs_num
                new_gamma_den_sqr = (1 - inv_tau) * gamma_den_sqr + inv_tau * obs_den

                gamma = torch.sqrt(new_gamma_num_sqr) / (torch.sqrt(new_gamma_den_sqr) + eps)
                if gamma_clip is not None:
                    gamma = torch.clamp(gamma, max=gamma_clip)

                # gamma~
                g_tilde = (g + gamma * new_mean_grad) / (1 + gamma)

                # alpha
                alpha = g - g_old_plain
                alpha_sqr = alpha * alpha

                new_mean_curvature = (1 - inv_tau) * mean_curvature + inv_tau * alpha
                new_mean_curvature_sqr = (
                    1 - inv_tau
                ) * mean_curvature_sqr + inv_tau * alpha_sqr

                # step size
                rms_dx = torch.sqrt(msdx_eff + eps)
                rms_curv = torch.sqrt(new_mean_curvature_sqr + eps)

                delta = -(rms_dx / rms_curv - cov_num / (new_mean_curvature_sqr + eps)) * g_tilde

                # delta moving averages
                new_mean_square_dx = (1 - inv_tau) * mean_square_dx + inv_tau * (
                    delta * delta
                )
                new_mean_dx = (1 - inv_tau) * mean_dx + inv_tau * delta

                # tau
                ratio = (mdx_eff * mdx_eff) / (msdx_eff + eps)
                new_tau_step = (1 - ratio) * tau + (1.0 + eps)

                # outlier detection (if triggered, reset tau to 2.2)
                var_g = torch.clamp(new_mean_square_grad - new_mean_grad**2, min=0.0)
                var_alpha = torch.clamp(
                    new_mean_curvature_sqr - new_mean_curvature**2, min=0.0
                )
                std_g = torch.sqrt(var_g)
                std_alpha = torch.sqrt(var_alpha)

                grad_outlier = (g - new_mean_grad).abs() > 2 * std_g
                curve_outlier = (alpha - new_mean_curvature).abs() > 2 * std_alpha
                is_outlier = grad_outlier | curve_outlier

                new_tau = torch.where(
                    is_outlier,
                    torch.full_like(new_tau_step, 2.2),
                    new_tau_step,
                )
                new_tau = torch.clamp(new_tau, min=tau_min, max=tau_max)

                # covariance E[Delta * alpha] (uses old tau)
                new_cov_num = (1 - inv_tau) * cov_num + inv_tau * (delta * alpha)

                # apply the parameter update
                p.data.add_(delta)

                state["mean_grad"] = new_mean_grad
                state["mean_square_grad"] = new_mean_square_grad
                state["mean_dx"] = new_mean_dx
                state["mean_square_dx"] = new_mean_square_dx
                state["mean_curvature"] = new_mean_curvature
                state["mean_curvature_sqr"] = new_mean_curvature_sqr
                state["gamma_num_sqr"] = new_gamma_num_sqr
                state["gamma_den_sqr"] = new_gamma_den_sqr
                state["cov_num"] = new_cov_num
                state["tau"] = new_tau

                state["g_old_plain"] = g.clone()
                if use_corrected_grad:
                    state["g_old"] = g_tilde.clone()

                state["step"] = step + 1

        return loss

import torch
from torch import nn

from src.utils.logger import LOGGER

SAVE_METRICS_EACH = 100


def clamp_quantile(weight, a_max):
    # quantile-based clamp on weight distribution
    if a_max is not None:
        max_value = torch.quantile(weight, a_max).item()
        weight = weight.clamp(min=0, max=max_value)
        return weight, max_value
    else:
        return weight, a_max


class DiscFormer:
    # tracker class for all things discformer

    def __init__(self, cfg):
        self.cfg = cfg
        param_string = [
            f"{key}={value}, "
            for key, value in self.cfg.discformer.discformation.items()
        ]
        LOGGER.info(f"Initialized discformer with {''.join(param_string)}")

    def prepare_training(self, max_steps):
        self.max_steps = max_steps
        self.keys = ["weight_cls", "weight_gen", "weight_full", "weight_df", "factor"]
        self.alpha = []
        self.clamp_ratio = {
            f"{n_jets}j": {key: [] for key in self.keys}
            for n_jets in self.cfg.data.n_jets_list
        }
        self.efficiency = {
            f"{n_jets}j": {key: [] for key in self.keys}
            for n_jets in self.cfg.data.n_jets_list
        }
        self.weight_quantiles = {
            f"{n_jets}j": {
                key: {
                    quantile: []
                    for quantile in self.cfg.discformer.discformation.quantiles
                }
                for key in self.keys
            }
            for n_jets in self.cfg.data.n_jets_list
        }

    def alpha_scheduler(self, step, iteration, weight_cls=None):
        # do not implement classifier-dependent alpha for now
        fac = step / self.max_steps

        # feel free to implement more schedulers!
        if self.cfg.discformer.discformation.alpha_scheduler is None:
            # constant alpha
            return self.cfg.discformer.discformation.alpha
        elif self.cfg.discformer.discformation.alpha_scheduler == "OneCycle":
            # one cycle scheduler (from 1 to a maximum value of 1+alpha)
            if fac < 0.5:
                return 1 + 2 * fac * (self.cfg.discformer.discformation.alpha - 1)
            else:
                return 1 + 2 * (1 - fac) * (self.cfg.discformer.discformation.alpha - 1)
        elif self.cfg.discformer.discformation.alpha_scheduler == "StepIteration":
            # step iteration scheduler
            if iteration == 1:
                return self.cfg.discformer.discformation.alpha * 1
            if iteration == 2:
                return self.cfg.discformer.discformation.alpha * 1
            if iteration == 3:
                return self.cfg.discformer.discformation.alpha * 1
            if iteration == 4:
                return self.cfg.discformer.discformation.alpha * 2
            if iteration == 5:
                return self.cfg.discformer.discformation.alpha * 2
        elif self.cfg.discformer.discformation.alpha_scheduler == "DiscifierDependent":
            D = weight_cls / (1 + weight_cls)
            return self.cfg.discformer.discformation.alpha * (0.5 - D).abs()
        elif self.cfg.discformer.discformation.alpha_scheduler == "DiscFlow":
            D = weight_cls / (1 + weight_cls)
            alpha_min = self.cfg.discformer.discformation.alpha
            alpha_max = 3.0
            alpha0 = alpha_min + fac * (alpha_max - alpha_min)
            return alpha0 * (0.5 - D).abs()
        else:
            raise NotImplementedError(
                f"Alpha scheduler {self.cfg.discformer.discformation.alpha_scheduler} not implemented"
            )

    def get_weight(
        self,
        weight_cls,
        weight_gen=None,
        alpha=None,
        ijet=None,
        step=None,
        track_weights=False,
    ):
        """
        Computes and clamps each component of the discformer loss factor.
        """
        alpha = (self.cfg.discformer.discformation.alpha) if alpha is None else alpha

        if weight_gen is None:
            weight_gen = torch.ones_like(weight_cls)
        else:
            weight_gen = clamp_quantile(
                weight_gen,
                a_max=self.cfg.discformer.discformation.weight_gen_max,
            )[0]

        weight_cls = clamp_quantile(
            weight_cls,
            a_max=self.cfg.discformer.discformation.weight_cls_max,
        )[0]
        weight_full = clamp_quantile(
            weight_cls * weight_gen,
            a_max=self.cfg.discformer.discformation.weight_full_max,
        )[0]
        weight_df = clamp_quantile(
            torch.pow(weight_full, alpha),
            a_max=self.cfg.discformer.discformation.weight_df_max,
        )[0]

        if not self.cfg.discformer.discformation.sample_model:
            factor = weight_df
        else:
            factor = weight_df * weight_cls
        factor = clamp_quantile(
            factor,
            a_max=self.cfg.discformer.discformation.factor_max,
        )[0]
        if track_weights:
            assert ijet is not None, "ijet must be provided to track weights"
            assert step is not None, "step must be provided to track weights"
            self.track_loss_factor_weights(
                weight_cls,
                weight_gen,
                weight_full,
                weight_df,
                factor,
                alpha,
                ijet,
                step,
            )
        return factor

    def loss_factor(
        self, log_prob1, log_prob0, weight_cls, ijet, iteration, step, train=True
    ):
        """
        Calculates the discformer loss pre-factor.
        """
        if self.cfg.discformer.discformation.detach_factor:
            log_prob1 = log_prob1.detach()
        weight_gen = torch.exp(log_prob0 - log_prob1)
        alpha = self.alpha_scheduler(step, iteration, weight_cls)
        factor = self.get_weight(
            weight_cls, weight_gen, alpha, ijet, step, track_weights=True
        )
        return factor

    def track_loss_factor_weights(
        self, weight_cls, weight_gen, weight_full, weight_df, factor, alpha, ijet, step
    ):
        """
        This functions is designed to be called by get_weights() during training.
        Tracks the weights of the different components of the discformer loss factor.
        """
        n_jets = {
            ijet: n_jets for ijet, n_jets in enumerate(self.cfg.data.n_jets_list)
        }[ijet]
        if ijet == 0:
            # alpha is the same for all multiplicities -> only store it once
            if torch.is_tensor(alpha):
                self.alpha.append(alpha.mean().cpu().numpy())
            else:
                self.alpha.append(alpha)
        if step % SAVE_METRICS_EACH == 0:
            for key in self.keys:
                w = eval(key).detach().cpu()
                a_max = eval(f"self.cfg.discformer.discformation.{key}_max")
                w_cut = clamp_quantile(w, a_max=a_max)[1]
                self.clamp_ratio[f"{n_jets}j"][key].append(
                    (w > w_cut).float().mean() if w_cut is not None else 0
                )
                self.efficiency[f"{n_jets}j"][key].append(w.mean() / w.max())
                for quantile in self.cfg.discformer.discformation.quantiles:
                    self.weight_quantiles[f"{n_jets}j"][key][quantile].append(
                        torch.quantile(w, quantile)
                    )

import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from src.eventgen.jetgpt import JetGPT
from src.eventgen.jetgptcomstop import get_running_com

ZMUMU_TABLE = torch.tensor(
    [
        [10.10, 0.0, 0.0, 0.0, 0.0, 0.0],
        [35.42, 13.27, 0.0, 0.0, 0.0, 0.0],
        [53.91, 31.14, 16.10, 0.0, 0.0, 0.0],
        [71.16, 47.25, 30.57, 19.16, 0.0, 0.0],
        [87.90, 63.05, 44.07, 31.68, 22.27, 0.0],
        [103.99, 78.37, 57.24, 43.30, 33.24, 25.15],
    ]
)
# ZMUMU_TABLE = torch.tensor(  # pT ordered
#     [
#         [10.10, 0.00, 0.00, 0.00, 0.00, 0.00],
#         [35.42, 13.27, 0.00, 0.00, 0.00, 0.00],
#         [53.91, 30.99, 16.10, 0.00, 0.00, 0.00],
#         [71.16, 47.08, 30.25, 19.16, 0.00, 0.00],
#         [87.90, 62.87, 43.73, 31.25, 22.27, 0.00],
#         [103.99, 78.17, 56.89, 42.82, 32.73, 25.15],
#     ]
# )


def pdf_normal_custom2d(x, std):
    return pdf_normal(x[..., [0]], 0, std) * pdf_normal(x[..., [1]], 0, std)


def pdf_normal(x, mean, std):
    return torch.exp(-0.5 * ((x - mean) / std) ** 2) / (std * (2 * torch.pi) ** 0.5)


class JetGPTPhysicsLoss(JetGPT):
    def __init__(
        self,
        *args,
        loss_lambda=1.0,
        warmup_steps=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_lambda = loss_lambda
        self.warmup_steps = warmup_steps
        self.register_buffer("std_table", ZMUMU_TABLE)  # specific to the zmumu dataset
        self.std_table[self.std_table == 0] = torch.inf

    def build_posterior(self, com, eps=1e-8):
        B, N, _ = com.shape
        pidx = torch.arange(N)
        likelihood_complete = pdf_normal_custom2d(
            com, self.std_table[pidx, pidx].unsqueeze(1)
        ).squeeze(2)
        evidence = pdf_normal_custom2d(
            com, self.std_table[:, pidx].T.repeat(B, 1, 1)
        ).sum(-1)

        posterior = likelihood_complete / (evidence + eps)  # uniform prior
        return posterior  # (B, N)

    def _batch_loss(self, fourmomenta, num_particles, step):

        # get kinematic logprob and stop logits
        log_prob_kinematics, stop_logit, masks = self.forward(
            fourmomenta, num_particles
        )

        # no stop signal during hard process
        M = self.n_hard_particles
        stop_logit = stop_logit[:, M - 1 :]
        eos_mask = masks["eos"][:, M - 1 :]
        padding_mask = masks["padding"][:, M - 1 :]

        # stop probability
        stop_target = torch.where(eos_mask, 1.0, 0.0)

        # build stop posterior based on px & py of CoM
        running_com = get_running_com(fourmomenta)
        px_py_isr = running_com[:, M - 1 :, [1, 2]]  # ignore hard particles
        physics_target = self.build_posterior(px_py_isr)

        # combine targets
        lam = self.loss_lambda if (step is None or step >= self.warmup_steps) else 0.0
        stop_target = (1 - lam) * stop_target + lam * physics_target

        # get loss
        log_prob_stop = -binary_cross_entropy_with_logits(
            stop_logit, stop_target, reduction="none"
        )

        # don't include zero-padded particles
        log_prob_stop[padding_mask] = 0.0

        # combine terms
        log_prob = log_prob_kinematics.sum(dim=[-1, -2]) + log_prob_stop.sum(dim=-1)

        return -log_prob.mean()

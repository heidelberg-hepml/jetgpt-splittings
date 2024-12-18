import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from src.eventgen.jetgpt import JetGPT


class JetGPTExtendLoss(JetGPT):
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

    def _batch_loss(self, fourmomenta, num_particles, step):

        # get kinematic logprob and stop logits
        log_prob_kinematics, stop_logit, masks = self.forward(
            fourmomenta, num_particles
        )

        # stop probability
        stop_target = torch.where(masks["eos"], 1.0, 0.0)

        # adjust stop prob for max-length events
        if step is None or step >= self.warmup_steps:
            stop_target[:, -1] = stop_target[:, -1] * (1 - self.loss_lambda)

        # adjust stop prob for max-length events
        log_prob_stop = -binary_cross_entropy_with_logits(
            stop_logit, stop_target, reduction="none"
        )

        # don't stop for hard particles
        log_prob_stop[:, : self.n_hard_particles - 1] = 0.0
        # don't include zero-padded particles
        log_prob_stop[masks["padding"]] = 0.0

        # combine terms
        log_prob = log_prob_kinematics.sum(dim=[-1, -2]) + log_prob_stop.sum(dim=-1)

        return -log_prob.mean()

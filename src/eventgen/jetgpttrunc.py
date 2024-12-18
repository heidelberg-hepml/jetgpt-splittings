import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from src.eventgen.jetgpt import JetGPT


class JetGPTTrunc(JetGPT):
    def _batch_loss(self, fourmomenta, num_particles, step):

        # get kinematic logprob and stop logits
        log_prob_kinematics, stop_logit, masks = self.forward(
            fourmomenta, num_particles
        )

        # stop probability
        stop_target = torch.where(masks["eos"], 1.0, 0.0)

        # get bce loss, ignoring the stop for max-length events
        log_prob_stop = -binary_cross_entropy_with_logits(
            stop_logit[:, :-1], stop_target[:, :-1], reduction="none"
        )

        log_prob_stop[:, : self.n_hard_particles - 1] = 0.0
        log_prob_stop[masks["padding"][:, :-1]] = 0.0

        log_prob = log_prob_kinematics.sum(dim=[-1, -2]) + log_prob_stop.sum(dim=-1)

        return -log_prob.mean()

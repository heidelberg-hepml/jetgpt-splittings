import torch

from src.eventgen.jetgpt import JetGPT
from src.eventgen.helpers import EPPP_to_PtPhiEtaM2
from src.networks.mlp import MLP


class JetGPTCoMStop(JetGPT):
    def __init__(
        self,
        *args,
        com_channels=(0, 2, 3),
        use_hard_cond=True,
        hard_cond_dim=64,
        mlp_hidden_channels=64,
        mlp_layers=2,
        mlp_pdrop=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.com_channels = com_channels
        self.use_hard_cond = use_hard_cond
        mlp_in_channels = len(com_channels) + (hard_cond_dim if use_hard_cond else 0)
        self.stop_mlp = MLP(
            in_shape=mlp_in_channels,
            out_shape=1,
            hidden_channels=mlp_hidden_channels,
            hidden_layers=mlp_layers,
            dropout_prob=mlp_pdrop,
        )

    def _extract_condition(self, condition, fourmomenta=None, pidx=None, sampling=None):

        # pformer condition
        pcondition = condition[..., 1:]  # (B, N, C)

        if fourmomenta is None:
            return pcondition, None  # when sampling hard process

        B, N, _ = fourmomenta.shape

        # hard process condition
        hcondition = condition[:, [self.n_hard_particles], :]  # (B, 1, C)

        if sampling:

            # calculate running center of momentum
            com = EPPP_to_PtPhiEtaM2(
                self.preprocessing.undo_preprocess(
                    fourmomenta[:, 1:, :], pidx[:, 1:]
                ).sum(-2)
            )  # (B, 4)
            com = self.preprocess_com(com)

            # stack stopping variables
            stop_vars = [com[..., self.com_channels]]
            if self.use_hard_cond:
                stop_vars.append(hcondition.squeeze(1))
            stop_vars = torch.cat(stop_vars, dim=-1)  # (B, C)

            stop_logit = self.stop_mlp(stop_vars)

        else:
            # calculate running center of momenta
            running_com = EPPP_to_PtPhiEtaM2(get_running_com(fourmomenta))  # (B, N, 4)
            running_com = self.preprocess_com(running_com)

            # stack stopping variables
            stop_vars = [running_com[..., self.com_channels]]
            if self.use_hard_cond:
                stop_vars.append(hcondition.repeat(1, N, 1))
            stop_vars = torch.cat(stop_vars, dim=-1)  # (B, N, C)

            # predict stop
            stop_logit = self.stop_mlp(stop_vars.flatten(0, 1)).reshape((B, -1))

        return pcondition, stop_logit

    def preprocess_com(self, com):
        com_copy = com.clone()
        com_copy[..., 0] = com[..., 0].add(1e-5).log() / 4.0  # log(pT)
        com_copy[..., 2] = com[..., 2].abs() / 5.0  # abs(eta)
        return com_copy


def get_running_com(fourmomenta):
    N = fourmomenta.size(1)
    event_square = fourmomenta.unsqueeze(-2).repeat(1, 1, N, 1)  # (B, N, N, 4)
    running_com = (
        event_square
        * torch.triu(torch.ones(N, N, device=fourmomenta.device))[None, :, :, None]
    ).sum(
        axis=1
    )  # (B, N, 4)
    return running_com

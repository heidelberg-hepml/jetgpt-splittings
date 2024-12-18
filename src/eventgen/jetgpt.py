import torch
from torch import nn
from torch.nn.functional import one_hot
from torch.nn.functional import sigmoid, binary_cross_entropy_with_logits

from src.eventgen import preprocessing
from src.eventgen.mixturemodel import CustomMixtureModel
from src.logger import LOGGER


class JetGPT(nn.Module):
    def __init__(
        self,
        pformer,
        cformer,
        n_gauss,
        preprocessing_cfg,
        bayesian=False,
    ):
        super().__init__()
        self.pformer = pformer
        self.cformer = cformer
        self.preprocessing_cfg = preprocessing_cfg
        self.mixturemodel = CustomMixtureModel(n_gauss)

        self.bayesian = bayesian
        if self.bayesian:
            self.register_buffer("traindata_size", torch.tensor(0))

    def extra_information(
        self,
        units,
        pt_min,
        onshell_list,
        n_hard_particles,
        n_jets_max,
        device,
    ):
        self.units = units
        self.pt_min = [x / units for x in pt_min]
        self.onshell_list = onshell_list
        self.n_hard_particles = n_hard_particles
        self.n_jets_max = n_jets_max
        self.device = device
        self.num_classes = n_hard_particles + 1

        dtype_prep = torch.float64 if self.preprocessing_cfg.float64 else torch.float32
        preprocessing_cls = getattr(preprocessing, self.preprocessing_cfg.preprocessor)
        self.preprocessing = preprocessing_cls(
            pt_min=self.pt_min,
            units=self.units,
            onshell_list=self.onshell_list,
            n_hard_particles=self.n_hard_particles,
            device=device,
            dtype=dtype_prep,
        )

    def batch_loss(self, fourmomenta, num_particles, step):
        # wrapper around _batch_loss that adds KL divergence if BNN
        loss = self._batch_loss(fourmomenta, num_particles, step)

        if self.bayesian:
            assert self.traindata_size > 0
            loss += self.KL() / self.traindata_size
        return loss

    def _batch_loss(self, fourmomenta, num_particles, step):
        return -self.log_prob(fourmomenta, num_particles).mean()

    def forward(self, fourmomenta, num_particles):

        assert torch.isfinite(fourmomenta).all()
        B, N, _ = fourmomenta.shape

        pidx = torch.arange(N, device=fourmomenta.device)

        pidx[pidx > self.n_hard_particles] = self.n_hard_particles
        x = self.preprocessing.preprocess(fourmomenta, pidx)
        # set unphysical components to zero to avoid nans and other confusion
        x[..., self.onshell_list, 3] = 0.0
        padding_mask = torch.arange(N, device=x.device).repeat(
            B, 1
        ) >= num_particles.unsqueeze(-1)
        x[padding_mask] = 0.0
        finite_mask = torch.isfinite(x).all(dim=[-1, -2])
        x = x[finite_mask]
        num_particles = num_particles[finite_mask]
        assert torch.isfinite(x).all()

        # call pformer
        x_condition = torch.cat((torch.zeros_like(x[:, [0], :]), x), dim=-2)
        pidx_extended = torch.cat(
            (pidx, self.n_hard_particles * torch.ones_like(pidx[[0]]))
        )
        condition = self._get_pformer_condition(x_condition, pidx_extended)
        pcondition, stop_logit = self._extract_condition(condition, fourmomenta)
        pcondition = pcondition[:, :-1, :]  # only need stop_logit from last particle

        # call cformer
        pcondition = pcondition.unsqueeze(-2).repeat(1, 1, 4, 1)
        x_condition = torch.cat((torch.zeros_like(x[:, :, [0]]), x[:, :, :-1]), dim=-1)
        cidx = torch.arange(4, device=x.device)
        ccondition = self._get_cformer_condition(x_condition, pcondition, cidx)
        is_angle = cidx == 1
        log_prob_kinematics = self.mixturemodel.log_prob(x, ccondition, is_angle)
        log_prob_kinematics[
            ..., self.onshell_list, 3
        ] = 0.0  # mass of onshell events is not learned
        log_prob_kinematics[padding_mask] = 0.0  # dont include zero-padded particles

        eos_mask = (
            torch.arange(N, device=fourmomenta.device).repeat(B, 1)
            == (num_particles - 1)[:, None]
        )
        masks = {"padding": padding_mask, "eos": eos_mask}

        return log_prob_kinematics, stop_logit, masks

    def log_prob(self, fourmomenta, num_particles):

        # get kinematic logprob and stop logits
        log_prob_kinematics, stop_logit, masks = self.forward(
            fourmomenta, num_particles
        )

        # stop probability
        stop_target = torch.where(masks["eos"], 1.0, 0.0)
        log_prob_stop = -binary_cross_entropy_with_logits(
            stop_logit, stop_target, reduction="none"
        )

        # don't stop for hard particles
        log_prob_stop[:, : self.n_hard_particles - 1] = 0.0
        # don't include zero-padded particles
        log_prob_stop[masks["padding"]] = 0.0

        log_prob = log_prob_kinematics.sum(dim=[-1, -2]) + log_prob_stop.sum(dim=-1)
        return log_prob

    def _extract_condition(self, condition, fourmomenta=None, sampling=False):
        stop_logit = condition[:, 1:, 0]
        pcondition = condition[..., 1:]
        return pcondition, stop_logit

    def _get_pformer_condition(self, x, pidx):
        pidx_embedding = one_hot(pidx, num_classes=self.num_classes)
        pidx_embedding = pidx_embedding.repeat(x.shape[0], 1, 1)
        embedding = torch.cat((x, pidx_embedding), dim=-1)
        condition = self.pformer(embedding, is_causal=True)
        return condition

    def _get_cformer_condition(self, x, pcondition, cidx):
        cidx_embedding = one_hot(cidx, num_classes=4)
        cidx_embedding = cidx_embedding.repeat(*x.shape[:-1], 1, 1)
        embedding = torch.cat((x[..., None], cidx_embedding, pcondition), dim=-1)
        condition = self.cformer(embedding, is_causal=True)
        return condition

    def _sample_particle(self, pcondition, i, device, dtype):
        nevents = pcondition.shape[0]
        ncomponents = 3 if i in self.onshell_list else 4
        cidx_full = torch.arange(ncomponents, device=device)
        xparticle = torch.zeros(nevents, 1, device=device, dtype=dtype)
        for icomponent in range(ncomponents):
            cidx = cidx_full[: icomponent + 1]
            pcondition2 = pcondition.repeat(1, icomponent + 1, 1)
            ccondition = self._get_cformer_condition(
                xparticle,
                pcondition2,
                cidx,
            )
            ccondition = ccondition[..., [-1], :]
            is_angle = (cidx[-1] == 1).item()
            xnext = self.mixturemodel.sample(ccondition, is_angle)
            xparticle = torch.cat((xparticle, xnext), dim=-1)
        if ncomponents == 3:  # append zeros for the mass
            xparticle = torch.cat(
                (xparticle, torch.zeros(nevents, 1, device=device, dtype=dtype)),
                dim=-1,
            )
        xparticle = xparticle[:, 1:]
        return xparticle

    def sample(self, nevents, device, dtype):
        valid_events_dict = {
            ijet: torch.empty(
                (0, self.n_hard_particles + ijet, 4), device=device, dtype=dtype
            )
            for ijet in range(self.n_jets_max)
        }
        prob_stop_dict = {
            ijet: torch.empty(0, ijet + 1, device=device, dtype=dtype)
            for ijet in range(self.n_jets_max)
        }

        # hard process
        x = torch.zeros(nevents, 1, 4, device=device, dtype=dtype)
        prob_stop = torch.zeros(nevents, self.n_jets_max, device=device, dtype=dtype)
        pidx_hard = torch.arange(self.n_hard_particles, device=device)
        for ihard in range(self.n_hard_particles):
            # extract pcondition
            pidx = pidx_hard[: ihard + 1][None, :]
            condition = self._get_pformer_condition(x, pidx)
            pcondition, _ = self._extract_condition(condition)
            pcondition = pcondition[..., [-1], :]

            # generate particle
            xparticle = self._sample_particle(pcondition, ihard, device, dtype)
            x = torch.cat((x, xparticle[..., None, :]), dim=-2)

        # extra jets
        for ijet in range(self.n_jets_max):
            # extract pcondition and stop_logit
            pidx = torch.cat(
                (pidx, self.n_hard_particles * torch.ones_like(pidx[:, [0]])), dim=-1
            )
            condition = self._get_pformer_condition(x, pidx)
            pcondition, stop_logit = self._extract_condition(
                condition, fourmomenta=x, sampling=True
            )

            pcondition, stop_logit = pcondition[..., [-1], :], stop_logit[..., -1]

            # extract valid events based on stop_logit
            prob_stop_next = sigmoid(stop_logit)
            prob_stop[:, ijet] = prob_stop_next
            stop_mask = torch.rand_like(prob_stop_next) < prob_stop_next
            valid_events_dict[ijet] = x[stop_mask][:, 1:, :]
            prob_stop_dict[ijet] = prob_stop[stop_mask, : ijet + 1]
            prob_stop = prob_stop[~stop_mask]
            x = x[~stop_mask]
            pcondition = pcondition[~stop_mask]
            if x.shape[0] == 0:
                # nothing left to generate -> stop
                break

            # generate xnext based on pcondition
            xparticle = self._sample_particle(
                pcondition, self.n_hard_particles, device, dtype
            )
            x = torch.cat((x, xparticle[..., None, :]), dim=-2)

        # undo preprocessing
        for ijet, events in valid_events_dict.items():
            pidx_jets = self.n_hard_particles * torch.ones(
                ijet, device=device, dtype=torch.long
            )
            pidx = torch.cat((pidx_hard, pidx_jets))
            valid_events_dict[ijet] = self.preprocessing.undo_preprocess(events, pidx)
            invalid_mask = ~torch.isfinite(valid_events_dict[ijet]).all(dim=[-1, -2])
            if invalid_mask.any():
                LOGGER.warning(f"Generated {invalid_mask.sum()} invalid {ijet}j events")
                valid_events_dict[ijet] = valid_events_dict[ijet][~invalid_mask]

        return valid_events_dict, prob_stop_dict

    def KL(self):
        return self.cformer.KL() + self.pformer.KL()

    def reset_BNN(self):
        self.pformer.reset_BNN()
        self.cformer.reset_BNN()

    def extra_plots(self, filename):
        pass

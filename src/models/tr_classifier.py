import torch
from torch import nn

from hydra.utils import instantiate

from src.utils.logger import LOGGER
from src.models.base_classifier import BaseClassifier
from src.utils.preprocessing_cls import preprocess_tr


class TransformerClassifier(BaseClassifier):
    def __init__(self, label, cfg, device, dtype):
        super().__init__(label, cfg, device, dtype)

    def init_model(self):
        self.num_particles = [
            self.cfg.data.n_hard_particles + i for i in self.cfg.data.n_jets_list
        ]
        self.num_pairs = [i * (i - 1) // 2 for i in self.num_particles]
        if self.cfg.prep_cls.virtual.use:
            # add tokens for virtual particles
            extra_particles = (
                len(self.cfg.data.virtual_components)
                if self.cfg.prep_cls.virtual.components is None
                else len(self.cfg.prep_cls.virtual.components)
            )
            self.num_particles = [i + extra_particles for i in self.num_particles]
        else:
            self.num_particles = self.num_particles

        self.model = instantiate(
            self.cfg.classifier.net,
            in_channels=4 + max(self.num_particles),
            out_channels=1,
            is_causal=False,
        )

        # print information
        num_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        LOGGER.info(
            f"Created Transformer Classifier {self.label} with num_particles={max(self.num_particles)}, "
            f"{num_parameters} parameters"
        )
        self.model.to(self.device, dtype=self.dtype)

    def _preprocess(self, data, prep_params=None):
        return preprocess_tr(data, self.cfg, prep_params)

    def _batch_loss(self, data):
        loss = 0.0
        bces = []
        for ijet, batch in enumerate(data):
            data_true, data_fake = batch
            data_true, data_fake = data_true.to(
                self.device, dtype=self.dtype
            ), data_fake.to(self.device, dtype=self.dtype)
            data_combined = torch.cat([data_true, data_fake], dim=0)
            labels = torch.cat(
                [
                    torch.ones_like(data_true[:, [0]]),
                    torch.zeros_like(data_fake[:, [0]]),
                ],
                dim=0,
            )
            x = self._embed(data_combined, ijet)
            logits = self.model(x)
            logits = self._extract(logits)
            bce = self.loss(logits, labels)
            loss += bce / len(data)
            bces.append(bce)
        assert torch.isfinite(loss)

        metrics = {
            f"bce.{n_jets}j": bces[ijet].detach().cpu()
            for ijet, n_jets in enumerate(self.cfg.data.n_jets_list)
        }
        return loss, metrics

    def _encode_tokens(self, num_particles, num_pairs, global_token, batchsize):
        # create type_token_particles of shape (batchsize, num_tokens, max(self.num_particles))
        type_token_particles_raw = torch.arange(num_particles, device=self.device)
        type_token_particles = nn.functional.one_hot(
            type_token_particles_raw, num_classes=max(self.num_particles)
        )
        type_token_particles = (
            type_token_particles.unsqueeze(0)
            .expand(batchsize, *type_token_particles.shape)
            .float()
        )

        # create type_token_pairs of shape (batchsize, num_tokens, max(self.num_particles))
        type_token_pairs = torch.zeros(
            batchsize, num_pairs, max(self.num_particles), device=self.device
        )
        num_particles_reduced = int(
            (1 + (1 + 8 * num_pairs) ** 0.5) / 2
        )  # number of particles, but not including virtual particles
        idxs = [
            (idx1, idx2)
            for idx1 in range(num_particles_reduced)
            for idx2 in range(num_particles_reduced)
            if idx1 < idx2
        ]  # index pairs (in the same order as in preprocessing_cls)
        for i, (idx1, idx2) in enumerate(idxs):
            # the type token for the pairs of particles i and j
            # has 1's at the positions i and j and 0's otherwise
            type_token_pairs[:, i, idx1] = 1
            type_token_pairs[:, i, idx2] = 1

        # create global_token of shape (batchsize, 1, max(self.num_particles)+4)
        global_token_raw = torch.tensor(global_token, device=self.device)
        global_token = nn.functional.one_hot(
            global_token_raw, num_classes=max(self.num_particles) + 4
        )
        global_token = (
            global_token.unsqueeze(0)
            .expand(batchsize, *global_token.shape)
            .float()
            .unsqueeze(1)
        )

        return type_token_particles, type_token_pairs, global_token

    def _embed(self, x, ijet):
        # shape of original x: (batchsize, num_tokens, 4)
        # shape of embedded x: (batchsize, 1+num_particles+num_pairs, 4+max(self.num_particles))
        # - prepended one token which global information (which kind of process are we looking at)
        #   and contains the classifier score (also a global object) in the end
        # - appended max(self.num_particles) objects which is one-hot-encoded information about the particle type
        #   for both preprocessed particles and pairs
        num_particles = self.num_particles[ijet]
        num_pairs = self.num_pairs[ijet]

        particles = x[..., : 4 * num_particles]
        particles = particles.reshape(particles.shape[0], particles.shape[1] // 4, 4)

        pairs = x[..., 4 * num_particles :]
        temp = pairs.reshape(pairs.shape[0], num_pairs, -1)
        pairs = torch.zeros(
            pairs.shape[0], num_pairs, 4, dtype=pairs.dtype, device=pairs.device
        )
        pairs[..., : temp.shape[-1]] = temp
        type_token_particles, type_token_pairs, global_token = self._encode_tokens(
            num_particles, num_pairs, global_token=ijet, batchsize=x.shape[0]
        )

        particles = torch.cat((particles, type_token_particles), dim=2)
        pairs = torch.cat((pairs, type_token_pairs), dim=2)
        x = torch.cat((global_token, particles, pairs), dim=1)
        return x

    def _extract(self, x):
        # extract the global token
        return x[:, 0, :]

    def _evaluate(self, data, ijet):
        data = data.to(self.device, dtype=self.dtype)
        x = self._embed(data, ijet)
        logit = self.model(x)
        logit = self._extract(logit)
        return logit

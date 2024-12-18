import torch
from torch import nn
from src.models.base_classifier import BaseClassifier

from hydra.utils import instantiate
from src.utils.logger import LOGGER

from src.utils.preprocessing_cls import preprocess_mlp


class MLPClassifier(BaseClassifier):
    def __init__(self, label, cfg, device, dtype):
        super().__init__(label, cfg, device, dtype)

    def init_model(self):
        models = []
        for ijet in range(len(self.cfg.data.n_jets_list)):
            # dynamically set number of inputs based on data
            in_channels = self.prepd_true[ijet]["trn"].shape[1]

            model = instantiate(self.cfg.classifier.net, in_channels=in_channels)
            models.append(model)

            # print information
            num_parameters = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            LOGGER.info(
                f"Created MLP {self.label} for {self.cfg.data.n_jets_list[ijet]}j "
                f"data with in_channels={in_channels} and {num_parameters} parameters"
            )
        self.model = nn.ModuleList(models)
        self.model.to(self.device, dtype=self.dtype)

    def _preprocess(self, data, prep_params=None):
        return preprocess_mlp(data, self.cfg, prep_params)

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
            logits = self.model[ijet](data_combined)
            bce = self.loss(logits, labels)
            loss += bce / len(data)
            bces.append(bce)
        assert torch.isfinite(loss)

        metrics = {
            f"bce.{n_jets}j": bces[ijet].detach().cpu()
            for ijet, n_jets in enumerate(self.cfg.data.n_jets_list)
        }
        return loss, metrics

    def _evaluate(self, data, ijet):
        data = data.to(self.device, dtype=self.dtype)
        logit = self.model[ijet](data)
        return logit

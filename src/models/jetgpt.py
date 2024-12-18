import torch
import torch.nn.functional as F

from hydra.utils import instantiate

from src.models.base_generator import BaseGenerator
from src.utils.logger import LOGGER
import src.networks
from src.utils.distributions import GaussianMixtureModel


class JetGPT(BaseGenerator):
    def __init__(self, cfg, warm_start, device, dtype):
        super().__init__(cfg, warm_start, device, dtype)

    def init_model(self):
        # initialize GMM
        self.n_gauss = (
            self.cfg.generator.net.hidden_channels // 3
            if self.cfg.generator.n_gauss is None
            else self.cfg.generator.n_gauss
        )
        self.distribution = GaussianMixtureModel(self.n_gauss)

        # note: could be more efficient here,
        # but the network will just learn to ignore the redundant inputs
        self.channel_max = max(self.cfg.data.channels) + 1
        self.ijet_max = len(self.cfg.data.n_jets_list)

        # initialize network
        n_parameters = 3 * self.n_gauss
        self.model = instantiate(
            self.cfg.generator.net,
            in_channels=1 + (self.channel_max + 1) + self.ijet_max,
            out_channels=3 * self.n_gauss,
            is_causal=True,
        )
        LOGGER.info(f"Created JetGPT with n_gauss={self.n_gauss}")

    def forward(self, x, channels, ijet):
        x_embedded = self._embed(x, channels, ijet)
        logits = self.model(x_embedded)
        return logits

    def log_prob(self, x, context):
        x = x.clone()
        x = x.to(self.device, dtype=self.dtype)
        channels = torch.tensor(context["channels"], dtype=torch.long, device=x.device)
        ijet = torch.tensor(context["ijet"], device=x.device, dtype=torch.long)

        # append x0=0 in beginning
        x = torch.cat(
            (torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype), x), dim=1
        )
        channels = torch.cat(
            (
                self.channel_max * torch.ones(1, device=x.device, dtype=torch.long),
                channels,
            )
        )

        # slice
        inputs = x[:, :-1]
        targets = x[:, 1:]
        channels = channels[1:]  # channels are input

        # pass to network
        logits = self.forward(inputs, channels, ijet)
        log_prob_single = self.distribution.log_prob(targets, logits)
        log_prob = log_prob_single.sum(dim=-1)
        return log_prob

    def sample_single(self, nsamples, context):
        nchannels = len(context["channels"])
        samples_prepd = []
        n_batches = int(nsamples / self.cfg.sampling.batchsize) + 1

        for _ in range(n_batches):
            x = torch.zeros(
                self.cfg.sampling.batchsize, 1, device=self.device, dtype=self.dtype
            )
            channels = torch.tensor(
                context["channels"], device=x.device, dtype=torch.long
            )
            channels = torch.cat(
                (
                    self.channel_max
                    * torch.ones(1, device=channels.device, dtype=torch.long),
                    channels,
                )
            )
            ijet = torch.tensor(context["ijet"], device=x.device, dtype=torch.long)

            for ichannel in range(1, nchannels + 1):
                channels_now = channels[1 : ichannel + 1]
                with torch.no_grad():
                    logits = self.forward(x, channels_now, ijet)
                x_now = self.distribution.sample(logits[:, [-1], :])

                x = torch.cat((x, x_now), dim=-1)
            x = x[:, 1:]
            samples_prepd.append(x)

        samples_prepd = torch.cat(samples_prepd, dim=0)[:nsamples, :].cpu().numpy()
        return samples_prepd

    def _embed(self, x, channels, ijet):
        x = x.unsqueeze(-1)

        # one-hot-encode channel and ijet
        channels_embedding = F.one_hot(channels, num_classes=self.channel_max + 1)
        ijet_embedding = F.one_hot(ijet, num_classes=self.ijet_max)

        # reshape
        channels_embedding = channels_embedding.expand(x.shape[0], x.shape[1], -1)
        ijet_embedding = ijet_embedding.expand(x.shape[0], 1, -1).expand(
            x.shape[0], x.shape[1], -1
        )

        x = torch.cat((x, channels_embedding, ijet_embedding), dim=-1)
        return x

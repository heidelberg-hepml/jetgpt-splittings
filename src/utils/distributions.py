import torch
import torch.distributions as D
import torch.nn.functional as F


class GaussianMixtureModel:
    def __init__(self, n_gauss):
        self.n_gauss = n_gauss

    def _extract_logits(self, logits, min_sigmaarg=-20, max_sigmaarg=10):
        logits = logits.reshape(logits.size(0), logits.size(1), self.n_gauss, 3)

        weights = F.softmax(logits[:, :, :, 2], dim=-1)
        mu = logits[:, :, :, 0]

        # avoid inf and 0 (both unstable in D.Normal)
        sigmaarg = torch.clamp(logits[:, :, :, 1], min=min_sigmaarg, max=max_sigmaarg)
        sigma = torch.exp(sigmaarg)
        assert torch.isfinite(sigma).all()

        return mu, sigma, weights

    def build_gmm(self, logits):
        mu, sigma, weights = self._extract_logits(logits)
        mix = D.Categorical(weights)
        comp = D.Normal(mu, sigma)
        gmm = D.MixtureSameFamily(mix, comp)
        return gmm

    def log_prob(self, x, logits):
        gmm = self.build_gmm(logits)
        return gmm.log_prob(x)

    def sample(self, logits):
        gmm = self.build_gmm(logits)
        return gmm.sample((1,))[:, :, 0].permute(1, 0)

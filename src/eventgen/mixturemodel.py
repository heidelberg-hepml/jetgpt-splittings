import torch
import torch.distributions as D
import torch.nn.functional as F


class CustomMixtureModel:
    # Mixture model with vonMises distribution for angular variables and Normal distribution for others
    # Note that we could also preprocess angles to real space, but then the model would not be periodic
    def __init__(self, n_gauss):
        self.n_gauss = n_gauss

    def _extract_logits(self, logits, min_sigmaarg=-5, max_sigmaarg=5):
        logits = logits.view(*logits.shape[:-1], self.n_gauss, 3)

        weights = F.softmax(logits[..., 2], dim=-1)
        mu = logits[..., 0]

        # avoid inf and 0 (unstable in D.Normal and D.VonMises)
        sigmaarg = torch.clamp(logits[..., 1], min=min_sigmaarg, max=max_sigmaarg)
        sigma = torch.exp(sigmaarg)
        # assert torch.isfinite(sigma).all()

        return mu, sigma, weights

    def build_mm(self, logits):
        mu, sigma, weights = self._extract_logits(logits)
        mix = D.Categorical(probs=weights, validate_args=False)
        normal = D.Normal(mu, sigma, validate_args=False)
        gmm = D.MixtureSameFamily(mix, normal)
        vonmises = D.VonMises(mu, sigma, validate_args=False)
        vmmm = D.MixtureSameFamily(mix, vonmises)
        return gmm, vmmm

    def log_prob(self, x, logits, is_angle):
        assert x.shape == logits.shape[:-1] and logits.shape[-1] % 3 == 0
        gmm, vmmm = self.build_mm(logits)
        is_angle = is_angle[None, None, :].repeat(x.shape[0], x.shape[1], 1)
        log_prob = torch.where(is_angle, vmmm.log_prob(x), gmm.log_prob(x))
        return log_prob

    def sample(self, logits, is_angle):
        gmm, vmmm = self.build_mm(logits)
        mm = vmmm if is_angle else gmm
        x = mm.sample((1,))[0]
        return x

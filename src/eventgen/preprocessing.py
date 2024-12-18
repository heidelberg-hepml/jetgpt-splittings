import torch

from src.eventgen.helpers import (
    EPS1,
    PtPhiEtaM2_to_EPPP,
    EPPP_to_PtPhiEtaM2,
    ensure_angle,
)


class StandardPreprocessing:
    def __init__(
        self, pt_min, units, onshell_list, device, dtype=torch.float64, **kwargs
    ):
        self.dtype, self.device = dtype, device
        self.pt_min = torch.tensor(pt_min, device=device, dtype=dtype)
        self.units = units
        self.onshell_list = onshell_list
        self.params = None

    def preprocess(self, fourmomenta_in, pidx, standardize=True):
        fourmomenta = fourmomenta_in.clone().to(self.device, self.dtype)
        fourmomenta /= self.units
        x = EPPP_to_PtPhiEtaM2(fourmomenta)
        x[..., 0] = torch.log(x[..., 0] + EPS1 - self.pt_min[pidx])
        x[..., 3] = torch.log(x[..., 3].clamp(min=0) + EPS1)
        if standardize:
            assert self.params is not None
            x = (x - self.params["mean"][pidx]) / self.params["std"][pidx]

        return x.to(fourmomenta_in.device, fourmomenta_in.dtype)

    def undo_preprocess(self, x_in, pidx, standardize=True):
        x = x_in.clone().to(self.device, self.dtype)

        if standardize:
            assert self.params is not None
            x = self.params["std"][pidx] * x + self.params["mean"][pidx]

        x[..., 0] = torch.exp(x[..., 0]) - EPS1 + self.pt_min[pidx]
        x[..., 3] = torch.exp(x[..., 3]) - EPS1

        fourmomenta = PtPhiEtaM2_to_EPPP(x)
        fourmomenta *= self.units

        return fourmomenta.to(x_in.device, x_in.dtype)

    def initialize_parameters(self, fourmomenta, idxs):
        max_type = max([idx.max() for idx in idxs]) + 1
        collections = {ijet: [] for ijet in range(max_type)}
        for ijet in range(len(fourmomenta)):
            fm, idx = fourmomenta[ijet], idxs[ijet]
            for itype in range(max_type):
                mask = idx == itype
                x = self.preprocess(fm[:, mask], idx[mask], standardize=False)
                x = x.reshape(-1, 4)
                collections[itype].append(x)
        collections = {itype: torch.cat(x) for itype, x in collections.items()}

        self.params = {
            "mean": torch.zeros(
                (max_type, 4), dtype=collections[0].dtype, device=collections[0].device
            ),
            "std": torch.ones(
                (max_type, 4), dtype=collections[0].dtype, device=collections[0].device
            ),
        }
        for itype, x in collections.items():
            mask = [0, 2, 3]
            self.params["mean"][itype][mask] = x.mean(dim=0)[mask]
            self.params["std"][itype][mask] = x.std(dim=0).clamp(min=1e-3)[mask]
        self.params = {key: value.to(self.device) for key, value in self.params.items()}


class PtRatioPreprocessing(StandardPreprocessing):
    def __init__(self, *args, n_hard_particles=0, **kwargs):
        self.n_hard_particles = n_hard_particles
        super().__init__(*args, **kwargs)

    def preprocess(self, fourmomenta_in, pidx, standardize=True):

        fourmomenta = fourmomenta_in.clone().to(self.device, self.dtype)
        fourmomenta /= self.units
        x = EPPP_to_PtPhiEtaM2(fourmomenta)

        # enforce pT cut
        x[..., 0] -= self.pt_min[pidx] - EPS1

        i = self.n_hard_particles
        # Express jet pTs as ratio to 'previous' jet, starting from n_hard_particles
        x[:, i + 1 :, 0] = x[:, i + 1 :, 0] / x[:, i:-1, 0]
        # logit pt ratios and log-scale pts
        x[:, : i + 1, 0] = torch.log(x[:, : i + 1, 0])
        x[:, i + 1 :, 0] = torch.logit(x[:, i + 1 :, 0] * (1 - 2 * EPS1) + EPS1)

        # log-scale masses
        x[..., 3] = torch.log(x[..., 3].clamp(min=0) + EPS1)

        if standardize:
            assert self.params is not None
            x = (x - self.params["mean"][pidx]) / self.params["std"][pidx]

        return x.to(fourmomenta_in.device, fourmomenta_in.dtype)

    def undo_preprocess(self, x_in, pidx, standardize=True):
        x = x_in.clone().to(self.device, self.dtype)

        if standardize:
            assert self.params is not None
            x = self.params["std"][pidx] * x + self.params["mean"][pidx]

        i = self.n_hard_particles
        x[:, : i + 1, 0] = torch.exp(x[:, : i + 1, 0])
        x[:, i + 1 :, 0] = (torch.sigmoid(x[:, i + 1 :, 0]) - EPS1) / (1 - 2 * EPS1)

        x[..., 3] = torch.exp(x[..., 3]) - EPS1

        # Recover physical jet pTs from relative ratios.
        for j in range(i + 1, x.shape[-2]):
            x[:, j, 0] *= x[:, j - 1, 0]

        # restore pT cut
        x[..., 0] += self.pt_min[pidx] - EPS1

        fourmomenta = PtPhiEtaM2_to_EPPP(x)
        fourmomenta *= self.units

        return fourmomenta.to(x_in.device, x_in.dtype)


class NormalizedPreprocessing(StandardPreprocessing):
    def __init__(
        self, pt_min, units, onshell_list, device, dtype=torch.float64, **kwargs
    ):
        self.dtype, self.device = dtype, device
        self.pt_min = torch.tensor(pt_min, device=device, dtype=dtype)
        self.units = units
        self.onshell_list = onshell_list
        self.params = None

    def preprocess(self, fourmomenta_in, pidx, standardize=True):
        pidx = torch.arange(fourmomenta_in.size(1), device=self.device)
        return super().preprocess(fourmomenta_in, pidx, standardize)

    def undo_preprocess(self, x_in, pidx, standardize=True):
        pidx = torch.arange(x_in.size(1), device=self.device)
        return super().undo_preprocess(x_in, pidx, standardize)

    def initialize_parameters(self, fourmomenta, idxs):

        dtype = fourmomenta[0].dtype
        device = fourmomenta[0].device
        max_type = len(self.pt_min)
        self.params = {
            "mean": torch.zeros((max_type, 4), dtype=dtype, device=device),
            "std": torch.zeros((max_type, 4), dtype=dtype, device=device),
            "norm": torch.zeros((max_type, 4), dtype=dtype, device=device),
        }

        for ijet in range(len(fourmomenta)):

            fm = fourmomenta[ijet]
            pidx = torch.arange(fm.size(1), device=device)
            x = self.preprocess(fm, pidx, standardize=False)

            N = x.size(1)
            self.params["mean"][:N] += x.mean(dim=0)
            self.params["std"][:N] += x.std(dim=0)
            self.params["norm"][:N] += 1.0

        self.params["mean"] /= self.params["norm"]
        self.params["std"] /= self.params["norm"]

        # don't change phi
        self.params["mean"][:, 1] = 0.0
        self.params["std"][:, 1] = 1.0

        self.params = {key: value.to(self.device) for key, value in self.params.items()}


class PtRatioNormPreprocessing(PtRatioPreprocessing):
    def initialize_parameters(self, fourmomenta, idxs):

        dtype = fourmomenta[0].dtype
        device = fourmomenta[0].device
        max_type = len(self.pt_min)
        self.params = {
            "mean": torch.zeros((max_type, 4), dtype=dtype, device=device),
            "std": torch.zeros((max_type, 4), dtype=dtype, device=device),
            "norm": torch.zeros((max_type, 4), dtype=dtype, device=device),
        }

        for ijet in range(len(fourmomenta)):

            fm = fourmomenta[ijet]
            pidx = torch.arange(fm.size(1), device=device)
            x = self.preprocess(fm, pidx, standardize=False)

            N = x.size(1)
            self.params["mean"][:N] += x.mean(dim=0)
            self.params["std"][:N] += x.std(dim=0)
            self.params["norm"][:N] += 1.0

        self.params["mean"] /= self.params["norm"]
        self.params["std"] /= self.params["norm"]

        # don't change phi
        self.params["mean"][:, 1] = 0.0
        self.params["std"][:, 1] = 1.0

        self.params = {key: value.to(self.device) for key, value in self.params.items()}


class RelPhiPreprocessing(NormalizedPreprocessing):
    def __init__(
        self, pt_min, units, onshell_list, device, dtype=torch.float64, **kwargs
    ):
        self.dtype, self.device = dtype, device
        self.pt_min = torch.tensor(pt_min, device=device, dtype=dtype)
        self.units = units
        self.onshell_list = onshell_list
        self.params = None

    def preprocess(self, fourmomenta_in, pidx, standardize=True):
        fourmomenta = fourmomenta_in.clone().to(self.device, self.dtype)
        fourmomenta /= self.units
        x = EPPP_to_PtPhiEtaM2(fourmomenta)
        x[..., 0] = torch.log(x[..., 0] + EPS1 - self.pt_min[pidx])
        x[..., 3] = torch.log(x[..., 3].clamp(min=0) + EPS1)

        # make phi rel
        x[..., 1:, 1] = ensure_angle(x[..., 1:, 1] - x[..., [0], 1])

        if standardize:
            assert self.params is not None
            x = (x - self.params["mean"][pidx]) / self.params["std"][pidx]

        return x.to(fourmomenta_in.device, fourmomenta_in.dtype)

    def undo_preprocess(self, x_in, pidx, standardize=True):
        x = x_in.clone().to(self.device, self.dtype)

        if standardize:
            assert self.params is not None
            x = self.params["std"][pidx] * x + self.params["mean"][pidx]

        # undo phi rel
        x[..., 1:, 1] = ensure_angle(x[..., 1:, 1] + x[..., [0], 1])

        x[..., 0] = torch.exp(x[..., 0]) - EPS1 + self.pt_min[pidx]
        x[..., 3] = torch.exp(x[..., 3]) - EPS1

        fourmomenta = PtPhiEtaM2_to_EPPP(x)
        fourmomenta *= self.units

        return fourmomenta.to(x_in.device, x_in.dtype)

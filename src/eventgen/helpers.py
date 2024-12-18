import math
import torch

# log(x) -> log(x+EPS1)
# in (invertible) preprocessing functions to avoid being close to log(0)
EPS1 = 1e-5

# generic numerical stability cutoff
EPS2 = 1e-10

# exp(x) -> exp(x.clamp(max=CUTOFF))
CUTOFF = 10

# these functions are only used for plotting,
# with the exception of delta_r_fast


def unpack_last(x):
    # unpack along the last dimension
    n = len(x.shape)
    return torch.permute(x, (n - 1, *list(range(n - 1))))


def EPPP_to_PtPhiEtaM2(fourmomenta, sqrt_mass=False):
    E, px, py, pz = unpack_last(fourmomenta)
    pt = torch.sqrt(px**2 + py**2)
    phi = torch.arctan2(py, px)
    p_abs = torch.sqrt(pz**2 + pt**2)
    eta = stable_arctanh(pz / p_abs).clamp(min=-CUTOFF, max=CUTOFF)
    m2 = E**2 - px**2 - py**2 - pz**2
    m2 = torch.sqrt(m2.clamp(min=EPS2)) if sqrt_mass else m2
    return torch.stack((pt, phi, eta, m2), dim=-1)


def PtPhiEtaM2_to_EPPP(x):
    pt, phi, eta, m2 = unpack_last(x)
    eta = eta.clamp(min=-CUTOFF, max=CUTOFF)
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    E = torch.sqrt(m2 + pt**2 * torch.cosh(eta) ** 2)
    return torch.stack((E, px, py, pz), dim=-1)


def stay_positive(x):
    # flip sign for entries with x<0 such that always x>0
    x = torch.where(x >= 0, x, -x)
    return x


def stable_arctanh(x, eps=EPS2):
    # implementation of arctanh that avoids log(0) issues
    return 0.5 * (torch.log((1 + x).clamp(min=eps)) - torch.log((1 - x).clamp(min=eps)))


def ensure_angle(phi):
    return (phi + math.pi) % (2 * math.pi) - math.pi


def delta_phi(jetmomenta, idx1, idx2, abs=False):
    dphi = jetmomenta[..., idx1, 1] - jetmomenta[..., idx2, 1]
    dphi = ensure_angle(dphi)
    return torch.abs(dphi) if abs else dphi


def delta_eta(jetmomenta, idx1, idx2, abs=False):
    deta = jetmomenta[..., idx1, 2] - jetmomenta[..., idx2, 2]
    return torch.abs(deta) if abs else deta


def delta_r(jetmomenta, idx1, idx2):
    return (
        delta_phi(jetmomenta, idx1, idx2) ** 2 + delta_eta(jetmomenta, idx1, idx2) ** 2
    ) ** 0.5


def get_virtual_particle(jetmomenta, components):
    fourmomenta = PtPhiEtaM2_to_EPPP(jetmomenta)

    particle = fourmomenta[..., components, :].sum(dim=-2)
    particle = EPPP_to_PtPhiEtaM2(particle, sqrt_mass=True)
    return particle


def ensure_onshell(fourmomenta, onshell_list, onshell_mass):
    onshell_mass = torch.tensor(
        onshell_mass, device=fourmomenta.device, dtype=fourmomenta.dtype
    )
    onshell_mass = onshell_mass.unsqueeze(0).expand(
        fourmomenta.shape[0], onshell_mass.shape[-1]
    )
    fourmomenta[..., onshell_list, 0] = torch.sqrt(
        onshell_mass**2 + torch.sum(fourmomenta[..., onshell_list, 1:] ** 2, dim=-1)
    )
    return fourmomenta


def enforce_pt_ordering(event, n_hard_particles, exyz=True):
    """Enforce pt ordering of extra jets within a set of events."""
    hard_process = event[:, :n_hard_particles]
    extra_jets = event[:, n_hard_particles:]
    if exyz:
        pts = (extra_jets[..., [1]] ** 2 + extra_jets[..., [2]] ** 2).sqrt()
    else:
        pts = extra_jets[..., [0]]
    sort_idx = torch.argsort(-pts, dim=1)
    extra_jets = extra_jets.take_along_dim(sort_idx, dim=1)
    event = torch.cat([hard_process, extra_jets], dim=1)
    return event


def ensure_angle(phi):
    return (phi + torch.pi) % (2 * torch.pi) - torch.pi

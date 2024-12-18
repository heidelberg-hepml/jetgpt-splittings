import numpy as np

import matplotlib.pyplot as plt  # debugging


def fourmomenta_to_jetmomenta(fourmomenta):
    pt = get_pt(fourmomenta)
    phi = get_phi(fourmomenta)
    eta = get_eta(fourmomenta)
    mass = get_mass(fourmomenta)

    jetmomenta = np.stack((pt, phi, eta, mass), axis=-1)
    assert np.isfinite(jetmomenta).all()
    return jetmomenta


def jetmomenta_to_fourmomenta(jetmomenta, cutoff=10):
    pt, phi, eta, mass = jetmomenta.transpose(2, 0, 1)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(np.clip(eta, -cutoff, cutoff))
    E = np.sqrt(mass**2 + px**2 + py**2 + pz**2)

    fourmomenta = np.stack((E, px, py, pz), axis=-1)
    assert np.isfinite(fourmomenta).all()
    return fourmomenta


def get_pt(particle):
    return np.sqrt(particle[..., 1] ** 2 + particle[..., 2] ** 2)


def get_phi(particle):
    return np.arctan2(particle[..., 2], particle[..., 1])


def get_eta(particle, eps=1e-10):
    # eta = np.arctanh(particle[...,3] / p_abs) # numerically unstable
    p_abs = np.sqrt(np.sum(particle[..., 1:] ** 2, axis=-1))
    eta = 0.5 * (
        np.log(np.clip(np.abs(p_abs + particle[..., 3]), eps, None))
        - np.log(np.clip(np.abs(p_abs - particle[..., 3]), eps, None))
    )
    return eta


def get_mass(particle, eps=1e-6):
    return np.sqrt(
        np.clip(
            particle[..., 0] ** 2 - np.sum(particle[..., 1:] ** 2, axis=-1), eps, None
        )
    )


def ensure_angle(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi


def delta_phi(event, idx1, idx2, abs=False):
    dphi = event[..., idx1, 1] - event[..., idx2, 1]
    dphi = ensure_angle(dphi)
    return np.abs(dphi) if abs else dphi


def delta_eta(event, idx1, idx2, abs=False):
    deta = event[..., idx1, 2] - event[..., idx2, 2]
    return np.abs(deta) if abs else deta


def delta_r(event, idx1, idx2):
    return (
        delta_phi(event, idx1, idx2) ** 2 + delta_eta(event, idx1, idx2) ** 2
    ) ** 0.5


def get_virtual_particle(event, components):
    jetmomenta = event.copy()
    fourmomenta = jetmomenta_to_fourmomenta(jetmomenta)

    particle = fourmomenta[..., components, :].sum(axis=-2)
    particle = fourmomenta_to_jetmomenta(particle)
    return particle


def make_phi_rel(event):
    event[..., 1:, 1] = ensure_angle(event[..., 1:, 1] - event[..., [0], 1])
    return event


def undo_phi_rel(event):
    event[..., 1:, 1] = ensure_angle(event[..., 1:, 1] + event[..., [0], 1])
    return event


def make_eta_rel(event):
    event[..., 1:, 2] = event[..., 1:, 2] - event[..., [0], 2]
    return event


def undo_eta_rel(event):
    event[..., 1:, 2] = event[..., 1:, 2] + event[..., [0], 2]
    return event


def lorentz_boost(p_target, p_frame_particle):
    # applies a lorentz boost to p_target, whose velocity is inferred
    # from p_frame_particle, such that the new frame becomes the rest
    # frame of the particle with momentum p_frame_particle;
    # note that this algorithm is numerically unstable to a significant
    # extend for highly relativistic events
    K = np.array(
        [
            [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]],
        ]
    )

    beta = p_frame_particle[..., 1:] / p_frame_particle[..., 0:1]  # p/E

    # don't boost momenta when norm(beta) is zero
    boost_mask = np.linalg.norm(beta, axis=-1) != 0.0
    n = beta[boost_mask] / np.linalg.norm(
        beta[boost_mask], axis=-1, keepdims=True
    )  # unit vectors in boost direction
    rapidity = np.arctanh(np.sqrt(np.sum(beta[boost_mask] ** 2, axis=-1)))

    n = np.expand_dims(n, axis=(-1, -2))
    rapidity = np.expand_dims(rapidity, axis=(-1, -2))

    # boost direction
    n_dot_K = 0
    for i in range(len(K)):  # could parallelize this with torch.einsum
        n_dot_K += n[..., i, :, :] * K[i]

    # transformation matrix (Lambda)
    B = (
        np.eye(4, 4)
        - np.sinh(rapidity) * n_dot_K
        + (np.cosh(rapidity) - 1) * (n_dot_K @ n_dot_K)
    )

    p_boosted = np.full_like(p_target, np.nan)
    p_boosted[boost_mask] = (B @ np.expand_dims(p_target[boost_mask], axis=-1)).squeeze(
        -1
    )
    p_boosted[~boost_mask] = p_target[~boost_mask]
    return p_boosted


def stable_arctanh(x, eps):
    # numerically stable implementation of arctanh that avoids log(0) issues
    return 0.5 * (
        np.log(np.clip(1 + x, a_min=eps, a_max=None))
        - np.log(np.clip(1 - x, a_min=eps, a_max=None))
    )

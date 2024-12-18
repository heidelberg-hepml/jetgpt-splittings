import numpy as np
import src.utils.physics as phys
from src.utils.physics import stable_arctanh

import matplotlib.pyplot as plt  # debugging

# for numerical stability
EPS = 1e-3

# might have to mix channels (eg turn mass into pt) when doing the virtual preprocessing step
BOOST_CHANNELS_MIXER = {"zmumu": {4: 3}, "ttbar": {}}  # pT_l2 -> m1 for zmumu


def preprocess(event, cfg, prep_params):
    event = event.copy()
    if prep_params is None:
        prep_params = {}
    channels = cfg.data.channels
    if cfg.prep_gen.virtual:
        channels = [BOOST_CHANNELS_MIXER[cfg.exp_type].get(i, i) for i in channels]

    # boost
    if cfg.prep_gen.virtual:
        if cfg.exp_type == "zmumu":
            event = _virtual_zmumu(event)
        else:
            raise ValueError(
                f"boost preprocessing not implemented for exp_type={cfg.exp_type}"
            )
    assert (
        (1 in cfg.data.channels or not cfg.prep_gen.phi_rel)
        and (2 in cfg.data.channels or not cfg.prep_gen.eta_rel)
    ) or not cfg.prep_gen.virtual, (
        "This collection of settings is not correctly implemented: "
        "Model predicts angles relative to phi_Z (hence phi_Z should be sampled uniformly, "
        "but phi_l1 is sampled uniformly."
    )

    # make angles relative
    assert (
        1 in cfg.data.channels or cfg.prep_gen.phi_rel
    ), "phi_l1 either has to be given explicitly or determined with phi_rel"
    assert (
        2 in cfg.data.channels or cfg.prep_gen.eta_rel
    ), "eta_l1 either has to be given explicitly or determined with eta_rel"
    if cfg.prep_gen.phi_rel:
        event = phys.make_phi_rel(event)
    if cfg.prep_gen.eta_rel:
        event = phys.make_eta_rel(event)

    # transformations to make the distributions more gaussian
    if cfg.prep_gen.gaussianize:
        for i, pt_cut in enumerate(cfg.prep_gen.pt_cut):
            if i >= event.shape[1]:
                break
            event[..., i, 0] -= pt_cut - EPS
        event[..., 0] = np.log(event[..., 0])
        event[..., 1] = stable_arctanh(event[..., 1] / np.pi, EPS)
        event[..., 3] = np.log(event[..., 3])

    # discard channels
    event = event.reshape(event.shape[0], event.shape[1] * event.shape[2])
    event = event[:, channels]

    # standardize to N(0,1)
    if cfg.prep_gen.standardize:
        mean, std = prep_params.get("mean", None), prep_params.get("std", None)
        if mean is None or std is None:
            mean, std = event.mean(axis=0), event.std(axis=0)
            prep_params["mean"], prep_params["std"] = mean, std
        event = (event - mean) / std

    # arcsinh trick
    if len(cfg.prep_gen.final_arcsinh) > 0:
        idx = cfg.prep_gen.final_arcsinh
        event[..., idx] = np.arcsinh(event[..., idx])

    assert np.isfinite(
        event
    ).all(), f"{np.isnan(event).sum(axis=0)} {np.isinf(event).sum(axis=0)}"
    return event, prep_params


def undo_preprocess(event, cfg, prep_params):
    # undo all operations in preproess()
    event = event.copy()
    channels = cfg.data.channels
    if cfg.prep_gen.virtual:
        channels = [BOOST_CHANNELS_MIXER[cfg.exp_type].get(i, i) for i in channels]

    if len(cfg.prep_gen.final_arcsinh) > 0:
        idx = cfg.prep_gen.final_arcsinh
        event[..., idx] = np.sinh(event[..., idx])

    if cfg.prep_gen.standardize:
        mean = prep_params.get("mean", None)
        std = prep_params.get("std", None)
        assert mean is not None and std is not None
        event = event * std + mean

    temp = event.copy()
    event = np.zeros(
        (event.shape[0], (cfg.data.n_hard_particles + cfg.data.n_jets_max) * 4)
    )
    event[..., channels] = temp
    event = event.reshape(event.shape[-2], event.shape[-1] // 4, 4)

    if cfg.prep_gen.gaussianize:
        event[..., 0] = np.exp(
            event[..., 0].clip(max=10)
        )  # clip to avoid numerical issues
        event[..., 1] = np.tanh(event[..., 1]) * np.pi
        event[..., 3] = np.exp(
            event[..., 3].clip(max=10)
        )  # clip to avoid numerical issues
        for i, pt_cut in enumerate(cfg.prep_gen.pt_cut):
            if i >= event.shape[1]:
                break
            event[..., i, 0] += pt_cut - EPS

    if not 1 in channels:
        event[..., 0, 1] = np.random.uniform(-np.pi, np.pi, size=event.shape[0])

    if cfg.prep_gen.phi_rel:
        event = phys.undo_phi_rel(event)
    if cfg.prep_gen.eta_rel:
        event = phys.undo_eta_rel(event)

    if cfg.prep_gen.virtual:
        if cfg.exp_type == "zmumu":
            event = _undo_virtual_zmumu(event)
        else:
            raise ValueError(
                f"boost preprocessing not implemented for exp_type={cfg.exp_type}"
            )

    # make sure the channels that anything in the channels that were not used is 0
    mask = np.zeros(4 * (cfg.data.n_hard_particles + cfg.data.n_jets_max), dtype=bool)
    channels_to_keep = [
        ch
        for ch in channels
        if ch < 4 * (cfg.data.n_hard_particles + max(cfg.data.n_jets_list))
    ]
    mask[channels_to_keep] = True
    mask = mask.reshape(-1, 4)
    # this was incorrectly implemented in the original code
    # event[..., ~mask] = 0.0

    assert np.isfinite(
        event
    ).all(), f"{np.isnan(event).sum(axis=0)} {np.isinf(event).sum(axis=0)}"
    return event


# helper functions


def _virtual_zmumu(event):
    jets = event[..., 2:, :]
    zsystem = event[..., :2, :]

    zsystem = phys.jetmomenta_to_fourmomenta(zsystem)
    p2 = zsystem[..., 1, :]
    pz = zsystem.sum(axis=-2)
    p2 = phys.lorentz_boost(p2, pz)

    zsystem = np.stack((pz, p2), axis=-2)
    zsystem = phys.fourmomenta_to_jetmomenta(zsystem)
    return np.concatenate((zsystem, jets), axis=-2)


def _undo_virtual_zmumu(event):
    jets = event[..., 2:, :]
    zsystem = event[..., :2, :]

    # reconstruct z system
    zsystem[..., 1, 3] = 1.0e-3  # muon mass vanishes
    zsystem[..., 1, 0] = (
        zsystem[..., 0, 3] / 2 / np.cosh(zsystem[..., 1, 2])
    )  # pt = |p| / cosh(eta) with |p| = mZ/2 in the Z rest frame

    zsystem = phys.jetmomenta_to_fourmomenta(zsystem)
    pz = zsystem[..., 0, :]
    p2 = zsystem[..., 1, :]
    p2 = phys.lorentz_boost(p2, _flip(pz))
    p1 = pz - p2

    zsystem = np.stack((p1, p2), axis=-2)
    zsystem = phys.fourmomenta_to_jetmomenta(zsystem)
    return np.concatenate((zsystem, jets), axis=-2)


def _flip(p):
    # helper for undoing lorentz boosts
    # flip the spatial components of the 4-momentum p
    p_flipped = np.empty_like(p)
    p_flipped[:, 0] = p[:, 0]
    p_flipped[:, 1:] = -p[:, 1:]
    return p_flipped

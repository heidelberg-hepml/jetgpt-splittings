import numpy as np
from src.utils.physics import (
    delta_r,
    delta_phi,
    delta_eta,
    get_virtual_particle,
    stable_arctanh,
)

from src.utils.preprocessing_gen import EPS
from src.utils.logger import LOGGER

EPS2 = 1e-5


def preprocess_mlp(event, cfg, prep_params=None):
    prepd = []
    if prep_params is None:
        prep_params = {}

    if cfg.prep_cls.naive.use:
        naive = _get_naive(event, cfg, for_mlp=True)
        prepd.append(naive)

    if cfg.prep_cls.virtual.use:
        virtual = _get_virtual(event, cfg)
        prepd.append(virtual)

    if cfg.prep_cls.delta_r.use:
        deltaR = _get_deltaR(event, cfg)
        prepd.append(deltaR)

    if cfg.prep_cls.delta_phi.use:
        deltaphi = _get_deltaphi(event, cfg)
        prepd.append(deltaphi)

    if cfg.prep_cls.delta_eta.use:
        deltaeta = _get_deltaeta(event, cfg)
        prepd.append(deltaeta)

    # combine
    prepd = np.concatenate(prepd, axis=1)

    # standardize
    if cfg.prep_cls.standardize:
        mean, std = prep_params.get("mean", None), prep_params.get("std", None)
        if mean is None or std is None:
            mean, std = prepd.mean(axis=0), prepd.std(axis=0)
            assert (std > 1e-10).all(), (
                f"WARNING: Get zero std (std={std}) in channels {np.where(std < 1e-10)}, "
                f"i.e. all inputs are the same. You probably want to remove these components"
            )
            prep_params["mean"], prep_params["std"] = mean, std
        prepd = (prepd - mean) / std

    assert np.isfinite(prepd).all()
    return prepd, prep_params


def preprocess_tr(event, cfg, prep_params=None):
    # Could improve this
    # (but make sure that particles are all preprocessed in the same way,
    # otherwise the transformer gets problems)

    prepd_particles = []
    prepd_pairs = []
    if prep_params is None:
        prep_params = {}

    if hasattr(cfg.prep_cls, "anja_test_Zm"):
        test_Zm = cfg.prep_cls.anja_test_Zm
    else:
        test_Zm = False
    if hasattr(cfg.prep_cls, "anja_test_deltaR"):
        test_deltaR = cfg.prep_cls.anja_test_deltaR
    else:
        test_deltaR = False

    if cfg.prep_cls.naive.use:
        naive = _get_naive(event, cfg, for_mlp=False)
        if test_Zm or test_deltaR:
            naive = np.zeros_like(naive)
        prepd_particles.append(naive)

    if cfg.prep_cls.virtual.use:
        virtual = _get_virtual(event, cfg)
        prepd_particles.append(virtual)

    if cfg.prep_cls.delta_r.use:
        deltaR = _get_deltaR(event, cfg)
        prepd_pairs.append(deltaR)

    if cfg.prep_cls.delta_phi.use:
        deltaphi = _get_deltaphi(event, cfg)
        prepd_pairs.append(deltaphi)

    if cfg.prep_cls.delta_eta.use:
        deltaeta = _get_deltaeta(event, cfg)
        prepd_pairs.append(deltaeta)

    # combine
    prepd_particles = np.concatenate(prepd_particles, axis=-1)
    prepd_particles = prepd_particles.reshape(
        prepd_particles.shape[0], prepd_particles.shape[1] // 4, 4
    )
    prepd_pairs = np.stack(prepd_pairs, axis=2)
    prepd = {"particles": prepd_particles, "pairs": prepd_pairs}

    # standardize
    if cfg.prep_cls.standardize:
        # iterate over particles and pairs seperately
        for key, x in prepd.items():
            mean, std = prep_params.get(f"{key}_mean", None), prep_params.get(
                f"{key}_std", None
            )
            if mean is None or std is None:
                # take mean and std over number of elements and number of tokens
                mean, std = x.mean(axis=(0)), x.std(axis=(0))
                # assert (std > 1e-10).all(), (
                #     f"WARNING: Get zero std (std={std}) in channels {np.where(std < 1e-10)}, "
                #     f"i.e. all inputs are the same. You probably want to remove these components"
                # )
                prep_params[f"{key}_mean"], prep_params[f"{key}_std"] = mean, std
            x = (x - mean) / (std + 1e-10)

            # flatten along token dimension
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
            prepd[key] = x

    prepd = np.concatenate([x for x in prepd.values()], axis=1)
    # assert np.isfinite(prepd).all(), f"{np.isnan(prepd).sum(axis=0)} {np.isinf(prepd).sum(axis=0)}"
    # drop infinite events and print how many they were
    if not np.isfinite(prepd).all():
        n_inf = np.sum(np.isinf(prepd).any(axis=1))
        LOGGER.info(f"Found {n_inf} infinite events")
        # set to 0 the infinite values
        prepd[np.isinf(prepd)] = 0.0
        # prepd = prepd[~np.isinf(prepd).any(axis=1)]
    if not np.isfinite(prepd).all():
        n_nan = np.sum(np.isnan(prepd).any(axis=1))
        LOGGER.info(f"Found {n_nan} nan events")
        prepd[np.isnan(prepd)] = 0.0
        # prepd = prepd[~np.isnan(prepd).any(axis=1)]
    return prepd, prep_params


def _get_naive(event, cfg, for_mlp=True):
    naive = event.copy()
    if cfg.prep_cls.naive.gaussianize:
        for i, pt_cut in enumerate(cfg.prep_cls.naive.pt_cut):
            if i >= event.shape[1]:
                break
            naive[..., i, 0] -= pt_cut - 2 * EPS
        naive[..., 0] = np.log(naive[..., 0])
        naive[..., 1] = stable_arctanh(naive[..., 1] / np.pi, EPS)
        naive[..., 3] = np.log(naive[..., 3])

    naive = naive.reshape(naive.shape[0], naive.shape[1] * naive.shape[2])
    if for_mlp:
        if cfg.prep_cls.naive.channels is not None:
            # keep the maximum of the prep_cls.naive.channels that are also in cfg.data.channels
            channels = [
                i for i in cfg.prep_cls.naive.channels if i in cfg.data.channels
            ]
        else:
            # remove only the channels that are not in data.channels
            channels = [i for i in range(naive.shape[1]) if i in cfg.data.channels]
        naive = naive[..., channels]
    else:
        # manually set unused channels to zero (can not drop them)
        channels_out = [i for i in range(naive.shape[1]) if i not in cfg.data.channels]
        naive[..., channels_out] = 0.0

    return naive


def _get_virtual(event, cfg):
    virtual = []
    components_list = (
        cfg.data.virtual_components
        if cfg.prep_cls.virtual.components is None
        else cfg.prep_cls.virtual.components
    )
    for i, components in enumerate(components_list):
        particle = get_virtual_particle(event, components)
        if cfg.prep_cls.virtual.gaussianize:
            # note: virtual particles do not have a pt cut
            particle[..., 0] = np.log(particle[..., 0])
            particle[..., 1] = np.arctanh(particle[..., 1] / np.pi)
            particle[..., 3] = np.log(particle[..., 3])
        if cfg.prep_cls.virtual.channels is not None:
            assert len(cfg.prep_cls.virtual.channels) == len(components_list)
            # make particle = 0 when it is not in the channels
            particle[
                ...,
                [
                    i
                    for i in range(particle.shape[1])
                    if i not in cfg.prep_cls.virtual.channels
                ],
            ] = 0.0
        virtual.append(particle)
    virtual = np.concatenate(virtual, axis=1)
    return virtual


def _get_deltaR(event, cfg):
    deltaR = []
    num_particles = event.shape[1]
    particles = np.arange(num_particles)

    idxs = [(idx1, idx2) for idx1 in particles for idx2 in particles]
    for idx1, idx2 in idxs:
        if idx1 >= idx2:
            continue
        deltaR.append(delta_r(event, idx1, idx2))
    deltaR = np.stack(deltaR, axis=1)
    if cfg.prep_cls.delta_r.invert:
        deltaR = 1 / np.clip(deltaR, EPS2, None)
    return deltaR


def _get_deltaphi(event, cfg):
    deltaphi = []
    num_particles = event.shape[1]
    particles = np.arange(num_particles)

    idxs = [(idx1, idx2) for idx1 in particles for idx2 in particles]
    for idx1, idx2 in idxs:
        if idx1 >= idx2:
            continue
        deltaphi.append(delta_phi(event, idx1, idx2))
    deltaphi = np.stack(deltaphi, axis=1)
    return deltaphi


def _get_deltaeta(event, cfg):
    deltaeta = []
    num_particles = event.shape[1]
    particles = np.arange(num_particles)

    idxs = [(idx1, idx2) for idx1 in particles for idx2 in particles]
    for idx1, idx2 in idxs:
        if idx1 >= idx2:
            continue
        deltaeta.append(delta_eta(event, idx1, idx2))
    deltaeta = np.stack(deltaeta, axis=1)
    return deltaeta

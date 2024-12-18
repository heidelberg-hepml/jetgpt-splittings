import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.eventgen.helpers import (
    delta_eta,
    delta_phi,
    delta_r,
    get_virtual_particle,
    EPPP_to_PtPhiEtaM2,
)
from src.base_plots import plot_loss, plot_metric
from src.eventgen.plots import (
    plot_histogram,
    plot_histogram_2d,
)


def plot_losses(exp, filename, model_label):
    with PdfPages(filename) as file:
        plot_loss(
            file,
            [exp.train_loss, exp.val_loss],
            exp.train_lr,
            labels=["train loss", "val loss"],
            logy=False,
        )
        plot_metric(
            file,
            [exp.train_grad_norm],
            "Gradient norm",
            logy=True,
        )


def plot_fourmomenta(exp, filename, model_label, weights, mask_dict):
    obs_names = []
    for name in exp.obs_names_index:
        obs_names.extend(
            [
                r"E_{" + name + "} \;[\mathrm{GeV}]",
                r"p_{x," + name + "} \;[\mathrm{GeV}]",
                r"p_{y," + name + "} \;[\mathrm{GeV}]",
                r"p_{z," + name + "} \;[\mathrm{GeV}]",
            ]
        )

    with PdfPages(filename) as file:
        for ijet in range(exp.cfg.data.n_jets_max):
            num_components = 4 * (exp.n_hard_particles + ijet)
            for channel in range(num_components):

                def extract(event):
                    if event.shape[0] == 0:
                        return event
                    event = event.clone()
                    event = event.reshape(event.shape[0], -1)[:, channel]
                    return event

                truth = extract(exp.data[ijet])
                model = [extract(samples) for samples in exp.samples[ijet]]
                xlabel = obs_names[channel]
                xrange = exp.fourmomentum_ranges[channel % 4]
                logy = False
                plot_histogram(
                    file=file,
                    truth=truth,
                    model=model,
                    title=exp.plot_titles[ijet],
                    xlabel=xlabel,
                    xrange=xrange,
                    logy=logy,
                    model_label=model_label,
                    weights=weights[ijet],
                    mask_dict=mask_dict[ijet],
                )


def plot_conservation(exp, filename, model_label, weights, mask_dict):
    with PdfPages(filename) as file:
        for ijet in range(exp.cfg.data.n_jets_max):

            def extract(event):
                if event.shape[0] == 0:
                    return event
                event = event.clone()
                event = event.reshape(event.shape[0], -1, 4).sum(dim=-2)
                return event

            truth = extract(exp.data[ijet])
            model = [extract(samples) for samples in exp.samples[ijet]]

            obs_names = [
                r"p_{x,\Sigma} \;[\mathrm{GeV}]",
                r"p_{y,\Sigma} \;[\mathrm{GeV}]",
                r"p_{z,\Sigma} \;[\mathrm{GeV}]",
                r"p_{T,\Sigma} \;[\mathrm{GeV}]",
            ]
            operations = [
                lambda x: x[..., 1],
                lambda x: x[..., 2],
                lambda x: x[..., 3],
                lambda x: (x[..., 1] ** 2 + x[..., 2] ** 2) ** 0.5,
            ]
            dp = 100
            xranges = [[-dp, dp], [-dp, dp], [-20 * dp, 20 * dp], [0, 2**0.5 * dp]]
            logy = False
            for obs_name, operation, xrange in zip(obs_names, operations, xranges):
                plot_histogram(
                    file=file,
                    truth=operation(truth),
                    model=[operation(m) for m in model],
                    title=exp.plot_titles[ijet],
                    xlabel=obs_name,
                    xrange=xrange,
                    logy=logy,
                    model_label=model_label,
                    weights=weights[ijet],
                    mask_dict=mask_dict[ijet],
                )


def plot_jetmomenta(exp, filename, model_label, weights, mask_dict):
    obs_names = []
    for name in exp.obs_names_index:
        obs_names.extend(
            [
                r"p_{T," + name + "} \;[\mathrm{GeV}]",
                r"\phi_{" + name + "}",
                r"\eta_{" + name + "}",
                r"m_{" + name + "} \;[\mathrm{GeV}]",
            ]
        )
    logys = [True, False, False, False]

    with PdfPages(filename) as file:
        for ijet in range(exp.cfg.data.n_jets_max):
            num_components = 4 * (exp.n_hard_particles + ijet)
            for channel in range(num_components):

                def extract(event):
                    if event.shape[0] == 0:
                        return event
                    event = event.clone()
                    event = EPPP_to_PtPhiEtaM2(event, sqrt_mass=True)
                    event = event.reshape(event.shape[0], -1)[:, channel]
                    return event

                truth = extract(exp.data[ijet])
                model = [extract(samples) for samples in exp.samples[ijet]]
                xlabel = obs_names[channel]
                xrange = exp.jetmomentum_ranges[channel % 4]
                logy = logys[channel % 4]
                plot_histogram(
                    file=file,
                    truth=truth,
                    model=model,
                    title=exp.plot_titles[ijet],
                    xlabel=xlabel,
                    xrange=xrange,
                    logy=logy,
                    model_label=model_label,
                    weights=weights[ijet],
                    mask_dict=mask_dict[ijet],
                )


def plot_preprocessed(exp, filename, model_label, weights, mask_dict):
    with PdfPages(filename) as file:
        for ijet in range(exp.cfg.data.n_jets_max):

            def extract(x, channel):
                if x.shape[0] == 0:
                    # happens when no training events are available for a multiplicity
                    return x
                pidx = torch.tensor(
                    list(range(exp.n_hard_particles)) + [exp.n_hard_particles] * ijet
                )
                x = exp.model.preprocessing.preprocess(torch.tensor(x), pidx)
                x = x.reshape(x.shape[0], -1)
                return x[:, channel]

            nevents = exp.samples[ijet][0].shape[0]
            if nevents == 0:
                continue

            for channel in range(exp.samples[ijet][0].reshape(nevents, -1).shape[1]):
                if channel in 3 + 4 * np.array(exp.onshell_list):
                    continue
                try:
                    truth = extract(exp.data[ijet], channel)
                    model = [extract(samples, channel) for samples in exp.samples[ijet]]
                except Exception:
                    continue
                xlabel = r"\mathrm{channel}\ " + str(channel)
                xrange = [-5, 5]
                logy = False
                plot_histogram(
                    file=file,
                    truth=truth,
                    model=model,
                    title=exp.plot_titles[ijet],
                    xlabel=xlabel,
                    xrange=xrange,
                    logy=logy,
                    model_label=model_label,
                    weights=weights[ijet],
                    mask_dict=mask_dict[ijet],
                )


def plot_delta(exp, filename, model_label, weights, mask_dict):
    with PdfPages(filename) as file:
        for ijet in range(exp.cfg.data.n_jets_max):
            num_particles = exp.n_hard_particles + ijet
            particles = np.arange(num_particles)

            for idx1 in particles:
                for idx2 in particles:
                    if idx1 >= idx2:
                        continue

                    # delta eta
                    get_delta_eta = lambda x: delta_eta(
                        EPPP_to_PtPhiEtaM2(x), idx1, idx2
                    )
                    xlabel = (
                        r"\Delta \eta_{%s}"
                        % f"{exp.obs_names_index[idx1]},{exp.obs_names_index[idx2]}"
                    )
                    xrange = [-6.0, 6.0]
                    truth = get_delta_eta(exp.data[ijet])
                    model = [get_delta_eta(samples) for samples in exp.samples[ijet]]
                    plot_histogram(
                        file=file,
                        truth=truth,
                        model=model,
                        title=exp.plot_titles[ijet],
                        xlabel=xlabel,
                        xrange=xrange,
                        logy=False,
                        model_label=model_label,
                        weights=weights[ijet],
                        mask_dict=mask_dict[ijet],
                    )

                    # delta phi
                    get_delta_phi = lambda x: delta_phi(
                        EPPP_to_PtPhiEtaM2(x), idx1, idx2
                    )
                    xlabel = (
                        r"\Delta \phi_{%s}"
                        % f"{exp.obs_names_index[idx1]},{exp.obs_names_index[idx2]}"
                    )
                    xrange = [-math.pi, math.pi]
                    truth = get_delta_phi(exp.data[ijet])
                    model = [get_delta_phi(samples) for samples in exp.samples[ijet]]
                    plot_histogram(
                        file=file,
                        truth=truth,
                        model=model,
                        title=exp.plot_titles[ijet],
                        xlabel=xlabel,
                        xrange=xrange,
                        logy=False,
                        model_label=model_label,
                        weights=weights[ijet],
                        mask_dict=mask_dict[ijet],
                    )

                    # delta R
                    get_delta_r = lambda x: delta_r(EPPP_to_PtPhiEtaM2(x), idx1, idx2)
                    xlabel = (
                        r"\Delta R_{%s}"
                        % f"{exp.obs_names_index[idx1]},{exp.obs_names_index[idx2]}"
                    )
                    xrange = [0.0, 8.0]
                    truth = get_delta_r(exp.data[ijet])
                    model = [get_delta_r(samples) for samples in exp.samples[ijet]]
                    plot_histogram(
                        file=file,
                        truth=truth,
                        model=model,
                        title=exp.plot_titles[ijet],
                        xlabel=xlabel,
                        xrange=xrange,
                        logy=False,
                        model_label=model_label,
                        weights=weights[ijet],
                        mask_dict=mask_dict[ijet],
                    )


def plot_virtual(exp, filename, model_label, weights, mask_dict):
    logys = [True, False, False, False]
    with PdfPages(filename) as file:
        for ijet in range(exp.cfg.data.n_jets_max):
            for i, components in enumerate(exp.virtual_components):

                def get_virtual(x):
                    jetmomenta = get_virtual_particle(EPPP_to_PtPhiEtaM2(x), components)
                    return jetmomenta

                truth = get_virtual(exp.data[ijet])
                model = [get_virtual(samples) for samples in exp.samples[ijet]]
                for j in range(4):
                    plot_histogram(
                        file=file,
                        truth=truth[:, j],
                        model=[m[:, j] for m in model],
                        title=exp.plot_titles[ijet],
                        xlabel=exp.virtual_names[4 * i + j],
                        xrange=exp.virtual_ranges[4 * i + j],
                        logy=logys[j],
                        model_label=model_label,
                        weights=weights[ijet],
                        mask_dict=mask_dict[ijet],
                    )


def plot_deta_dphi(exp, filename, model_label):
    with PdfPages(filename) as file:
        for ijet in range(exp.cfg.data.n_jets_max):
            num_particles = exp.n_hard_particles + ijet
            particles = np.arange(num_particles)

            for idx1 in particles:
                for idx2 in particles:
                    if idx1 >= idx2:
                        continue

                    def construct(event):
                        deta = delta_eta(EPPP_to_PtPhiEtaM2(event), idx1, idx2)
                        dphi = delta_phi(EPPP_to_PtPhiEtaM2(event), idx1, idx2)
                        return np.stack([deta, dphi], axis=-1)

                    truth = construct(exp.data[ijet])
                    model = np.concatenate(
                        [construct(samples) for samples in exp.samples[ijet]], axis=0
                    )
                    xlabel = (
                        r"\Delta \eta_{%s}"
                        % f"{exp.obs_names_index[idx1]},{exp.obs_names_index[idx2]}"
                    )
                    ylabel = (
                        r"\Delta \phi_{%s}"
                        % f"{exp.obs_names_index[idx1]},{exp.obs_names_index[idx2]}"
                    )
                    xrange = [-4.0, 4.0]
                    yrange = [-np.pi, np.pi]
                    plot_histogram_2d(
                        file=file,
                        truth=truth,
                        model=model,
                        title=exp.plot_titles[ijet],
                        xlabel=xlabel,
                        ylabel=ylabel,
                        xrange=xrange,
                        yrange=yrange,
                        model_label=model_label,
                    )


def plot_prob_stop(exp, filename, model_label):
    colors = ["black", "#A52A2A"]
    with PdfPages(filename) as file:
        max_mult = max(len(exp.data.values()), len(exp.samples.values()))
        mult_data = [0] * max_mult
        for ijet, x in exp.data.items():
            if ijet in exp.cfg.data.n_jets_train:
                mult_data[ijet] = x.shape[0]
        mult_data = np.array(mult_data) / np.sum(mult_data)
        mult_model = [[0 for _ in range(len(exp.samples[0]))]] * max_mult
        for ijet, x in exp.samples.items():
            mult_model[ijet] = [y.shape[0] for y in x]
        mult_model = np.array(mult_model) / np.array(mult_model).mean(axis=1).sum(
            axis=0
        )
        mult_model_mean = np.mean(mult_model, axis=1)
        mult_model_std = np.std(mult_model, axis=1)

        # not log-scaled relative multiplicities
        plt.step(
            np.arange(len(mult_data)),
            mult_data,
            label="data",
            where="mid",
            color=colors[0],
        )
        plt.step(
            np.arange(len(mult_model_mean)),
            mult_model_mean,
            label="model",
            where="mid",
            color=colors[1],
        )
        plt.fill_between(
            np.arange(len(mult_model_mean)),
            mult_model_mean - mult_model_std,
            mult_model_mean + mult_model_std,
            alpha=0.5,
            step="mid",
            facecolor=colors[1],
        )
        plt.legend(loc=1)
        plt.ylabel("Fraction of events")
        plt.xlabel("Multiplicity")
        plt.xlim(0, max_mult)
        _, ymax = plt.ylim()
        plt.ylim(0, ymax)
        plt.savefig(file, bbox_inches="tight", format="pdf")
        plt.close()

        # log-scaled relative multiplicities
        plt.yscale("log")
        plt.step(
            np.arange(len(mult_data)),
            mult_data,
            label="data",
            where="mid",
            color=colors[0],
        )
        plt.step(
            np.arange(len(mult_model_mean)),
            mult_model_mean,
            label="model",
            where="mid",
            color=colors[1],
        )
        plt.fill_between(
            np.arange(len(mult_model_mean)),
            mult_model_mean - mult_model_std,
            mult_model_mean + mult_model_std,
            alpha=0.5,
            step="mid",
            facecolor=colors[1],
        )
        plt.legend(loc=1)
        plt.ylabel("Fraction of events")
        plt.xlabel("Multiplicity")
        plt.xlim(0, max_mult)
        _, ymax = plt.ylim()
        plt.ylim(0, ymax)
        plt.savefig(file, bbox_inches="tight", format="pdf")
        plt.close()

        for ijet in range(exp.cfg.data.n_jets_max):
            prob_stop = [
                torch.cat(
                    [
                        exp.prob_stop[ijet + ijet2][i][:, ijet]
                        for ijet2 in range(exp.cfg.data.n_jets_max - ijet)
                    ]
                )
                for i in range(len(exp.prob_stop[ijet]))
            ]
            prob_continue = [1 - ps for ps in prob_stop]
            y = np.array(
                [np.histogram(pc, bins=100, range=(0, 1))[0] for pc in prob_continue]
            )
            bins = np.linspace(0, 1, 101)
            y_mean = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)
            scale = 1 / np.sum((bins[1:] - bins[:-1]) * y)
            dup_last = lambda a: np.append(a, a[-1])
            plt.step(bins, dup_last(y_mean) * scale, where="post", color=colors[1])
            plt.fill_between(
                bins,
                dup_last(y_mean + y_std) * scale,
                dup_last(y_mean - y_std) * scale,
                alpha=0.5,
                step="post",
                facecolor=colors[1],
            )
            plt.ylabel("Number of such events")
            plt.xlabel("Probability for having an extra jet")
            _, ymax = plt.ylim()
            plt.ylim(0, ymax)
            plt.xlim(0, 1)
            plt.title(f"{ijet} jets")
            plt.savefig(file, bbox_inches="tight", format="pdf")
            plt.close()

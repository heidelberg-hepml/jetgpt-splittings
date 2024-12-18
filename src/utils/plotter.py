import numpy as np
import math
from matplotlib.backends.backend_pdf import PdfPages
from src.utils.mlflow import log_mlflow

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.calibration import calibration_curve
from src.utils.physics import delta_eta, delta_phi, delta_r, get_virtual_particle

from src.utils.plots_basic import plot_loss, plot_lines, plot_weight_quantiles
from src.utils.plots_generator import plot_histogram, plot_histogram_2d
from src.utils.plots_classifier import simple_histogram, plot_roc, plot_calibration
from src.models.discformer import SAVE_METRICS_EACH


def plot_gen_metrics(exp, filename):
    with PdfPages(filename) as file:
        plot_loss(
            file,
            losses=[exp.train_loss, exp.val_loss],
            lr=exp.train_lr,
            labels=["train", "val"],
            logy=False,
            ylabel="Loss",
        )
        for n_jets in exp.cfg.data.n_jets_list:
            plot_loss(
                file,
                losses=[
                    exp.train_metrics[f"neg_log_prob.{n_jets}j"],
                    exp.val_metrics[f"neg_log_prob.{n_jets}j"],
                ],
                lr=exp.train_lr,
                labels=["train", "val"],
                logy=False,
                ylabel=r"$-\log p$ for ${%s}j$" % n_jets,
            )
        plot_lines(
            file,
            lines=[exp.grad_norm],
            labels=["Gradient norm"],
            ylabel="Gradient norm",
            logy=True,
        )


def plot_cls_metrics(exp, classifier, filename):
    with PdfPages(filename) as file:
        plot_loss(
            file,
            losses=[classifier.train_loss, classifier.val_loss],
            lr=classifier.train_lr,
            labels=["train", "val"],
            logy=False,
            ylabel="Loss",
        )
        for n_jets in exp.cfg.data.n_jets_list:
            plot_loss(
                file,
                losses=[
                    classifier.train_metrics[f"bce.{n_jets}j"],
                    classifier.val_metrics[f"bce.{n_jets}j"],
                ],
                lr=classifier.train_lr,
                labels=["train", "val"],
                logy=False,
                ylabel=r"BCE for ${%s}j$" % n_jets,
            )


def plot_classifier(exp, classifier, filename):

    if exp.cfg.plotting.classifier_masked:
        # construct masks for generated events
        masks = []
        for ijet, gen_raw in enumerate(exp.generator.gen_raw):
            mask = np.zeros((gen_raw.shape[0]), dtype="bool")
            num_particles = exp.n_hard_particles + exp.cfg.data.n_jets_list[ijet]
            particles = np.arange(num_particles)
            for idx1 in particles:
                for idx2 in particles:
                    if idx1 >= idx2:
                        continue
                    deltaR = delta_r(gen_raw, idx1, idx2)
                    mask += deltaR < 0.4
            masks.append(mask)
        mask_dict = [
            {"condition": r"$\Delta R<0.4$", "mask": mask, "color": "violet"}
            for mask in masks
        ]
    else:
        mask_dict = [None for _ in exp.cfg.data.n_jets_list]

    with PdfPages(filename) as file:
        # histograms
        for ijet, n_jets in enumerate(exp.cfg.data.n_jets_list):
            # scores
            data = [
                classifier.results_eval["truth"]["scores"][ijet],
                classifier.results_eval["generator"]["scores"][ijet],
            ]
            simple_histogram(
                file,
                data,
                labels=["Test", "JetGPT"],
                xrange=[0, 1],
                xlabel="Classifier score",
                title=exp.plot_titles[ijet],
                logx=False,
                logy=False,
                mask_dict=mask_dict[ijet],
            )
            simple_histogram(
                file,
                data,
                labels=["Test", "JetGPT"],
                xrange=[0, 1],
                xlabel="Classifier score",
                title=exp.plot_titles[ijet],
                logx=False,
                logy=True,
                mask_dict=mask_dict[ijet],
            )

            # weights
            data = [
                classifier.results_eval["truth"]["weights"][ijet],
                classifier.results_eval["generator"]["weights"][ijet],
            ]
            simple_histogram(
                file,
                data,
                labels=["Test", "JetGPT"],
                xrange=[0, 5],
                xlabel="Classifier weights",
                title=exp.plot_titles[ijet],
                logx=False,
                logy=False,
                mask_dict=mask_dict[ijet],
            )
            simple_histogram(
                file,
                data,
                labels=["Test", "JetGPT"],
                xrange=[1e-3, 1e2],
                xlabel="Classifier weights",
                title=exp.plot_titles[ijet],
                logx=True,
                logy=True,
                mask_dict=mask_dict[ijet],
            )

        # roc curve
        for ijet, n_jets in enumerate(exp.cfg.data.n_jets_list):
            # prepare data
            results = [
                {
                    key: value[ijet]
                    for key, value in classifier.results_eval["truth"].items()
                },
                {
                    key: value[ijet]
                    for key, value in classifier.results_eval["generator"].items()
                },
            ]
            labels_predict = np.concatenate(
                [result["scores"] for result in results], axis=0
            )
            labels_true = np.concatenate(
                [
                    np.ones_like(results[0]["weights"]),
                    np.zeros_like(results[1]["weights"]),
                ],
                axis=0,
            )

            # scores
            fpr, tpr, th = roc_curve(labels_true, labels_predict)
            auc = roc_auc_score(labels_true, labels_predict)
            accuracy = accuracy_score(labels_true, np.round(labels_predict))
            metrics = {"auc": auc, "accuracy": accuracy}
            if exp.cfg.use_mlflow:
                for key, value in metrics.items():
                    log_mlflow(f"{classifier.label_short}.{key}.{n_jets}j", value)

            # plot
            plot_roc(file, tpr, fpr, auc, title=exp.plot_titles[ijet])

        # calibration curve
        for ijet, n_jets in enumerate(exp.cfg.data.n_jets_list):
            # preprate data
            scores_true = classifier.results_eval["truth"]["scores"][ijet]
            scores_fake = classifier.results_eval["generator"]["scores"][ijet]
            n_min = min(len(scores_true), len(scores_fake))
            scores = np.concatenate((scores_fake[:n_min], scores_true[:n_min]))
            labels = np.concatenate((np.zeros(n_min), np.ones(n_min)))
            prob_true, prob_pred = calibration_curve(labels, scores, n_bins=30)

            # plot
            plot_calibration(file, prob_true, prob_pred, exp.plot_titles[ijet])

        if exp.cfg.plotting.classifier_preprocessed:
            for ijet, n_jets in enumerate(exp.cfg.data.n_jets_list):
                for channel in range(classifier.prepd_true[ijet]["trn"].shape[-1]):
                    true = classifier.prepd_true[ijet]["trn"][:, channel]
                    fake = classifier.prepd_fake[ijet]["trn"][:, channel]

                    xlabel = f"channel {channel}"
                    xrange = [-5, 5]
                    logy = False
                    simple_histogram(
                        file=file,
                        data=[true, fake],
                        labels=["True", "JetGPT"],
                        title=exp.plot_titles[ijet],
                        xlabel=xlabel,
                        xrange=xrange,
                        logy=logy,
                    )


def plot_discformer(exp, filename):
    with PdfPages(filename) as file:
        for ijet, n_jets in enumerate(exp.cfg.data.n_jets_list):
            # clamp_ratio
            plot_lines(
                file,
                lines=[
                    exp.discformer.clamp_ratio[f"{n_jets}j"][key]
                    for key in exp.discformer.keys
                ],
                labels=[key for key in exp.discformer.keys],
                ylabel="Fraction of clipped events",
                title=exp.plot_titles[ijet],
                logy=False,
                step_multiplier=SAVE_METRICS_EACH,
            )

            # efficiency
            plot_lines(
                file,
                lines=[
                    exp.discformer.efficiency[f"{n_jets}j"][key]
                    for key in exp.discformer.keys
                ],
                labels=[key for key in exp.discformer.keys],
                ylabel="Unweighting efficiency",
                title=exp.plot_titles[ijet],
                logy=False,
                step_multiplier=SAVE_METRICS_EACH,
            )

            # weight_quantiles
            for key in exp.discformer.keys:
                plot_weight_quantiles(
                    file,
                    exp.discformer.weight_quantiles[f"{n_jets}j"][key],
                    ylabel=f"{key} quantiles",
                    title=exp.plot_titles[ijet],
                    step_multiplier=SAVE_METRICS_EACH,
                )

        # alpha (same for all n_jets)
        plot_lines(file, [exp.discformer.alpha], labels=[None], ylabel=r"$\alpha$")


def plot_jetmomenta(exp, filename, reweight_dict, mask_dict):
    with PdfPages(filename) as file:
        for ijet in range(len(exp.cfg.data.n_jets_list)):
            num_components = 4 * (exp.n_hard_particles + exp.cfg.data.n_jets_list[ijet])
            for channel in exp.channels[ijet]:
                if channel in exp.channels_out:
                    continue

                def extract(event):
                    event = event.copy()
                    event = event.reshape(event.shape[0], -1)[:, channel]
                    return event

                train = extract(exp.generator.truth_raw[ijet]["trn"])
                test = extract(exp.generator.truth_raw[ijet]["tst"])
                model = extract(exp.generator.gen_raw[ijet])
                xlabel = exp.obs_names[channel]
                xrange = exp.obs_ranges[channel]
                logy = exp.obs_logy[channel]
                plot_histogram(
                    file=file,
                    train=train,
                    test=test,
                    model=model,
                    title=exp.plot_titles[ijet],
                    xlabel=xlabel,
                    xrange=xrange,
                    logy=logy,
                    reweight_dict=reweight_dict[ijet]
                    if reweight_dict is not None
                    else None,
                    mask_dict=mask_dict[ijet] if mask_dict is not None else None,
                    jsdiv=exp.cfg.plotting.jsdiv,
                )


def plot_preprocessed(exp, filename, reweight_dict, mask_dict):
    with PdfPages(filename) as file:
        for ijet in range(len(exp.cfg.data.n_jets_list)):
            for channel in range(exp.generator.gen_prepd[ijet].shape[1]):
                train = exp.generator.truth_prepd[ijet]["trn"][:, channel]
                test = exp.generator.truth_prepd[ijet]["tst"][:, channel]
                model = exp.generator.gen_prepd[ijet][:, channel]
                xlabel = r"\mathrm{channel}\ " + str(exp.channels[ijet][channel])
                xrange = [-5, 5]
                logy = False
                plot_histogram(
                    file=file,
                    train=train,
                    test=test,
                    model=model,
                    title=exp.plot_titles[ijet],
                    xlabel=xlabel,
                    xrange=xrange,
                    logy=logy,
                    reweight_dict=reweight_dict[ijet]
                    if reweight_dict is not None
                    else None,
                    mask_dict=mask_dict[ijet] if mask_dict is not None else None,
                    jsdiv=exp.cfg.plotting.jsdiv,
                )


def plot_delta(exp, filename, reweight_dict, mask_dict):
    with PdfPages(filename) as file:
        for ijet in range(len(exp.cfg.data.n_jets_list)):
            num_particles = exp.n_hard_particles + exp.cfg.data.n_jets_list[ijet]
            particles = np.arange(num_particles)

            for idx1 in particles:
                for idx2 in particles:
                    if idx1 >= idx2:
                        continue

                    # delta eta
                    xlabel = (
                        r"\Delta \eta_{%s}"
                        % f"{exp.obs_names_index[idx1]},{exp.obs_names_index[idx2]}"
                    )
                    xrange = [-6.0, 6.0]
                    train = delta_eta(exp.generator.truth_raw[ijet]["trn"], idx1, idx2)
                    test = delta_eta(exp.generator.truth_raw[ijet]["tst"], idx1, idx2)
                    model = delta_eta(exp.generator.gen_raw[ijet], idx1, idx2)
                    plot_histogram(
                        file=file,
                        train=train,
                        test=test,
                        model=model,
                        title=exp.plot_titles[ijet],
                        xlabel=xlabel,
                        xrange=xrange,
                        logy=False,
                        reweight_dict=reweight_dict[ijet]
                        if reweight_dict is not None
                        else None,
                        mask_dict=mask_dict[ijet] if mask_dict is not None else None,
                        jsdiv=exp.cfg.plotting.jsdiv,
                    )

                    # delta phi
                    xlabel = (
                        r"\Delta \phi_{%s}"
                        % f"{exp.obs_names_index[idx1]},{exp.obs_names_index[idx2]}"
                    )
                    xrange = [-math.pi, math.pi]
                    train = delta_phi(exp.generator.truth_raw[ijet]["trn"], idx1, idx2)
                    test = delta_phi(exp.generator.truth_raw[ijet]["tst"], idx1, idx2)
                    model = delta_phi(exp.generator.gen_raw[ijet], idx1, idx2)
                    plot_histogram(
                        file=file,
                        train=train,
                        test=test,
                        model=model,
                        title=exp.plot_titles[ijet],
                        xlabel=xlabel,
                        xrange=xrange,
                        logy=False,
                        reweight_dict=reweight_dict[ijet]
                        if reweight_dict is not None
                        else None,
                        mask_dict=mask_dict[ijet] if mask_dict is not None else None,
                        jsdiv=exp.cfg.plotting.jsdiv,
                    )

                    # delta R
                    xlabel = (
                        r"\Delta R_{%s}"
                        % f"{exp.obs_names_index[idx1]},{exp.obs_names_index[idx2]}"
                    )
                    xrange = [0.0, 8.0]
                    train = delta_r(exp.generator.truth_raw[ijet]["trn"], idx1, idx2)
                    test = delta_r(exp.generator.truth_raw[ijet]["tst"], idx1, idx2)
                    model = delta_r(exp.generator.gen_raw[ijet], idx1, idx2)
                    plot_histogram(
                        file=file,
                        train=train,
                        test=test,
                        model=model,
                        title=exp.plot_titles[ijet],
                        xlabel=xlabel,
                        xrange=xrange,
                        logy=False,
                        reweight_dict=reweight_dict[ijet]
                        if reweight_dict is not None
                        else None,
                        mask_dict=mask_dict[ijet] if mask_dict is not None else None,
                        jsdiv=exp.cfg.plotting.jsdiv,
                    )


def plot_virtual(exp, filename, reweight_dict, mask_dict):
    with PdfPages(filename) as file:
        for ijet in range(len(exp.cfg.data.n_jets_list)):
            for i, components in enumerate(exp.virtual_components):
                train = get_virtual_particle(
                    exp.generator.truth_raw[ijet]["trn"], components
                )
                test = get_virtual_particle(
                    exp.generator.truth_raw[ijet]["tst"], components
                )
                model = get_virtual_particle(exp.generator.gen_raw[ijet], components)
                for j in range(4):
                    plot_histogram(
                        file=file,
                        train=train[:, j],
                        test=test[:, j],
                        model=model[:, j],
                        title=exp.plot_titles[ijet],
                        xlabel=exp.virtual_names[4 * i + j],
                        xrange=exp.virtual_ranges[4 * i + j],
                        logy=exp.virtual_logy[4 * i + j],
                        reweight_dict=reweight_dict[ijet]
                        if reweight_dict is not None
                        else None,
                        mask_dict=mask_dict[ijet] if mask_dict is not None else None,
                        jsdiv=exp.cfg.plotting.jsdiv,
                    )


def plot_deta_dphi(exp, filename, reweight_dict, mask_dict):
    with PdfPages(filename) as file:
        for ijet in range(len(exp.cfg.data.n_jets_list)):
            num_particles = exp.n_hard_particles + exp.cfg.data.n_jets_list[ijet]
            particles = np.arange(num_particles)

            for idx1 in particles:
                for idx2 in particles:
                    if idx1 >= idx2:
                        continue

                    def construct(event):
                        deta = delta_eta(event, idx1, idx2)
                        dphi = delta_phi(event, idx1, idx2)
                        return np.stack([deta, dphi], axis=-1)

                    test = construct(exp.generator.truth_raw[ijet]["tst"])
                    model = construct(exp.generator.gen_raw[ijet])
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
                        test=test,
                        model=model,
                        title=exp.plot_titles[ijet],
                        xlabel=xlabel,
                        ylabel=ylabel,
                        xrange=xrange,
                        yrange=yrange,
                        reweight_dict=reweight_dict[ijet]
                        if reweight_dict is not None
                        else None,
                        mask_dict=mask_dict[ijet] if mask_dict is not None else None,
                    )

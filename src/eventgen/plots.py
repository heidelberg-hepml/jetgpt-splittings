import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# load fonts
import matplotlib.font_manager as font_manager

font_dir = ["src/utils/bitstream-charter-ttf/Charter/"]
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)
font_manager.findSystemFonts(fontpaths=None, fontext="ttf")

# setup matplotlib
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams[
    "text.latex.preamble"
] = r"\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}"

FONTSIZE = 13  # pt
PAGEWIDTH = 11  # inches
MATPLOTLIB_PARAMS = {
    # Font sizes
    "font.size": FONTSIZE,  # controls default text sizes
    "axes.titlesize": FONTSIZE,  # fontsize of the axes title
    "axes.labelsize": FONTSIZE,  # fontsize of the x and y labels
    "xtick.labelsize": FONTSIZE,  # fontsize of the tick labels
    "ytick.labelsize": FONTSIZE,  # fontsize of the tick labels
    "legend.fontsize": FONTSIZE,  # legend fontsize
    "figure.titlesize": FONTSIZE,  # fontsize of the figure title
    # Figure size and DPI
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "figure.figsize": (PAGEWIDTH / 2, PAGEWIDTH / 2),
    # colors
    "lines.markeredgewidth": 0.8,
    "axes.edgecolor": "black",
    "axes.grid": False,
    "grid.color": "0.9",
    "axes.grid.which": "both",
    # x-axis ticks and grid
    "xtick.bottom": True,
    "xtick.direction": "out",
    "xtick.color": "black",
    "xtick.major.bottom": True,
    "xtick.major.size": 4,
    "xtick.minor.bottom": True,
    "xtick.minor.size": 2,
    # y-axis ticks and grid
    "ytick.left": True,
    "ytick.direction": "out",
    "ytick.color": "black",
    "ytick.major.left": True,
    "ytick.major.size": 4,
    "ytick.minor.left": True,
    "ytick.minor.size": 2,
}
matplotlib.rcParams.update(MATPLOTLIB_PARAMS)

COLORS = ["black", "#A52A2A"]
LEFT, BOTTOM, RIGHT, TOP = 0.15, 0.15, 0.95, 0.95
X_LABEL_POS, Y_LABEL_POS = -0.05, -0.13
N_BINS = 60
FIGSIZE = (5, 4)
RATIO_RANGE = [0.7, 1.3]
RATIO_TICKS = [0.8, 1.0, 1.2]


def plot_histogram(
    file,
    truth,
    model,
    title,
    xlabel,
    xrange,
    model_label,
    logy=False,
    weights=None,
    mask_dict=None,
):
    """
    Plotting code used for all 1d distributions

    Parameters:
    file: str
    truth: np.ndarray of shape (nevents)
    model: np.ndarray of shape (nevents)
    title: str
    xlabel: str
    xrange: tuple with 2 floats
    model_label: str
    logy: bool
    """
    # weights and mask_dict arguments are ignored

    # construct labels and colors
    labels = ["Truth", model_label]

    # construct histograms
    y_trn, bins = np.histogram(truth, bins=N_BINS, range=xrange)
    if len(model) == 1:
        # histogram from one set of samples (either non-BNN or BNN with 1 sample)
        model = model[0]
        y_mod, _ = np.histogram(model, bins=bins)
        hists = [y_trn, y_mod]
        hist_errors = [np.sqrt(y_trn), np.sqrt(y_mod)]
    else:
        # histogram from >1 BNN samples
        y_mod = np.stack(
            [np.histogram(model0, bins=bins)[0] for model0 in model], axis=0
        )
        hists = [y_trn, np.mean(y_mod, axis=0)]
        hist_errors = [np.sqrt(y_trn), np.std(y_mod, axis=0)]

    integrals = [np.sum((bins[1:] - bins[:-1]) * y) for y in hists]
    scales = [1 / integral if integral != 0.0 else 1.0 for integral in integrals]

    dup_last = lambda a: np.append(a, a[-1])

    fig, axs = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=FIGSIZE,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.00},
    )

    for i, y, y_err, scale, label, color in zip(
        range(len(hists)), hists, hist_errors, scales, labels, COLORS
    ):

        axs[0].step(
            bins,
            dup_last(y) * scale,
            label=label,
            color=color,
            linewidth=1.0,
            where="post",
        )
        axs[0].step(
            bins,
            dup_last(y + y_err) * scale,
            color=color,
            alpha=0.5,
            linewidth=0.5,
            where="post",
        )
        axs[0].step(
            bins,
            dup_last(y - y_err) * scale,
            color=color,
            alpha=0.5,
            linewidth=0.5,
            where="post",
        )
        axs[0].fill_between(
            bins,
            dup_last(y - y_err) * scale,
            dup_last(y + y_err) * scale,
            facecolor=color,
            alpha=0.3,
            step="post",
        )

        if label == "Train":
            axs[0].fill_between(
                bins,
                dup_last(y) * scale,
                0.0 * dup_last(y),
                facecolor=color,
                alpha=0.1,
                step="post",
            )
            continue

        ratio = (y * scale) / (hists[0] * scales[0])
        ratio_err = np.sqrt((y_err / y) ** 2 + (hist_errors[0] / hists[0]) ** 2)
        ratio_isnan = np.isnan(ratio)
        ratio[ratio_isnan] = 1.0
        ratio_err[ratio_isnan] = 0.0

        axs[1].step(bins, dup_last(ratio), linewidth=1.0, where="post", color=color)
        axs[1].step(
            bins,
            dup_last(ratio + ratio_err),
            color=color,
            alpha=0.5,
            linewidth=0.5,
            where="post",
        )
        axs[1].step(
            bins,
            dup_last(ratio - ratio_err),
            color=color,
            alpha=0.5,
            linewidth=0.5,
            where="post",
        )
        axs[1].fill_between(
            bins,
            dup_last(ratio - ratio_err),
            dup_last(ratio + ratio_err),
            facecolor=color,
            alpha=0.3,
            step="post",
        )

    axs[0].legend(loc="upper right", frameon=False, fontsize=FONTSIZE)
    axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)

    if logy:
        axs[0].set_yscale("log")

    _, ymax = axs[0].get_ylim()
    axs[0].set_ylim(0.0, ymax)
    axs[0].set_xlim(xrange)
    axs[0].tick_params(axis="both", labelsize=FONTSIZE)
    plt.xlabel(
        r"${%s}$" % xlabel,
        fontsize=FONTSIZE,
    )
    axs[0].text(
        0.05,
        0.95,
        s=title,
        horizontalalignment="left",
        verticalalignment="top",
        transform=axs[0].transAxes,
        fontsize=FONTSIZE,
    )

    axs[1].set_ylabel(
        r"$\frac{\mathrm{{%s}}}{\mathrm{Truth}}$" % model_label, fontsize=FONTSIZE
    )
    axs[1].set_yticks(RATIO_TICKS)
    axs[1].set_ylim(RATIO_RANGE)
    axs[1].axhline(y=RATIO_TICKS[0], c="black", ls="dotted", lw=0.5)
    axs[1].axhline(y=RATIO_TICKS[1], c="black", ls="--", lw=0.7)
    axs[1].axhline(y=RATIO_TICKS[2], c="black", ls="dotted", lw=0.5)
    axs[1].tick_params(axis="both", labelsize=FONTSIZE)

    fig.subplots_adjust(LEFT, BOTTOM, RIGHT, TOP)
    fig.savefig(file, format="pdf")
    plt.close()


def plot_histogram_2d(
    file,
    truth,
    model,
    title,
    xlabel,
    ylabel,
    xrange,
    yrange,
    model_label,
    n_bins=100,
):
    data = [truth, model]
    subtitles = ["Truth", model_label]

    fig, axs = plt.subplots(1, len(data), figsize=(4 * len(data), 4))
    for ax, dat, subtitle in zip(axs, data, subtitles):
        ax.set_title(subtitle)
        ax.hist2d(
            dat[:, 0],
            dat[:, 1],
            bins=n_bins,
            range=[xrange, yrange],
            rasterized=True,
        )
        ax.set_xlabel(r"${%s}$" % xlabel)
        ax.set_ylabel(r"${%s}$" % ylabel)
    fig.suptitle(title)
    plt.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()


def plot_calibration(file, prob_true, prob_pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(
        prob_true, prob_pred, color="#A52A2A", marker="o", markersize=3, linewidth=1
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("classifier probability for true events", fontsize=FONTSIZE)
    ax.set_ylabel("true fraction of true events", fontsize=FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICKLABELSIZE)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    plt.savefig(file, bbox_inches="tight", format="pdf")
    plt.close()


def plot_roc(file, tpr, fpr, auc):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#A52A2A", linewidth=1.0)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("false positive rate", fontsize=FONTSIZE)
    ax.set_ylabel("true positive rate", fontsize=FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICKLABELSIZE)
    ax.text(
        0.95,
        0.05,
        f"AUC = {auc:.4f}",
        verticalalignment="bottom",
        horizontalalignment="right",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )
    plt.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()


def simple_histogram(
    file, data, labels, xrange, xlabel, logx=False, logy=False, n_bins=80
):
    assert len(data) == 2 and len(labels) == 2
    colors = ["#0343DE", "#A52A2A"]
    dup_last = lambda a: np.append(a, a[-1])

    data = [np.clip(data_i.clone(), xrange[0], xrange[1]) for data_i in data]
    if logx:
        data = [np.log(data_i) for data_i in data]
        xrange = np.log(xrange)

    bins = np.histogram(data[0], bins=n_bins, range=xrange)[1]
    hists = [np.histogram(data_i, bins=bins, range=xrange)[0] for data_i in data]
    integrals = [np.sum((bins[1:] - bins[:-1]) * y) for y in hists]
    scales = [1 / integral if integral != 0.0 else 1.0 for integral in integrals]
    if logx:
        bins = np.exp(bins)
        xrange = np.exp(xrange)

    fig, ax = plt.subplots(figsize=(5, 4))
    for y, scale, label, color in zip(hists, scales, labels, colors):
        ax.step(
            bins,
            dup_last(y) * scale,
            label=label,
            color=color,
            linewidth=1.0,
            where="post",
        )
    ax.legend(loc="upper right", frameon=False, fontsize=FONTSIZE)
    ax.set_ylabel("Normalized", fontsize=FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)

    if logy:
        ax.set_yscale("log")
    else:
        _, ymax = ax.get_ylim()
        ax.set_ylim(0.0, ymax)
    if logx:
        ax.set_xscale("log")
    ax.set_xlim(xrange)
    ax.tick_params(axis="both", labelsize=TICKLABELSIZE)
    plt.savefig(file, bbox_inches="tight", format="pdf")
    plt.close()

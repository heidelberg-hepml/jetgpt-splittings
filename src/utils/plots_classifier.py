import numpy as np
import matplotlib.pyplot as plt

# use specific font
import matplotlib.font_manager as font_manager

font_dir = ["paper/bitstream-charter-ttf/Charter/"]
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)
font_manager.findSystemFonts(fontpaths=None, fontext="ttf")

# matplotlib settings
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams[
    "text.latex.preamble"
] = r"\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}"

# fontsize
FONTSIZE = 14
FONTSIZE_LEGEND = 12
TICKLABELSIZE = 11


def simple_histogram(
    file,
    data,
    labels,
    xrange,
    xlabel,
    title,
    logx=False,
    logy=False,
    n_bins=80,
    mask_dict=None,
):
    assert len(data) == 2 and len(labels) == 2
    colors = ["#0343DE", "#A52A2A"]
    dup_last = lambda a: np.append(a, a[-1])

    data = [np.clip(data_i.copy(), xrange[0], xrange[1]) for data_i in data]
    if logx:
        data = [np.log(data_i) for data_i in data]
        xrange = np.log(xrange)

    bins = np.histogram(data[0], bins=n_bins, range=xrange)[1]
    hists = [np.histogram(data_i, bins=bins, range=xrange)[0] for data_i in data]
    integrals = [np.sum((bins[1:] - bins[:-1]) * y) for y in hists]
    scales = [1 / integral if integral != 0.0 else 1.0 for integral in integrals]

    if mask_dict is not None:
        colors.append(mask_dict["color"])
        labels.append(f"JetGPT {mask_dict['condition']}")
        y_masked = np.histogram(data[1][mask_dict["mask"]], bins=bins, range=xrange)[0]
        hists.append(y_masked)
        scales.append(scales[1])  # handle normalization properly!

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
    ax.legend(loc="upper right", frameon=False, fontsize=FONTSIZE_LEGEND)
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
    ax.text(
        0.05,
        0.95,
        s=title,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )
    plt.savefig(file, bbox_inches="tight", format="pdf")
    plt.close()


def plot_roc(file, tpr, fpr, auc, title):
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
    ax.text(
        0.05,
        0.95,
        s=title,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )

    plt.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()


# should be called plot_calibration_scores
def plot_calibration(file, prob_true, prob_pred, title):

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
    ax.text(
        0.05,
        0.95,
        s=title,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )
    plt.savefig(file, bbox_inches="tight", format="pdf")
    plt.close()

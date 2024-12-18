import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ignore warnings here
import warnings

warnings.filterwarnings("ignore")

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

# fontsize
FONTSIZE = 14
FONTSIZE_LEGEND = 13
TICKLABELSIZE = 10

col = mpl.cm.Set1(range(10))


def plot_loss(file, losses, lr=None, labels=None, logy=False, ylabel="Loss"):
    if len(losses[1]) == 0:  # catch no-validations case
        losses = [losses[0]]
        labels = [labels[0]]
    labels = [None for _ in range(len(losses))] if labels is None else labels
    iterations = range(1, len(losses[0]) + 1)
    fig, ax = plt.subplots()
    for i, loss, label in zip(range(len(losses)), losses, labels):
        if len(loss) == len(iterations):
            its = iterations
        else:
            frac = len(losses[0]) / len(loss)
            its = np.arange(1, len(loss) + 1) * frac
        ax.plot(its, loss, label=label)

    if logy:
        ax.set_yscale("log")
    if lr is not None:
        axright = ax.twinx()
        axright.plot(iterations, lr, label="learning rate", color="crimson")
        axright.set_ylabel("Learning rate", fontsize=FONTSIZE)
    ax.set_xlabel("Number of iterations", fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False, loc="upper right")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()


def plot_lines(file, lines, labels, ylabel, title=None, logy=False, step_multiplier=1):
    """
    Plot a set of line (used for discformer tracking)
    """
    assert len(lines) == len(labels)
    colors = col[: len(lines)]

    fig, ax = plt.subplots()
    if logy:
        ax.set_yscale("log")
    for line, label, color in zip(lines, labels, colors):
        its = np.arange(1, len(line) + 1) * step_multiplier
        ax.plot(its, line, color=color, label=label)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.set_xlabel("iteration", fontsize=FONTSIZE)
    if np.all([label is not None for label in labels]):
        ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False, loc="upper right")
    ax.text(
        0.05,
        0.95,
        s=title,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()


def plot_weight_quantiles(
    file, weight_quantiles, ylabel, logy=True, title=None, step_multiplier=1
):
    # This code could be more elegantly

    def convolute(x, window_size=10):
        # determine window_size dynamically?
        if len(x) < window_size:  # no convolution
            return x

        window = np.ones(int(window_size)) / float(window_size)
        convd = np.convolve(x, window, "same")
        convd[:window_size] = x[:window_size]
        convd[-window_size:] = x[-window_size]
        return convd

    assert (
        len(weight_quantiles) % 2 == 1
    ), "weight_quantiles should have odd number of elements (mean and symmetric quantiles"

    # extract information from weight_quantiles dict
    keys = [key for key in list(weight_quantiles.keys())]
    values = [convolute(value) for value in list(weight_quantiles.values())]

    fig, ax = plt.subplots()
    alpha = np.linspace(0.1, 0.6, len(weight_quantiles) // 2)
    y = np.arange(1, len(values[0]) + 1) * step_multiplier
    for i in range(len(weight_quantiles)):
        if i < len(weight_quantiles) // 2:
            ax.plot(y, values[i], color="b", lw=0.2)
            ax.plot(y, values[-(1 + i)], color="b", lw=0.2)
            ax.fill_between(
                y,
                values[i],
                values[-(1 + i)],
                color="b",
                alpha=alpha[i],
                label=f"{keys[i]}" + r" $<p<$ " + f"{keys[-(1+i)]}",
            )
        elif i == (len(weight_quantiles) - 1) / 2:
            ax.plot(y, values[i], color="b", lw=1)
    ax.plot(y, np.zeros_like(y), "k--")

    if logy:
        ax.set_yscale("log")
    ax.set_xlim(min(y), max(y))
    ax.set_xlabel("iteration", fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False, loc="upper right")
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

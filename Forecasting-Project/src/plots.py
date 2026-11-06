"""Static report figures (matplotlib)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"     # categorical slot 1
ORANGE = "#eb6834"   # categorical slot 2

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "legend.frameon": False,
})

SN = "Seasonal naive"
ENS = "Ensemble (mean of ML models)"


def _k(ax):
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v/1000:,.0f}k"))


def _grid(n):
    fig, axes = plt.subplots(4, 2, figsize=(11, 11), sharex=True)
    axes = axes.ravel()
    axes[-1].axis("off")
    return fig, axes[:n], axes[-1]


def _legend_panel(src_ax, panel):
    handles, labels = src_ax.get_legend_handles_labels()
    panel.legend(handles, labels, loc="center", fontsize=10)


def plot_holdout(df, hold, path):
    ids = sorted(df["id"].unique())
    fig, axes, panel = _grid(len(ids))
    start = hold["Date"].min()
    for ax, sid in zip(axes, ids):
        a = df[(df["id"] == sid) & (df["Date"] >= start - np.timedelta64(26, "W"))]
        ax.plot(a["Date"], a["Weekly_Sales"], color=INK_2, lw=1.6, label="Actual")
        for name, color, ls in ((SN, ORANGE, (0, (4, 2))), (ENS, BLUE, "-")):
            p = hold[(hold["id"] == sid) & (hold["model"] == name)]
            ax.plot(p["Date"], p["yhat"], color=color, lw=2, ls=ls, label=name)
        ax.axvline(start, color=INK_2, lw=0.8, ls=":")
        ax.set_title(f"Store {sid.split('_')[0]}, Dept {sid.split('_')[1]}", loc="left")
        _k(ax)
    _legend_panel(axes[0], panel)
    fig.suptitle("52-week holdout: forecast vs. actual weekly sales "
                 "(dotted line = forecast start)", x=0.02, ha="left", fontweight="bold")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(hold_summary, roll_summary, path):
    order = hold_summary.sort_values("WAPE_%", ascending=False)["model"].tolist()
    h = hold_summary.set_index("model").loc[order, "WAPE_%"]
    r = roll_summary.set_index("model").loc[order, "WAPE_%"]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bh = 0.36
    ax.barh(y + bh / 2 + 0.01, h, height=bh, color=BLUE, label="52-week holdout")
    ax.barh(y - bh / 2 - 0.01, r, height=bh, color=ORANGE, label="Rolling 4 x 13-week backtest")
    for yy, v in zip(y, h):
        ax.text(v + 0.1, yy + bh / 2, f"{v:.1f}%", va="center", fontsize=8, color=INK_2)
    for yy, v in zip(y, r):
        ax.text(v + 0.1, yy - bh / 2, f"{v:.1f}%", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, order)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_xlabel("WAPE (lower is better)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    ax.set_title("Forecast error by model", loc="left", pad=26)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_future(df, future, path):
    ids = sorted(df["id"].unique())
    fig, axes, panel = _grid(len(ids))
    for ax, sid in zip(axes, ids):
        a = df[df["id"] == sid]
        f = future[future["id"] == sid]
        ax.plot(a["Date"], a["Weekly_Sales"], color=INK_2, lw=1.4, label="Actual")
        ax.fill_between(f["Date"], f["lower_80"], f["upper_80"], color=BLUE,
                        alpha=0.18, lw=0, label="80% interval")
        ax.plot(f["Date"], f["forecast"], color=BLUE, lw=2, label="Ensemble forecast")
        ax.set_title(f"Store {sid.split('_')[0]}, Dept {sid.split('_')[1]}", loc="left")
        _k(ax)
    _legend_panel(axes[0], panel)
    fig.suptitle("Next 52 weeks: ensemble forecast with empirical 80% interval",
                 x=0.02, ha="left", fontweight="bold")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

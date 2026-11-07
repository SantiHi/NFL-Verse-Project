"""Charts for the notebook. Each function draws one figure; the notebook shows it."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import summaries
from .detect import UNICORN_SHARE, UNICORN_TYPES
from .prepare import DRILLS, POSITION_GROUPS

TYPE_COLOR = {"Freak": "#E69F00", "Odd build": "#7B5EA7", "Red flag": "#D1495B", "Typical": "#9AA3AD"}
INK = "#22303C"
DRILL_LABELS = {"forty": "40", "bench": "Bench", "vertical": "Vertical", "broad_jump": "Broad",
                "cone": "3-cone", "shuttle": "Shuttle"}


def use_style() -> None:
    """Apply the shared chart look."""
    plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e6", "axes.axisbelow": True, "legend.frameon": False})


def sample_overview(prospects: pd.DataFrame, missing: pd.DataFrame) -> None:
    """Left: weight against 40 time for every prospect. Right: share of each drill missing, by position."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    for group in POSITION_GROUPS:
        rows = prospects[prospects["grp"] == group]
        axes[0].scatter(rows["wt"], rows["forty"], s=4, alpha=.35, label=group)
    axes[0].invert_yaxis()
    axes[0].set(title="Every prospect: weight vs 40 time", xlabel="Weight (lb)", ylabel="40-yard dash (s), faster is up")
    axes[0].legend(ncol=2, fontsize=7, markerscale=3)

    image = axes[1].imshow(missing, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    axes[1].set_xticks(range(len(DRILLS)), [DRILL_LABELS[drill] for drill in DRILLS])
    axes[1].set_yticks(range(len(POSITION_GROUPS)), POSITION_GROUPS)
    axes[1].grid(False)
    for i in range(len(POSITION_GROUPS)):
        for j in range(len(DRILLS)):
            value = missing.iloc[i, j]
            axes[1].text(j, i, f"{value:.0f}%", ha="center", va="center", fontsize=7.5,
                         color="white" if value > 55 else "#233")
    axes[1].set_title("Share of each drill missing, by position")
    plt.colorbar(image, ax=axes[1], fraction=.03)
    plt.tight_layout()


def chi2_qq(scored: pd.DataFrame, groups: tuple[str, ...] = ("WR", "DT", "CB")) -> None:
    """Q-Q plots of the full-view robust D² against the chi-squared line, one panel per position."""
    fig, axes = plt.subplots(1, len(groups), figsize=(12, 3.6))
    for ax, group in zip(axes, groups):
        table = summaries.chi2_qq(scored, group)
        k = int(scored.loc[scored["grp"] == group, "n_features"].iloc[0])
        ax.scatter(table["theoretical"], table["observed"], s=6, color="#2A6F97", alpha=.6)
        limit = table["theoretical"].max() * 1.05
        ax.plot([0, limit], [0, limit], color=INK, lw=1, ls="--", label="χ² expectation")
        cut = np.quantile(table["observed"], 1 - UNICORN_SHARE)
        ax.axhline(cut, color=TYPE_COLOR["Freak"], lw=1, label=f"top-{UNICORN_SHARE:.0%} cutoff")
        ax.set(title=f"{group}: robust D² vs χ²({k})", xlabel="χ² quantile", ylabel="Observed robust D²",
               ylim=(0, table["observed"].max() * 1.05))
        ax.legend(fontsize=7)
    plt.tight_layout()


def _scatter_types(ax, rows: pd.DataFrame, x: str, y: str) -> None:
    typical = rows[~rows["unicorn"]]
    ax.scatter(typical[x], typical[y], s=5, color=TYPE_COLOR["Typical"], alpha=.45, lw=0)
    for kind in UNICORN_TYPES:
        flagged = rows[rows["utype"] == kind]
        ax.scatter(flagged[x], flagged[y], s=24, color=TYPE_COLOR[kind], edgecolor="white", lw=.4, label=kind, zorder=3)


def _label_points(ax, rows: pd.DataFrame, x: str, y: str, fontsize: float = 6.5) -> None:
    for _, row in rows.iterrows():
        ax.annotate(row["player_name"], (row[x], row[y]), xytext=(3, 3), textcoords="offset points",
                    fontsize=fontsize, color=INK)


def unicorn_scatter(scored: pd.DataFrame, groups: tuple[str, ...] = ("WR", "EDGE", "CB"), n_labels: int = 6) -> None:
    """Weight against 40 time for a few positions, unicorns coloured by type and the best careers named."""
    fig, axes = plt.subplots(1, len(groups), figsize=(13, 4.2))
    for ax, group in zip(axes, groups):
        rows = scored[scored["grp"] == group]
        _scatter_types(ax, rows, "wt", "forty")
        _label_points(ax, rows[rows["unicorn"]].nlargest(n_labels, "w_av"), "wt", "forty")
        ax.invert_yaxis()
        ax.set(title=group, xlabel="Weight (lb)", ylabel="40-yard dash (s), faster is up")
    axes[0].legend(fontsize=7, loc="lower left")
    plt.tight_layout()

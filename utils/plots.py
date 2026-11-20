"""Figures for the Combine sample and outlier scores."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from . import summaries
from .detect import UNICORN_SHARE, UNICORN_TYPES
from .prepare import DRILLS, POSITION_GROUPS

TYPE_COLOR = {
    "Freak": "#B35C00",
    "Odd build": "#7156A4",
    "Red flag": "#BA4355",
    "Typical": "#A0ACB8",
}
TYPE_LABEL = {
    "Freak": "Favorable outlier",
    "Odd build": "Unusual build",
    "Red flag": "Unfavorable outlier",
    "Typical": "Typical",
}
TYPE_MARKER = {"Freak": "^", "Odd build": "D", "Red flag": "v", "Typical": "o"}
POSITION_LABEL = {"WR": "Wide receivers", "DT": "Defensive tackles", "EDGE": "Edge rushers", "CB": "Cornerbacks"}
DRILL_LABELS = dict(zip(DRILLS, ("40", "Bench", "Vertical", "Broad", "3-cone", "Shuttle"), strict=True))
INK = "#22303C"


def use_style() -> None:
    """Apply common settings to the notebook figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E6E6E6",
            "axes.axisbelow": True,
            "legend.frameon": False,
        }
    )


def sample_overview(prospects: pd.DataFrame, missing: pd.DataFrame) -> Figure:
    """Show weight and 40 time alongside drill missingness by position."""
    fig, (scatter, heatmap) = plt.subplots(
        1, 2, figsize=(12, 4.6), width_ratios=(1, 1.1), layout="constrained"
    )
    palette = plt.get_cmap("tab20")
    colors = palette([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 3])
    for group, color in zip(POSITION_GROUPS, colors, strict=True):
        rows = prospects.loc[prospects["grp"] == group]
        scatter.scatter(rows["wt"], rows["forty"], s=5, color=color, alpha=0.4, label=group)
    scatter.invert_yaxis()
    scatter.set(
        title="Weight and speed by position",
        xlabel="Weight (lb)",
        ylabel="40-yard dash (s, faster is higher)",
    )
    scatter.legend(ncols=2, fontsize=8, markerscale=2)

    matrix = missing.loc[POSITION_GROUPS, DRILLS].to_numpy()
    image = heatmap.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    heatmap.set_xticks(range(len(DRILLS)), [DRILL_LABELS[drill] for drill in DRILLS])
    heatmap.set_yticks(range(len(POSITION_GROUPS)), POSITION_GROUPS)
    heatmap.grid(False)
    for (row, col), value in np.ndenumerate(matrix):
        heatmap.text(
            col,
            row,
            f"{value:.0f}%",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if value > 55 else INK,
        )
    heatmap.set_title("Share of drill results missing")
    fig.colorbar(image, ax=heatmap, shrink=0.9, format="{x:.0f}%")
    return fig


def chi2_qq(scored: pd.DataFrame, groups: tuple[str, ...] = ("WR", "DT", "CB")) -> Figure:
    """Compare full-workout distances with a chi-square reference."""
    fig, axes = plt.subplots(
        1, len(groups), figsize=(3.8 * len(groups), 3.7), squeeze=False, layout="constrained"
    )
    for ax, group in zip(axes[0], groups, strict=True):
        reference = summaries.chi2_qq(scored, group)
        theoretical, observed = reference["theoretical"], reference["observed"]
        ax.scatter(theoretical, observed, s=9, color="#2A6F97", alpha=0.75)
        limit = theoretical.max()
        ax.plot([0, limit], [0, limit], color=INK, linewidth=1, linestyle="--")
        ax.axhline(observed.quantile(1 - UNICORN_SHARE), color=TYPE_COLOR["Freak"], linewidth=1)
        ax.set(
            title=POSITION_LABEL.get(group, group),
            xlabel="Chi-square quantile",
            ylabel="Observed squared distance",
        )
    axes[0, 0].legend(
        handles=[
            Line2D([], [], color=INK, linestyle="--", label="Chi-square reference"),
            Line2D([], [], color=TYPE_COLOR["Freak"], label="Full-workout 95th percentile"),
        ],
        fontsize=8,
    )
    return fig


def _scatter_types(ax: Axes, rows: pd.DataFrame) -> None:
    typical = rows.loc[~rows["unicorn"]]
    ax.scatter(typical["wt"], typical["forty"], s=6, color=TYPE_COLOR["Typical"], alpha=0.4)
    for kind in UNICORN_TYPES:
        flagged = rows.loc[rows["utype"] == kind]
        ax.scatter(
            flagged["wt"],
            flagged["forty"],
            s=35,
            color=TYPE_COLOR[kind],
            marker=TYPE_MARKER[kind],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )


def unicorn_scatter(
    scored: pd.DataFrame, groups: tuple[str, ...] = ("WR", "EDGE", "CB"), n_labels: int = 2
) -> Figure:
    """Show flagged prospects in weight and 40 time for the selected positions."""
    fig, axes = plt.subplots(
        1, len(groups), figsize=(4.1 * len(groups), 4.2), squeeze=False, layout="constrained"
    )
    for ax, group in zip(axes[0], groups, strict=True):
        rows = scored.loc[scored["grp"] == group]
        _scatter_types(ax, rows)
        labels = rows.loc[rows["unicorn"]].nlargest(n_labels, "md_pct")
        for row in labels.itertuples():
            x_offset = 5 if row.wt < rows["wt"].median() else -5
            ax.annotate(
                row.player_name,
                (row.wt, row.forty),
                xytext=(x_offset, 6),
                textcoords="offset points",
                ha="left" if x_offset > 0 else "right",
                fontsize=8,
                color=INK,
            )
        ax.invert_yaxis()
        ax.set(title=POSITION_LABEL.get(group, group), xlabel="Weight (lb)")
    axes[0, 0].set_ylabel("40-yard dash (s, faster is higher)")
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color=TYPE_COLOR[kind],
                marker=TYPE_MARKER[kind],
                linestyle="none",
                label=TYPE_LABEL[kind],
            )
            for kind in (*UNICORN_TYPES, "Typical")
        ],
        loc="outside lower center",
        ncols=4,
    )
    return fig

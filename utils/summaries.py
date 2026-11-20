"""Descriptive tables for the Combine sample and outlier scores."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .detect import TYPES
from .prepare import DRILLS, POSITION_GROUPS


def raw_overview(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rows and columns of each raw file."""
    return pd.DataFrame({"rows": [len(table) for table in raw.values()],
                         "columns": [table.shape[1] for table in raw.values()]}, index=list(raw))


def sample_census(prospects: pd.DataFrame) -> pd.DataFrame:
    """Sample size, number drafted, and average size, speed and drills done for each position group."""
    return prospects.groupby("grp").agg(
        prospects=("player_name", "size"), drafted=("drafted", "sum"),
        mean_ht=("ht", "mean"), mean_wt=("wt", "mean"), mean_forty=("forty", "mean"),
        mean_drills=("n_drills", "mean")).reindex(POSITION_GROUPS)


def drill_missing_by_position(prospects: pd.DataFrame) -> pd.DataFrame:
    """Percentage of prospects with no result for each drill, by position group."""
    return (prospects[DRILLS].isna().groupby(prospects["grp"]).mean() * 100).reindex(POSITION_GROUPS)


def fastest_for_size(adjusted: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """The prospects whose 40 time beats what their size and era predict by the most."""
    columns = ["player_name", "season", "grp", "ht", "wt", "forty", "z_ht", "z_wt",
               "adj_forty", "adj_vertical", "ath_z"]
    return adjusted[columns].sort_values("adj_forty", ascending=False).head(n)


def unicorn_counts(scored: pd.DataFrame) -> pd.DataFrame:
    """How many prospects of each type every position group has."""
    counts = pd.crosstab(scored["grp"], scored["utype"]).reindex(index=POSITION_GROUPS, columns=TYPES)
    counts.loc["All"] = counts.sum()
    return counts


def chi2_qq(scored: pd.DataFrame, group: str) -> pd.DataFrame:
    """Sorted full-view D² for one position against the chi-squared quantiles a normal model predicts."""
    rows = scored[scored["grp"] == group]
    k = int(rows["n_features"].iloc[0])
    observed = np.sort(rows["d2_full"].to_numpy())
    probabilities = (np.arange(1, len(observed) + 1) - 0.5) / len(observed)
    return pd.DataFrame({"theoretical": chi2.ppf(probabilities, df=k), "observed": observed})


def share_beyond_chi2(scored: pd.DataFrame, tail: float = 0.025, view: str = "full") -> float:
    """Share of distances above the nominal upper-tail cutoff of the chi-square reference."""
    return float((scored[f"chi2_p_{view}"] < tail).mean())

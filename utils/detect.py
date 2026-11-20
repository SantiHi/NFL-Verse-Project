"""Rank unusual Combine profiles using robust Mahalanobis distance within position."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, rankdata
from sklearn.covariance import MinCovDet

from .features import DRIVER_LABEL, SEED, SIZE_FEATURES, feature_columns

UNICORN_SHARE = 0.05   # top 5% of unicorn scores within each position = unicorns
ATHLETIC_CUT = 0.75    # athleticism composite (SD) that separates Freak / Odd build / Red flag
MCD_SUPPORT = 0.85     # share of players the robust covariance is fitted on
CORE_FEATURES = SIZE_FEATURES + ["adj_forty"]
TYPES = ["Freak", "Odd build", "Red flag", "Typical"]
UNICORN_TYPES = TYPES[:3]


def robust_mahalanobis(X: np.ndarray, support: float = MCD_SUPPORT, seed: int = SEED) -> tuple[np.ndarray, MinCovDet]:
    """Squared robust Mahalanobis distance of every row of X, plus the fitted MCD model."""
    mcd = MinCovDet(support_fraction=support, random_state=seed).fit(X)
    return mcd.mahalanobis(X), mcd


def percentile_rank(values) -> np.ndarray:
    """Return fractional ranks, using the average rank for ties."""
    return rankdata(values) / len(values)


def biggest_driver(X: np.ndarray, mcd: MinCovDet, names: list[str]) -> np.ndarray:
    """Which feature contributes most to each player's distance: (x-μ)_j · [Σ⁻¹(x-μ)]_j."""
    centred = X - mcd.location_
    contribution = centred * (centred @ mcd.get_precision())
    return np.array(names)[contribution.argmax(axis=1)]


def unicorn_type(unicorn: pd.Series, ath_z: pd.Series, athletic_cut: float = ATHLETIC_CUT) -> pd.Series:
    """Typical for everyone not flagged; Freak, Red flag or Odd build by the athleticism composite."""
    kind = np.select([~unicorn, ath_z >= athletic_cut, ath_z <= -athletic_cut],
                     ["Typical", "Freak", "Red flag"], default="Odd build")
    return pd.Series(kind, index=unicorn.index)


def score_prospects(prospects: pd.DataFrame, unicorn_share: float = UNICORN_SHARE,
                    athletic_cut: float = ATHLETIC_CUT, support: float = MCD_SUPPORT,
                    seed: int = SEED) -> pd.DataFrame:
    """Score every prospect within position and flag the highest-ranked profiles.

    Takes the table from features.adjust_drills() and returns it with these columns added:
      d2_full, d2_core          squared robust Mahalanobis distances
      pct_full, pct_core        percentile ranks within position
      chi2_p_full, chi2_p_core  nominal chi-square reference tail areas
      driver, n_features        largest full-view contributor and number of full-view inputs
      md_score, md_pct          maximum view percentile and its within-position rank
      unicorn, utype            outlier flag and category
    """
    scores = []
    for _, group in prospects.groupby("grp"):
        views = {"full": feature_columns(group), "core": CORE_FEATURES}
        group_scores = pd.DataFrame(index=group.index)
        for view, columns in views.items():
            measurements = group[columns].to_numpy()
            distance, mcd = robust_mahalanobis(measurements, support, seed)
            group_scores[f"d2_{view}"] = distance
            group_scores[f"pct_{view}"] = percentile_rank(distance)
            group_scores[f"chi2_p_{view}"] = chi2.sf(distance, df=len(columns))
            if view == "full":
                group_scores["driver"] = biggest_driver(measurements, mcd, columns)
                group_scores["n_features"] = len(columns)
        group_scores["md_score"] = group_scores[["pct_full", "pct_core"]].max(axis=1)
        group_scores["md_pct"] = percentile_rank(group_scores["md_score"])
        scores.append(group_scores)

    scored = prospects.join(pd.concat(scores))
    scored["unicorn"] = scored["md_pct"] > 1 - unicorn_share
    scored["utype"] = unicorn_type(scored["unicorn"], scored["ath_z"], athletic_cut)
    scored["driver"] = scored["driver"].map(DRIVER_LABEL)
    return scored

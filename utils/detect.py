"""Score how unusual each prospect's profile is for his position: robust Mahalanobis distance.

For each prospect we ask how far his profile is from the centre of his position group, in units
that respect how the measurements normally move together. Fast players usually jump well, so a
player who is fast but not explosive, or heavy and fast, gets a large distance: unusual
combinations are what make a unicorn.

The centre and covariance come from the Minimum Covariance Determinant estimator, fitted on the
tightest 85% of the position only, so the outliers themselves cannot stretch the fit and hide.

Two views per position:
  full  height, weight and every adjusted drill (some imputed)
  core  height, weight and the adjusted 40 only, all measured for everyone; catches players
        whose only recorded drill is the 40
Each view's distance becomes a percentile within the position. The unicorn score is the higher
of the two percentiles, re-ranked; the top 5% are unicorns. Unicorns are typed by the athleticism
composite: Freak (well above what size predicts), Red flag (well below) or Odd build (between).

Under normality D² follows a chi-squared distribution with as many degrees of freedom as features;
the tail probability is kept as `chi2_p_<view>` for a model-based robustness check.

Nothing here sees draft position or career outcomes.
"""
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
    """Rank within the group as a share of the group (ties share their average rank)."""
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
    """Score every prospect inside his position group and flag the unicorns.

    Takes the table from features.adjust_drills() and returns it with these columns added:
      d2_full, d2_core          squared robust Mahalanobis distance in each view
      pct_full, pct_core        that distance's percentile within the position
      chi2_p_full, chi2_p_core  tail probability if profiles were multivariate normal
      driver                    the feature contributing most to the full-view distance
      n_features                how many features the full view uses for this position
      md_score, md_pct          the higher of the two percentiles, and its re-ranked percentile
      unicorn                   True for the top `unicorn_share` by md_pct
      utype                     Freak, Odd build, Red flag or Typical
    """
    scores = []
    for _, group in prospects.groupby("grp"):
        views = {"full": feature_columns(group), "core": CORE_FEATURES}
        s = pd.DataFrame(index=group.index)
        for view, columns in views.items():
            X = group[columns].to_numpy()
            d2, mcd = robust_mahalanobis(X, support, seed)
            s[f"d2_{view}"] = d2
            s[f"pct_{view}"] = percentile_rank(d2)
            s[f"chi2_p_{view}"] = chi2.sf(d2, df=len(columns))
            if view == "full":
                s["driver"] = biggest_driver(X, mcd, columns)
                s["n_features"] = len(columns)
        s["md_score"] = s[["pct_full", "pct_core"]].max(axis=1)
        s["md_pct"] = percentile_rank(s["md_score"])
        scores.append(s)

    scored = prospects.join(pd.concat(scores))
    scored["unicorn"] = scored["md_pct"] > 1 - unicorn_share
    scored["utype"] = unicorn_type(scored["unicorn"], scored["ath_z"], athletic_cut)
    scored["driver"] = scored["driver"].map(DRIVER_LABEL)
    return scored

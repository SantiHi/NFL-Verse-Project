"""Size- and era-adjusted drill scores: how a prospect tested compared with what his height,
weight and combine year predict for his position.

A 4.9-second 40 is slow for a receiver and remarkable for a nose tackle, and even within a
position heavier players run slower. So, inside each position group:
  1. Missing drills are filled by iterative imputation (each drill predicted from height, weight
     and the other drills). A drill fewer than 25% of the position did is left out, for example
     the bench press for quarterbacks.
  2. Each drill is regressed on height, weight, weight squared, year and a 2021 indicator (no
     combine was held that year; its numbers are hand-timed pro-day results), fitted on the
     prospects who actually did the drill. The residual says how much better or worse a player
     tested than his size and era predict.
  3. Residuals are scaled robustly (median and MAD rather than mean and SD, so the outliers we
     are hunting do not distort the yardstick) and signed so that positive always means better.
     Height and weight are scaled the same way.
  4. The athleticism composite `ath_z` is the average of the adjusted drill scores, standardised
     within the position.

Imputed values sit near the middle of the distribution, so imputation can only make a player
look less unusual, not more.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (unlocks IterativeImputer)
from sklearn.impute import IterativeImputer

from .prepare import DRILLS

SEED = 42                  # fixes every random step so results reproduce exactly
MIN_DRILL_COVERAGE = 0.25  # a drill is skipped for a position if fewer than 25% of its prospects did it
IMPUTER_ITERATIONS = 25
ERA_BASE_YEAR = 2013       # the year term counts classes from here
PRO_DAY_YEAR = 2021        # no combine was held; gets its own term
# Timed drills are better when lower; counted and measured drills are better when higher.
HIGHER_IS_BETTER = {"forty": False, "bench": True, "vertical": True, "broad_jump": True,
                    "cone": False, "shuttle": False}

SIZE_FEATURES = ["z_ht", "z_wt"]
ADJUSTED = ["adj_" + drill for drill in DRILLS]
DRIVER_LABEL = {"z_ht": "Height", "z_wt": "Weight", "adj_forty": "40 speed", "adj_bench": "Bench",
                "adj_vertical": "Vertical", "adj_broad_jump": "Broad jump", "adj_cone": "3-cone",
                "adj_shuttle": "Shuttle"}


def robust_z(values: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """(x - median) / (1.4826 * MAD); the location and scale come from `reference` if given."""
    ref = values if reference is None else reference
    median = np.median(ref)
    mad = 1.4826 * np.median(np.abs(ref - median))
    return (values - median) / mad


def drills_covered(group: pd.DataFrame, min_coverage: float = MIN_DRILL_COVERAGE) -> list[str]:
    """The drills enough of one position group did for them to be adjusted and imputed."""
    return [drill for drill in DRILLS if group[drill].notna().mean() >= min_coverage]


def feature_columns(group: pd.DataFrame) -> list[str]:
    """Height, weight and every adjusted drill available for one position group."""
    return SIZE_FEATURES + ["adj_" + drill for drill in drills_covered(group)]


def adjust_position(group: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Impute missing drills and return size- and era-adjusted, oriented drill scores for one position.

    Columns: z_ht, z_wt, adj_<drill> for each of the six drills (NaN for a drill the position skips)
    and the athleticism composite ath_z.
    """
    drills = drills_covered(group)
    raw = group[["ht", "wt"] + drills]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        imputer = IterativeImputer(random_state=seed, max_iter=IMPUTER_ITERATIONS)
        filled = pd.DataFrame(imputer.fit_transform(raw), columns=raw.columns, index=group.index)

    design = np.column_stack([np.ones(len(group)), filled["ht"], filled["wt"], filled["wt"] ** 2,
                              group["season"] - ERA_BASE_YEAR, (group["season"] == PRO_DAY_YEAR).astype(float)])
    out = pd.DataFrame(index=group.index)
    out["z_ht"] = robust_z(filled["ht"].to_numpy())
    out["z_wt"] = robust_z(filled["wt"].to_numpy())
    for drill in DRILLS:
        if drill not in drills:
            out["adj_" + drill] = np.nan
            continue
        measured = group[drill].notna().to_numpy()
        beta, *_ = np.linalg.lstsq(design[measured], filled[drill].to_numpy()[measured], rcond=None)
        residual = filled[drill].to_numpy() - design @ beta
        sign = 1 if HIGHER_IS_BETTER[drill] else -1
        out["adj_" + drill] = sign * robust_z(residual, residual[measured])
    athleticism = out[["adj_" + drill for drill in drills]].mean(axis=1)
    out["ath_z"] = (athleticism - athleticism.mean()) / athleticism.std()
    return out


def adjust_drills(prospects: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """The sample with adjusted drill scores and the athleticism composite added, position by position."""
    adjusted = pd.concat([adjust_position(group, seed) for _, group in prospects.groupby("grp")])
    return prospects.join(adjusted)

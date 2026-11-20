"""Adjust Combine drills for size and era within each position."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (unlocks IterativeImputer)
from sklearn.impute import IterativeImputer

from .prepare import DRILLS

SEED = 42
MIN_DRILL_COVERAGE = 0.25  # a drill is skipped for a position if fewer than 25% of its prospects did it
IMPUTER_ITERATIONS = 25
ERA_BASE_YEAR = 2013       # the year term counts classes from here
PRO_DAY_YEAR = 2021        # no combine was held; gets its own term
# Timed drills are better when lower; counted and measured drills are better when higher.
HIGHER_IS_BETTER = {"forty": False, "bench": True, "vertical": True, "broad_jump": True,
                    "cone": False, "shuttle": False}

SIZE_FEATURES = ["z_ht", "z_wt"]
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
    # Keep the iteration budget fixed. Some position fits reach this limit.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        imputer = IterativeImputer(random_state=seed, max_iter=IMPUTER_ITERATIONS)
        filled = pd.DataFrame(imputer.fit_transform(raw), columns=raw.columns, index=group.index)

    design = np.column_stack(
        [
            np.ones(len(group)),
            filled["ht"],
            filled["wt"],
            filled["wt"] ** 2,
            group["season"] - ERA_BASE_YEAR,
            group["season"].eq(PRO_DAY_YEAR).astype(float),
        ]
    )
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

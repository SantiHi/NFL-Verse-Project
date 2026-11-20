"""Draft selection and career value for the Combine sample."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

from .detect import TYPES
from .prepare import FIRST_DRAFT_CLASS

LAST_DRAFT_CLASS = 2022
OLDER_LAST_CLASS = 2018
REFERENCE_YEAR = 2011
BOOTSTRAP_SEED = 424

TYPICAL = "Typical"


def draft_selection(prospects: pd.DataFrame, group: str = "utype") -> pd.DataFrame:
    """Count selections and summarize pick number among drafted prospects."""
    return prospects.groupby(group).agg(
        prospects=("drafted", "size"),
        drafted=("drafted", "sum"),
        draft_rate=("drafted", "mean"),
        median_pick=("draft_ovr", "median"),
    )


def eligible_picks(scored: pd.DataFrame, last_year: int = LAST_DRAFT_CLASS) -> pd.DataFrame:
    """Return the drafted Combine prospects in the career-value analysis."""
    eligible = scored["drafted"] & scored["draft_year"].between(FIRST_DRAFT_CLASS, last_year)
    return scored.loc[eligible].copy()


def _benchmark_features(picks: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "log_pick": np.log(picks["draft_ovr"]),
            "year": (picks["draft_year"] - REFERENCE_YEAR) / 10,
            "grp": picks["grp"],
        },
        index=picks.index,
    )


def _benchmark_model(flexible: bool) -> Pipeline:
    pick_term = SplineTransformer(n_knots=4, include_bias=False) if flexible else "passthrough"
    year_term = (
        SplineTransformer(n_knots=4, include_bias=False, extrapolation="linear")
        if flexible else "passthrough"
    )
    terms = ColumnTransformer(
        [
            ("pick", pick_term, ["log_pick"]),
            ("year", year_term, ["year"]),
            ("position", OneHotEncoder(handle_unknown="error"), ["grp"]),
        ]
    )
    return make_pipeline(terms, Ridge(alpha=1.0))


def grade_picks(
    picks: pd.DataFrame, *, missing_as_zero: bool = False, flexible: bool = True
) -> pd.DataFrame:
    """Predict each class from other classes and keep any missing actual WAV explicit."""
    actual = picks["w_av"].fillna(0) if missing_as_zero else picks["w_av"]
    observed = actual.notna()
    data = _benchmark_features(picks)
    expected = pd.Series(index=picks.index, dtype=float)
    for year in sorted(picks["draft_year"].unique()):
        held_out = picks["draft_year"].eq(year)
        training = observed & ~held_out
        model = _benchmark_model(flexible).fit(data.loc[training], actual.loc[training])
        expected.loc[held_out] = model.predict(data.loc[held_out])
    return picks.assign(actual_wav=actual, expected_wav=expected, excess_wav=actual - expected)


def expected_curve(picks: pd.DataFrame) -> pd.DataFrame:
    """Fit with all observed outcomes and hold the reference mix of positions and years fixed."""
    recorded = picks.loc[picks["w_av"].notna()]
    model = _benchmark_model(flexible=True)
    model.fit(_benchmark_features(recorded), recorded["w_av"])

    mix = recorded.groupby(["grp", "draft_year"]).size().rename("weight").reset_index()
    mix["weight"] = mix["weight"] / mix["weight"].sum()
    pick_numbers = pd.DataFrame({"draft_ovr": np.arange(1, 1 + int(recorded["draft_ovr"].max()))})
    grid = pick_numbers.merge(mix, how="cross")
    grid["weighted_wav"] = grid["weight"] * model.predict(_benchmark_features(grid))
    return grid.groupby("draft_ovr", as_index=False)["weighted_wav"].sum().rename(
        columns={"weighted_wav": "expected_wav"}
    )


def calibration(graded: pd.DataFrame, by: str) -> pd.DataFrame:
    """Compare actual and held-out expected WAV on the same recorded outcomes."""
    observed = graded.loc[graded["actual_wav"].notna()]
    return observed.groupby(by).agg(
        players=("excess_wav", "size"),
        actual=("actual_wav", "mean"),
        expected=("expected_wav", "mean"),
        error=("excess_wav", "mean"),
    )


def accuracy(graded: pd.DataFrame) -> dict[str, float]:
    """Summarize held-out prediction errors for picks with observed WAV."""
    errors = graded["excess_wav"].dropna()
    return {"MAE": errors.abs().mean(), "RMSE": np.sqrt(errors.pow(2).mean())}


def career_summary(graded: pd.DataFrame, draws: int = 4000) -> pd.DataFrame:
    """Summarize WAV and estimate class-bootstrap intervals for gaps from typical prospects."""
    observed = graded.loc[graded["excess_wav"].notna()]
    eligible = graded.groupby("utype").size().rename("eligible")
    summary = observed.groupby("utype").agg(
        recorded=("actual_wav", "size"),
        actual=("actual_wav", "mean"),
        expected=("expected_wav", "mean"),
        excess=("excess_wav", "mean"),
    )
    summary = eligible.to_frame().join(summary).reindex(TYPES)
    by_class = observed.groupby(["draft_year", "utype"])["excess_wav"].agg(["sum", "size"])
    sums = by_class["sum"].unstack(fill_value=0).reindex(columns=TYPES, fill_value=0).to_numpy()
    counts = by_class["size"].unstack(fill_value=0).reindex(columns=TYPES, fill_value=0).to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, len(sums), size=(draws, len(sums)))
    totals, n = sums[samples].sum(axis=1), counts[samples].sum(axis=1)
    means = np.divide(totals, n, out=np.full(totals.shape, np.nan), where=n > 0)
    gaps = means - means[:, [TYPES.index(TYPICAL)]]
    summary["gap"] = summary["excess"] - summary.loc[TYPICAL, "excess"]
    summary["lower"], summary["upper"] = np.nanquantile(gaps, [0.025, 0.975], axis=0)
    return summary


def grade_rates(graded: pd.DataFrame, thresholds: tuple[int, ...] = (5, 10, 20)) -> pd.DataFrame:
    """Describe above- and below-expectation rates for several WAV cutoffs."""
    observed = graded.dropna(subset=["excess_wav"])
    tables = {}
    for threshold in thresholds:
        indicators = observed.assign(
            steal=observed["excess_wav"].ge(threshold),
            bust=observed["excess_wav"].le(-threshold),
        )
        tables[threshold] = indicators.groupby("utype")[["steal", "bust"]].mean().reindex(TYPES)
    return pd.concat(tables, names=["WAV cutoff", "utype"])


def individual_influence(graded: pd.DataFrame) -> pd.DataFrame:
    """Find the single removal that most changes the favorable-versus-typical mean gap."""
    observed = graded.loc[
        graded["utype"].isin(["Freak", TYPICAL]) & graded["excess_wav"].notna()
    ].copy()
    values = observed.groupby("utype")["excess_wav"]
    averages, sizes = values.transform("mean"), values.transform("size")
    delta = (averages - observed["excess_wav"]) / (sizes - 1)
    observed["change_in_gap"] = delta.where(observed["utype"].eq("Freak"), -delta)
    index = observed["change_in_gap"].abs().idxmax()
    return observed.loc[[index], ["player_name", "draft_year", "utype", "change_in_gap"]]


def favorable_gaps(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare the favorable-versus-typical gap across samples."""
    columns = ["recorded", "gap", "lower", "upper"]
    gaps = pd.DataFrame({
        label: career_summary(graded).loc["Freak", columns]
        for label, graded in samples.items()
    }).T
    return gaps.astype({"recorded": int})

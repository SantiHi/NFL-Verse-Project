"""Clean the raw nflverse files into the modelling sample of draft prospects.

Inputs (data/raw/, the October 2026 snapshot of https://github.com/nflverse/nflverse-data):
    combine.csv      one row per combine participant, 2000-2026: size, six drills, draft slot
    draft_picks.csv  one row per draft pick since 1980: pick number, team, career value
    players.csv      NFL roster records, used only for the position a player held in the NFL

`build_sample()` runs every cleaning step and returns the sample together with a log of how
many rows each step removed, so the notebook can show the preprocessing rather than describe it.
Run `python -m utils.prepare` to print that log and write data/processed/prospects.csv.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# nflverse updates the files during the season; these identify the October 2026 snapshot.
SNAPSHOT_SHA256 = {
    "combine.csv": "5648886b1bb9de8a0c42b05ee7b1d9cbe58897676caaa2f6a4a3d3c9228235bb",
    "draft_picks.csv": "769633acc15f9539cf3e57a8ed2bf95c6392b038e765cef60a4542d6da3a03f8",
    "players.csv": "7ea1f10f0c3b0bf5c026a4b5935c6d01e4a72864a3122594a6df493b5bf29d97",
}

DRILLS = ["forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]
CORE_MEASUREMENTS = ["ht", "wt", "forty"]  # every prospect in the sample has all three
FIRST_DRAFT_CLASS = 2000
UNLINKED_CLASS = 2026  # newest class: the combine file carries no draft slots for it yet
POSITION_GROUPS = ["QB", "RB", "WR", "TE", "OT", "IOL", "DT", "EDGE", "LB", "CB", "S"]

# Combine labels that map to exactly one position group.
DIRECT_GROUP = {
    "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE",
    "OT": "OT", "OG": "IOL", "G": "IOL", "C": "IOL",
    "DT": "DT", "DE": "EDGE", "EDGE": "EDGE",
    "ILB": "LB", "LB": "LB",
    "CB": "CB", "S": "S", "SAF": "S",
}
# Generic labels that could be two groups; resolved in assign_position_group().
AMBIGUOUS_LABELS = ["OLB", "OL", "DL", "DB"]
# Kickers, punters, long snappers and fullbacks get no group; "CB/WR" fits none.
EXCLUDED_LABELS = ["K", "P", "LS", "FB", "CB/WR"]

# Size cut-offs, used only when a label is generic and no NFL or drafted position is known.
EDGE_MIN_WEIGHT = 245       # an "OLB" this heavy is an edge rusher, lighter is an off-ball linebacker
TACKLE_MIN_HEIGHT = 77      # inches; an "OL" this tall is a tackle
INTERIOR_DL_MIN_WEIGHT = 285
SAFETY_MIN_WEIGHT = 200

# A weight outside these bounds means the row's position label is wrong, not that the player is
# a rare athlete (for example a 195-lb "offensive lineman"). Bounds are (min, max) in lb.
PLAUSIBLE_WEIGHT = {"OT": (270, np.inf), "IOL": (270, np.inf), "LB": (210, 275)}

# Broad families, used to decide which of two same-named prospects a draft pick belongs to.
POSITION_FAMILY = {
    **dict.fromkeys(["QB", "RB", "FB", "WR", "TE"], "skill"),
    **dict.fromkeys(["T", "OT", "G", "OG", "C", "OL"], "offensive line"),
    **dict.fromkeys(["DT", "NT", "DE", "DL", "EDGE", "LB", "OLB", "ILB"], "front seven"),
    **dict.fromkeys(["CB", "DB", "S", "SAF", "FS", "SS"], "secondary"),
}

# Franchises that moved keep one identity (Pro-Football-Reference team codes).
RELOCATED = {"STL": "LAR", "SDG": "LAC", "OAK": "LVR"}

DATA_ERROR_COLUMNS = ["player_name", "season", "pos", "grp", "ht", "wt", "forty"]


# --------------------------------------------------------------------------- loading

def load_raw(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Read the three raw files exactly as downloaded."""
    return {
        "combine": pd.read_csv(raw_dir / "combine.csv"),
        "draft_picks": pd.read_csv(raw_dir / "draft_picks.csv"),
        "players": pd.read_csv(raw_dir / "players.csv", low_memory=False),
    }


def verify_snapshot(raw_dir: Path = RAW_DIR) -> None:
    """Check that the raw files are the October 2026 snapshot; raise if any differs."""
    for name, expected in SNAPSHOT_SHA256.items():
        actual = hashlib.sha256((raw_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{name} is not the October 2026 snapshot")


# --------------------------------------------------------------------------- small helpers

def height_to_inches(height: object) -> float:
    """Convert a feet-inches string such as "6-4" to inches (76); anything unreadable becomes NaN."""
    try:
        feet, inches = str(height).split("-")
        return 12 * int(feet) + int(inches)
    except ValueError:
        return np.nan


# --------------------------------------------------------------------------- cleaning steps

def link_class_by_name(combine: pd.DataFrame, draft: pd.DataFrame,
                       season: int = UNLINKED_CLASS) -> tuple[pd.DataFrame, int]:
    """Fill in draft slots for a class the combine file has not been linked to yet.

    nflverse adds the draft columns to the combine file some time after each draft. Prospects of
    `season` are matched to that year's picks on exact player name. Returns the combine table and
    the number of prospects that received a draft slot.
    """
    combine = combine.copy()
    picks = (draft[draft["season"] == season].drop_duplicates("pfr_player_name")
             .set_index("pfr_player_name"))
    in_class = (combine["season"] == season) & combine["player_name"].isin(picks.index)
    names = combine.loc[in_class, "player_name"]
    combine.loc[in_class, "draft_year"] = season
    combine.loc[in_class, "draft_round"] = names.map(picks["round"])
    combine.loc[in_class, "draft_ovr"] = names.map(picks["pick"])
    return combine, int(in_class.sum())


def attach_draft_outcomes(combine: pd.DataFrame, draft: pd.DataFrame) -> pd.DataFrame:
    """Add the pick's team, drafted position and career value, matched on (draft year, overall pick)."""
    picks = (draft.loc[draft["season"] >= FIRST_DRAFT_CLASS, ["season", "pick", "team", "position", "w_av"]]
             .rename(columns={"season": "draft_year", "pick": "draft_ovr", "position": "draft_pos"}))
    return combine.merge(picks, on=["draft_year", "draft_ovr"], how="left", validate="many_to_one")


def attach_nfl_position(prospects: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Add the position a player held on an NFL roster (missing for players who never signed)."""
    nfl = (players[["pfr_id", "position"]].dropna().drop_duplicates("pfr_id")
           .rename(columns={"position": "nfl_pos"}))
    return prospects.merge(nfl, on="pfr_id", how="left", validate="many_to_one")


def drop_shared_pick_namesakes(prospects: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Resolve draft picks that two same-named prospects both claim.

    The pick stays with the prospect whose combine position is in the same family as the drafted
    position (the first such row if several match). The other rows are dropped because their real
    draft outcome cannot be recovered.
    """
    pick = ["draft_year", "draft_ovr"]
    shared = prospects["draft_ovr"].notna() & prospects.duplicated(pick, keep=False)
    same_family = prospects["pos"].map(POSITION_FAMILY) == prospects["draft_pos"].map(POSITION_FAMILY)
    claimants = prospects[shared].assign(match=same_family[shared].astype(int))
    keep = claimants.sort_values("match", ascending=False, kind="stable").drop_duplicates(pick).index
    drop = shared & ~prospects.index.isin(keep)
    return prospects[~drop].copy(), int(drop.sum())


def assign_position_group(prospects: pd.DataFrame) -> pd.Series:
    """Map each combine position label to one of the eleven position groups.

    Most labels map directly. Generic labels (OL, DL, DB, OLB) are resolved with the position the
    player held in the NFL, or the drafted position, and by body size when neither is known.
    Specialists and fullbacks get no group.
    """
    pos, height, weight = prospects["pos"], prospects["ht"], prospects["wt"]
    pro_pos = prospects["nfl_pos"].where(prospects["nfl_pos"].notna(), prospects["draft_pos"])
    group = pos.map(DIRECT_GROUP)  # keeps the label column's dtype; NaN where no rule applies yet

    olb = pos == "OLB"
    group[olb] = np.where(weight[olb] >= EDGE_MIN_WEIGHT, "EDGE", "LB")

    ol = pos == "OL"
    by_size = np.where(height[ol] >= TACKLE_MIN_HEIGHT, "OT", "IOL")
    group[ol] = np.select([pro_pos[ol].isin(["T", "OT"]), pro_pos[ol].isin(["G", "C"])],
                          ["OT", "IOL"], default=by_size)

    dl = pos == "DL"
    by_size = np.where(weight[dl] >= INTERIOR_DL_MIN_WEIGHT, "DT", "EDGE")
    group[dl] = np.select([pro_pos[dl].isin(["DT", "NT"]), pro_pos[dl].isin(["DE", "OLB", "LB"])],
                          ["DT", "EDGE"], default=by_size)

    db = pos == "DB"
    by_size = np.where(weight[db] >= SAFETY_MIN_WEIGHT, "S", "CB")
    group[db] = np.select([pro_pos[db] == "CB", pro_pos[db].isin(["S", "SAF", "FS", "SS"])],
                          ["CB", "S"], default=by_size)

    unknown = set(pos[group.isna()]) - set(EXCLUDED_LABELS)
    if unknown:
        raise ValueError(f"combine position labels with no rule: {sorted(unknown)}")
    return group


def implausible_size(prospects: pd.DataFrame) -> pd.Series:
    """Flag rows whose weight is impossible for the listed position (a mislabelled row)."""
    flagged = pd.Series(False, index=prospects.index)
    for group, (low, high) in PLAUSIBLE_WEIGHT.items():
        in_group = prospects["grp"] == group
        flagged |= in_group & ((prospects["wt"] < low) | (prospects["wt"] > high))
    return flagged


# --------------------------------------------------------------------------- pipeline

class Sample(NamedTuple):
    """What build_sample() returns."""
    prospects: pd.DataFrame    # the modelling sample, one row per prospect
    log: pd.DataFrame          # rows remaining and removed after each step
    data_errors: pd.DataFrame  # rows removed because their size is impossible for the position
    linked_by_name: int        # prospects of the newest class given a draft slot by name match


def build_sample(raw: dict[str, pd.DataFrame] | None = None, raw_dir: Path = RAW_DIR) -> Sample:
    """Run every cleaning step on the raw files (loaded from `raw_dir` if `raw` is not given).

    Steps: convert height to inches; link the newest class to its draft picks by name; attach each
    pick's team, drafted position and career value, and the NFL roster position; resolve namesakes
    sharing one pick; assign position groups; require height, weight and a 40 time; remove sizes
    impossible for the position. The rows that survive get a stable id `pid`, `drafted`,
    `franchise` and `n_drills`.
    """
    if raw is None:
        raw = load_raw(raw_dir)
    combine, draft, players = raw["combine"], raw["draft_picks"], raw["players"]
    log = []

    df = combine.copy()
    df["ht"] = df["ht"].map(height_to_inches)
    log.append((f"Combine records, {df['season'].min()}–{df['season'].max()}", len(df)))

    df, linked_by_name = link_class_by_name(df, draft)
    df = attach_draft_outcomes(df, draft)
    df = attach_nfl_position(df, players)

    df, _ = drop_shared_pick_namesakes(df)
    log.append(("Drop namesakes sharing one draft pick", len(df)))

    df["grp"] = assign_position_group(df)
    df = df[df["grp"].notna()]
    log.append((f"Keep the {len(POSITION_GROUPS)} position groups", len(df)))

    df = df.dropna(subset=CORE_MEASUREMENTS)
    log.append(("Require height, weight and 40 time", len(df)))

    impossible = implausible_size(df)
    data_errors = df.loc[impossible, DATA_ERROR_COLUMNS]
    df = df[~impossible].reset_index(drop=True)
    df["pid"] = np.arange(len(df))  # stable prospect id
    log.append(("Remove sizes impossible for the position", len(df)))

    df["drafted"] = df["draft_ovr"].notna()
    df["franchise"] = df["team"].replace(RELOCATED)
    df["n_drills"] = df[DRILLS].notna().sum(axis=1)

    log = pd.DataFrame(log, columns=["Step", "Rows remaining"])
    log["Removed"] = (-log["Rows remaining"].diff()).fillna(0).astype(int)
    return Sample(df, log, data_errors, linked_by_name)


if __name__ == "__main__":
    verify_snapshot()
    sample = build_sample()
    print(f"{UNLINKED_CLASS} prospects matched to a pick by name: {sample.linked_by_name}")
    print(sample.log.to_string(index=False))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sample.prospects.to_csv(PROCESSED_DIR / "prospects.csv", index=False)
    print(f"\nwrote {len(sample.prospects):,} rows to {PROCESSED_DIR / 'prospects.csv'}")

# Do unique players pay off?

COS 424 course project. We look back at NFL draft classes since 2000 and ask whether prospects whose
combine measurements are unusual for their position return more than their draft slot normally does.

## Repository layout

| path | contents |
|---|---|
| `COS_424_Milestone_2_PRELIM_ANALYSIS.ipynb` | the main notebook: data, the modelling sample, adjusted features, the detection model |
| `utils/prepare.py` | loads the raw files and builds the modelling sample, logging what each step removed |
| `utils/features.py` | size- and era-adjusted drill scores, with missing drills imputed, and the athleticism composite |
| `utils/detect.py` | the detection model: robust (MCD) Mahalanobis distance in two views; flags the unicorns and types them |
| `utils/summaries.py` | tables used by the notebook |
| `utils/plots.py` | charts used by the notebook |
| `data/raw/` | the three nflverse files, as downloaded (the October 2026 snapshot; the notebook checks their checksums) |
| `data/processed/prospects.csv` | the modelling sample written by `utils/prepare.py` |
| `docs/` | the project introduction slides |

The notebook contains no implementation logic of its own: every table and chart comes from a function in `utils/`.

## Reproducing the analysis

Developed with Python 3.12.

```
pip install -r requirements.txt
python -m utils.prepare     # optional: rebuilds data/processed/prospects.csv and prints the sample log
```

Then open `COS_424_Milestone_2_PRELIM_ANALYSIS.ipynb` and run every cell from the top (under a minute).

## Data

All three files come from the open [nflverse](https://github.com/nflverse/nflverse-data) project,
downloaded in October 2026:

- `combine.csv`: one row per NFL Combine participant, 2000-2026 (size, six drills, draft slot)
- `draft_picks.csv`: one row per draft pick since 1980, with career Approximate Value
- `players.csv`: NFL roster records, used for the position a player held in the NFL

# Do unique players pay off?

COS 424 course project. We look back at NFL draft classes since 2000 and ask whether prospects whose
combine measurements are unusual for their position return more than their draft slot normally does.

## Repository layout

| path | contents |
|---|---|
| `initial_analysis.ipynb` | the main notebook: data, preprocessing, descriptive statistics, exploration, first model |
| `utils/prepare.py` | loads the raw files and builds the cleaned table of prospects |
| `utils/summaries.py` | descriptive tables used by the notebook |
| `utils/plots.py` | charts used by the notebook |
| `utils/detect.py` | the detection model: Mahalanobis distance on size and speed, with Isolation Forest and Local Outlier Factor kept for comparison |
| `data/raw/` | the three nflverse files, as downloaded |
| `data/processed/prospects.csv` | cleaned table written by `utils/prepare.py` |
| `docs/` | the project introduction slides and precept notes |
| `archive/` | the October prototype (`unicorn_replication_package/`, its paper and output); not used by the notebook |

The notebook contains no implementation logic of its own: every table and chart comes from a function in `utils/`.

## Reproducing the analysis

Developed with Python 3.12.

```
pip install -r requirements.txt
python -m utils.prepare     # optional: rebuilds data/processed/prospects.csv and prints the cleaning log
```

Then open `initial_analysis.ipynb` and run every cell from the top (about 20 seconds).

## Data

All three files come from the open [nflverse](https://github.com/nflverse/nflverse-data) project,
downloaded in October 2026:

- `combine.csv`: one row per NFL Combine participant, 2000-2026 (size, six drills, draft slot)
- `draft_picks.csv`: one row per draft pick since 1980, with career Approximate Value
- `players.csv`: NFL roster records, used for the position a player held in the NFL

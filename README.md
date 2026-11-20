# Career Value of NFL Combine Outliers Relative to Draft Position

Santiago Criado, Phu Duong, Sri Indla, and Steven Goetz

COS 424: Reasoning with Data · Princeton University

We study whether NFL teams draft prospects with unusual physical and athletic profiles more often or earlier than typical prospects, and whether those players produce more career value than expected for their pick, position, and draft class. [The analysis notebook](nfl_draft_analysis.ipynb) documents the data and the method we use to identify outliers within each position.

## Reproduce the analysis

Run the following commands from the repository root using Python 3.12.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter execute --inplace --timeout=120 nfl_draft_analysis.ipynb
```

The last command runs the notebook from a fresh kernel and saves the tables and figures. To run it interactively, open the notebook in a Jupyter-compatible editor and select the interpreter from `.venv`.

We use an October 2026 snapshot of the [nflverse](https://github.com/nflverse/nflverse-data) Combine, Draft Picks, and Players tables. The notebook verifies the checksums of the files in [`data/raw/`](data/raw/) before loading them. The code for sample construction, outlier detection, tables, and figures is in [`utils/`](utils/).

Run `python -m utils.prepare` to export the cleaned prospect sample to `data/processed/prospects.csv`.

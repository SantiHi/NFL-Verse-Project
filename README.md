# Do unusual NFL Combine prospects pay off?

A Princeton COS 424 project by Santiago Criado, Phu Duong, Sri Indla, and Steven Goetz. We ask how often and how early teams draft prospects with unusual Combine profiles, and how much career value those prospects deliver relative to what their picks typically return.

Start with [the analysis notebook](nfl_draft_analysis.ipynb). It builds the prospect sample, explains the two-view outlier score, and shows which prospects the model flags.

## Reproduce the analysis

Run these commands from the repository root with Python 3.12 installed.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter execute --inplace --timeout=120 nfl_draft_analysis.ipynb
```

The command runs the notebook from a fresh kernel and refreshes its saved tables and figures. You can also open the notebook in a Jupyter-compatible editor, select the `.venv` interpreter, and run all cells.

The notebook verifies the October 2026 [nflverse snapshot](https://github.com/nflverse/nflverse-data) in [`data/raw/`](data/raw/) before reading it. The three CSVs provide Combine measurements, draft picks with career Weighted Approximate Value (WAV), and player roster positions used to resolve broad Combine position labels. Code for data preparation, outlier scoring, tables, and plots lives in [`utils/`](utils/).

To regenerate a cleaned CSV for use outside the notebook, run `python -m utils.prepare`. It writes `data/processed/prospects.csv`.

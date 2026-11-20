# Career Value of NFL Combine Outliers Relative to Draft Position

Santiago Criado, Phu Duong, Sri Indla, and Steven Goetz

COS 424: Reasoning with Data · Princeton University

We study how often NFL Combine outliers are drafted, how early they are picked, and whether they produce more career value than expected for their pick, position, and draft class. [The notebook](nfl_draft_analysis.ipynb) shows how we build the sample and identify outliers within each position.

## Reproduce the analysis

Run the following commands from the repository root using Python 3.12.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter execute --inplace --timeout=120 nfl_draft_analysis.ipynb
```

The last command runs the notebook and saves the tables and figures in it. In a notebook editor, select the Python interpreter from `.venv` and run all cells.

The October 2026 [nflverse](https://github.com/nflverse/nflverse-data) snapshot is in [`data/raw/`](data/raw/). The notebook uses the Python code in [`utils/`](utils/) to prepare the data, identify outliers, and produce the tables and figures.

Run `python -m utils.prepare` to export the cleaned prospect sample to `data/processed/prospects.csv`.

# Neural-Network Classification of Breast Fine-Needle-Aspirate Features (working title)

MCS 306 Introduction to Artificial Intelligence, BSM CS 3A. From-scratch NumPy
MLP on the Breast Cancer Wisconsin (Diagnostic) dataset (WDBC). Educational
study, not a clinical tool. Repo:
https://github.com/arjeldru-dev/IntroductionToAI-FCO.git

## Layout

| Path                                                         | Content                                                                             |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `code/config.py`                                             | All parameters (single source of truth; no magic numbers elsewhere)                 |
| `code/data_prep.py`                                          | Load, encode (malignant = 1), stratified 70/15/15 split, train-only standardization |
| `code/mlp.py`, `code/gradient_check.py`                      | From-scratch MLP and gradient check (Martin)                                        |
| `code/run_experiments.py`, `code/make_figures.py`            | Experiments, baselines, figures (Cruz) [not yet written]                            |
| `data/wdbc.data`, `data/wdbc.names`, `data/wdbc.data.sha256` | Official UCI files and checksum                                                     |
| `tests/`                                                     | `python -m pytest tests -q`                                                         |
| `results/`, `paper/`                                         | Script outputs; LaTeX sections                                                      |
| `verification/verify_ch10_example.py`                        | Independent NumPy check of the Ch 10 Part VIII example (not part of the test suite) |

## Setup and run

```
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python code/data_prep.py          # prints checksum, class counts per split
python -m pytest tests -q
```

Python 3.12.3 was used for the delivered run. The split is deterministic (seed
42).

## Data

WDBC, 569 x 30, target M/B. Source:
https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic (DOI
10.24432/C5DW2B, CC BY 4.0 per UCI listing). `data/wdbc.data` and
`data/wdbc.names` are the files from the UCI zip, downloaded by a group member
from that page [date accessed: TO FILL]. The file has no header; columns are ID,
diagnosis, 30 features (mean, SE, worst of 10 measures). `data_prep.py` drops
the ID and checks the SHA-256 before use.

## Reproducibility

Seeds: split 42; initialization 0-9. Libraries pinned in `requirements.txt`. To
re-pin on your own machine: `pip freeze > requirements.txt` (then trim to
packages used).

## Contributing

Own GitHub account per member; commits `type(scope): summary`; changes by pull
request (U-15).

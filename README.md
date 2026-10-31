# Neural-Network Classification of Breast Fine-Needle-Aspirate Features: A From-Scratch Multilayer Perceptron on the Wisconsin Diagnostic Dataset

MCS 306 Introduction to Artificial Intelligence, BSM CS 3A, Bulacan State University, College of
Science. Educational study, not a clinical tool.
Repo: https://github.com/arjeldru-dev/IntroductionToAI-FCO.git

## Summary
We implement a multilayer perceptron from scratch in NumPy (one tanh hidden layer, sigmoid output,
binary cross-entropy, L2 penalty, full-batch gradient descent) and use it to classify breast masses
as malignant or benign from 30 features of digitized fine-needle-aspirate images (WDBC, 569
examples). The learning rate, hidden width and penalty are chosen on validation data only. On the
86 held-out test examples the selected network reaches 98.8% accuracy and misses one malignant
case. Logistic regression makes the same single error, so no advantage of the MLP is observed on
this split. Method, derivations and discussion are in the paper (`paper/main.tex`); the results
below are the headline numbers only.

## Quick start
```
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q                  # 22 tests
python code/run_experiments.py             # grid + baselines + ONE test evaluation (~3 min)
python code/make_figures.py                # figures from results/*.csv
python code/posthoc_test_spread.py         # optional post-hoc analysis (see below)
```
To build the paper (needs LaTeX), run `paper/build.ps1` (Windows) or `paper/build.sh`. They write
`paper/commit.tex` from `git rev-parse`, so the commit cited in section 3 is always the one built
from; a `-dirty` suffix means uncommitted changes.

Python 3.12.3 was used for the delivered run. Outputs land in `results/` and are committed, so
you can inspect them without re-running.

## Pipeline
Run in this order; each script reads the previous one's outputs, not live training.

| Step | Script | Reads | Writes (`results/`) |
|---|---|---|---|
| 1 | `code/data_prep.py` | `data/wdbc.data` | prints SHA-256 check and class counts per split |
| 2 | `code/run_experiments.py` | data | `grid_runs.csv`, `grid_summary.csv`, `loss_curves.csv`, `selection.json`, `val_metrics.csv`, `test_metrics.csv`, `predictions.csv`, `split_indices.csv`, `baseline_logreg_validation.csv`, `run_info.json` |
| 3 | `code/make_figures.py` | step 2 CSVs | `fig_*.png` (300 dpi), `roc_pr_summary.csv` |
| 4 | `code/posthoc_test_spread.py` | `selection.json` | `posthoc_test_spread.csv` |

`run_experiments.py` trains 270 runs (3 learning rates x 3 hidden widths x 3 lambdas x 10 init
seeds). The configuration is chosen on **validation** only (minimum mean validation BCE over
seeds 0-9, no regularizer term so lambdas are comparable). The test set is scored once, in
`final_test_evaluation()`, with init seed 0. `baselines.py` can also be run alone (validation only).

`posthoc_test_spread.py` is a second look at the test set across seeds 0-9. It is reported as
post-hoc and never re-selects the configuration; seed 0 must reproduce the MLP row of
`test_metrics.csv`.

## Layout
```
IntroductionToAI-FCO/
├── code/
│   ├── config.py                 # all parameters
│   ├── data_prep.py              # load, checksum, split, standardize
│   ├── mlp.py                    # from-scratch MLP
│   ├── gradient_check.py         # finite-difference check
│   ├── exp_common.py             # shared data access and metrics
│   ├── baselines.py              # majority class, logistic regression
│   ├── run_experiments.py        # grid, selection, test evaluation
│   ├── make_figures.py           # figures from saved results
│   └── posthoc_test_spread.py    # post-hoc seed spread
├── data/                         # wdbc.data, wdbc.names, wdbc.data.sha256
├── tests/                        # test_data_prep, test_mlp, test_lecture_example
├── verification/
│   └── verify_ch10_example.py    # independent check of the Ch 10 example
├── results/                      # CSV, JSON and PNG outputs
├── paper/
│   ├── main.tex                  # title page, abstract, references
│   ├── abstract_draft.tex
│   ├── sections/                 # 01_introduction ... 04_conclusion
│   └── assets/                   # logos
├── requirements.txt              # pinned libraries
├── .gitignore
├── .gitattributes                # stops line-ending conversion on data files
└── README.md
```

| Path | Content |
|---|---|
| `code/config.py` | All parameters (single source of truth; no magic numbers elsewhere) |
| `code/data_prep.py` | Load, checksum, encode (malignant = 1), stratified 70/15/15 split, train-only standardization |
| `code/mlp.py` | From-scratch MLP: forward, backprop, `fit`, `predict_proba` |
| `code/gradient_check.py` | Centered finite-difference check of backprop at all parameters (max gap < 1e-6) |
| `code/exp_common.py` | Shared data access and metrics for the experiment scripts |
| `code/baselines.py` | Majority-class rule and L2 logistic regression (lambda mapped to sklearn `C`) |
| `code/run_experiments.py` | Grid study, validation-only selection, single test evaluation |
| `code/make_figures.py` | Figures and ROC/PR summary from saved predictions |
| `code/posthoc_test_spread.py` | Post-hoc test-set spread over init seeds |
| `data/` | Official UCI `wdbc.data`, `wdbc.names`, `wdbc.data.sha256` |
| `tests/` | `test_data_prep.py`, `test_mlp.py`, `test_lecture_example.py` |
| `verification/verify_ch10_example.py` | Independent NumPy check of the Ch 10 Part VIII example (not part of the test suite) |
| `results/` | Script outputs (CSV, JSON, PNG) |
| `paper/` | LaTeX source (`main.tex`, `sections/01-04`); build artifacts and PDF are git-ignored |

## Configuration
Everything lives in [code/config.py](code/config.py); change values there, not in scripts.

| Setting | Value |
|---|---|
| Split | stratified 70/15/15, seed 42 |
| Learning rate (eta) | 0.01, 0.1, 0.5 |
| Hidden width (n1) | 4, 8, 16 |
| L2 strength (lambda) | 0, 1e-3, 1e-2 |
| Init | normal, scale 0.1, zero biases; seeds 0-9 |
| Stopping | 5000 full-batch epochs or gradient norm < 1e-6 |

## Results (delivered run)
Selected on validation: eta = 0.01, n1 = 16, lambda = 0.01. Test set (n = 86; 32 malignant):

| Model | Accuracy | Recall | F1 | Log loss |
|---|---|---|---|---|
| MLP (seed 0) | 0.988 | 0.969 | 0.984 | 0.050 |
| Logistic regression | 0.988 | 0.969 | 0.984 | 0.061 |
| Majority class | 0.628 | 0.000 | 0.000 | 0.660 |

Both models make the same single error (1 false negative, 0 false positives). Treat the MLP and
logistic regression as tied on accuracy; the test set is too small to separate them.

Caveats:
- **The MLP did not meet the gradient tolerance.** Every run hit the 5000-epoch cap
  (`mlp_converged: false`, final gradient norm about 0.016 vs the 1e-6 tolerance), so the reported
  numbers are for a fixed training budget, not a converged optimum. See `grid_summary.csv`
  (`n_converged`, `n_hit_max`).
- Post-hoc over seeds 0-9: accuracy and F1 do not vary; log loss is 0.0494 +/- 0.0005
  (`posthoc_test_spread.csv`).
- `exp_common.get_data()` falls back to scikit-learn's bundled WDBC copy if `data/wdbc.data` is
  missing, and records `data_source` in `run_info.json`. Check that field before trusting a run.

## Data
WDBC, 569 x 30, target M/B (212 malignant, 357 benign, counted from `data/wdbc.data`). Source: https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic
(DOI 10.24432/C5DW2B, CC BY 4.0 per UCI listing). `data/wdbc.data` and `data/wdbc.names` are the
files from the UCI zip, downloaded by a group member from that page [date accessed: October 26, 2026].
The dataset was introduced in Street, Wolberg & Mangasarian (1993), "Nuclear feature extraction
for breast tumor diagnosis" (SPIE 1905, https://doi.org/10.1117/12.148698).
The file has no header; columns are ID, diagnosis, 30 features (mean, SE, worst of 10 measures).
`data_prep.py` drops the ID and checks the SHA-256 before use.

## Reproducibility
Seeds: split 42; initialization 0-9. Libraries pinned in `requirements.txt`. Each run records
versions, platform, and data source in `results/run_info.json`. Results are deterministic for a
given platform and library set; runtimes in `run_info.json` are from the author's machine.
To re-pin on your own machine: `pip freeze > requirements.txt` (then trim to packages used).

## Contributing
Own GitHub account per member; commits `type(scope): summary`; changes by pull request.

| Member | GitHub | Contribution |
|---|---|---|
| Cristobal, Ardiel Drew L. | arjeldru-dev | Repository setup, data preparation and splits, pull-request merging, independent reproduction runs on Windows |
| Cruz, Eunice Angeline Y. | Meuni098 | Experiment grid, baselines, figures, post-hoc seed check |
| Denilla, Kerby C. | (no account) | LaTeX and video |
| Martin, Carl Louise P. | carssmm | From-scratch NumPy MLP, gradient check, unit tests |
| Nicolas, Yangel Mica P. | (no account) | LaTeX and video |

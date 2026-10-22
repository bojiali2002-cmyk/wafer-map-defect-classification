# Classical Dimensionality-Reduction Baseline for Wafer Maps

Compare flattened pixels, PCA and PCA+LDA using 3-NN and nearest-class-mean classifiers. This is a completed retrospective learning experiment, not a replacement CNN, production inspection system or hardware benchmark.

## Data and selection

The portable `data/study_data.npz` includes 3,757 training, 791 validation and 1,616 test maps, preprocessed to 64 × 64. Original split/lot boundaries are preserved. Samples are class-capped and exact preprocessed input duplicates are removed. CNN predictions are cached from the earlier study; it was trained with substantially more samples.

Only validation macro-F1 selects the configuration. PCA16 + 3NN obtains test macro-F1 0.4347 versus 0.3481 for the CNN on the identical subset. The paired lot-bootstrap difference interval is [0.0596, 0.1122]. Validation macro-F1 is 0.7990: the generalization gap remains unresolved. Normal-class recall is only 17.59%, so the baseline is not a drop-in improvement.

## Read or reproduce

Open `notebooks/01_complete_study.ipynb` to view saved outputs. No raw dataset or CNN weights are needed for the portable study.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_notebook.py
```

Run from this experiment directory. `requirements.txt` preserves the original recorded version pins; these have not been freshly installed during publication preparation. The parent repository requirements provide broader compatibility ranges. Restart kernels when switching between this experiment and the main project: both use a directory named `src`.

Execution overwrites results/figures and regenerates the excluded `selected_baseline.joblib` used by `predict.py`. Do not load untrusted pickle/joblib files. Test curves are retrospective diagnostics, not permission to reselect configurations using test results. No comparable CNN latency was measured.

See [original Chinese explanation](README.zh-CN.md), [engineering limitations](ENGINEERING_NOTES.md), [data provenance](DATA_SOURCES.md), and `data/audit.json`. The recorded raw-data regeneration command is in the Chinese explanation. No final English course report is claimed.

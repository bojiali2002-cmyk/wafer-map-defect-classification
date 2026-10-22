# Publication preparation checks

- wafer-map-defect-classification: Python syntax, Notebook JSON structure, code-cell compilation and saved-error checks passed
- wafer-map-defect-classification: common token/private-key/local-home-path patterns not found (heuristic scan, not a security guarantee)
- wafer-map-defect-classification: relative Markdown links resolve
- Wafer CNN: full saved test metrics recomputed; no row or lot overlap in saved manifests
- PCA baseline: portable subset loads without pickle; cached test indices/labels align and both macro-F1 results recompute

## Limits

- No full CNN retraining or full SECOM model refit was performed.
- No fresh pip installation or cross-platform environment validation was performed.
- No scientific claim of prospective validation or deployment is made.
- PyTorch forward execution was not run: PyTorch and scikit-learn are absent from this packaging runtime; metric recomputation uses NumPy definitions.

These checks validate packaging and saved artifacts, not new model training. Original desktop project files were not edited.

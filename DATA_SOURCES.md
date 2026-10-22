# Data sources and attribution

WM-811K: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map . The source data card lists CC0: Public Domain. Reference: M.-J. Wu, J.-S. R. Jang, and J.-L. Chen, “Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets,” IEEE Transactions on Semiconductor Manufacturing 28(1), 1–12 (2015).

The full raw archive is not redistributed. The dimensionality experiment includes a small derived 64x64 input subset, original row/lot metadata and cached CNN predictions for reproducibility; these are derived from WM-811K, not newly collected manufacturing data. See experiments/dimensionality_baseline/DATA_SOURCES.md and data/audit.json for provenance.

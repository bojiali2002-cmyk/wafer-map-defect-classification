# Data provenance

Source: [WM-811K wafer map on Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map).
Kaggle data card lists CC0 Public Domain. Credit the original research dataset even when using this derived subset.

Ming-Ju Wu, Jyh-Shing R. Jang, Jui-Long Chen, "Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets," IEEE Transactions on Semiconductor Manufacturing, vol. 28, no. 1, pp. 1–12, February 2015.

Method background: Xudong Jiang, "Linear Subspace Learning-Based Dimensionality Reduction," IEEE Signal Processing Magazine, vol. 28, no. 2, pp. 16–26, March 2011.

The subset was generated from the original LSWMD.pkl using the original project's train, validation and test manifests. Original CNN outputs were copied only for the selected test source indices; the original CNN was not retrained. Raw and subset SHA-256 values and exact counts are in data/audit.json. The portable data includes no pretrained CNN weights and no raw 2 GB pickle.

Data values are wafer test-map categories, not grayscale microscopy. The 64×64 arrays preserve numerical values 0/0.5/1 following the source project's fixed preprocessing.

# 晶圆缺陷分类的降维实验

独立课程实验项目，包含已执行 Notebook、中文讲解、数据、全部结果、图表与候选推理模型。**没有撰写英文报告。**

## 从这里开始

打开 `notebooks/01_complete_study.ipynb`。已保存表格、图像、输出和结论；无需先重跑即可检查。

建议 Python 3.12。在项目文件夹打开 Terminal：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install jupyterlab
python -m jupyterlab
```

在 JupyterLab 打开 Notebook，使用当前虚拟环境的 Python kernel，选择 **Restart Kernel and Run All Cells**。也可执行：

```bash
python run_notebook.py
```

此命令会用当前 Python 创建临时 kernel，重新执行全部 Notebook，覆盖生成的结果与图表。现有结果若需保留，可先复制项目。依赖版本记录于 `requirements.txt`；实测 Python 3.12.14。标准实验只需要随附数据，不需要网络、GPU、PyTorch、原始 2 GB 文件或原项目路径。

## 实验范围

- 原始数据：WM-811K，来源 https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
- 64×64 中心补零、floor-nearest resize、除以 2，与原 CNN 的预处理规则一致。
- 4096 维像素，比较不降维 / PCA / PCA128+LDA。
- 固定 3-NN，附加最近类均值 NCM 作为简单线性基线。
- PCA 维数 1,2,4,8,16,32,64,128；LDA 维数 1–8。
- 原项目训练/验证/测试边界保持不变；集合内按类限额抽样，然后去掉完全重复的 64×64 输入。
- 最终训练 3757、验证 791、测试 1616；批次不重叠。Near-full 验证只有 9 张，选择不确定性必须保留。
- 只用验证 Macro F1 选择模型和维数。测试曲线覆盖预先定义的网格，不能根据测试最高点重新选参数。
- CNN 比较来自原项目保存的预测，匹配同一测试行号；没有重新训练 CNN。

## 主要真实结果

验证选出 PCA 16 + 3-NN。共同测试子集：

| 模型 | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| 不降维 + 3NN | 27.41% | 31.09% | 0.2938 |
| PCA16 + 3NN（验证选出） | 39.91% | 41.82% | 0.4347 |
| 原 CNN（相同测试子集） | 35.95% | 38.57% | 0.3481 |

相对不降维，PCA16+3NN 准确率增加 12.50 个百分点，Macro F1 增加 0.1410。相对 CNN，Macro F1 增加 0.0866，lot bootstrap 95% 区间 [0.0596, 0.1122]。

**工程价值有限且具体**：能提供独立的传统基线，提示 Center/Loc/Scratch 等类别存在提升空间；但新基线对 Edge-Ring、none 等类别更差，none 召回率只有 17.59%，不能替换 CNN 或直接部署。验证 Macro F1 0.7990 与测试 0.4347 的差距提示跨批次泛化/验证代表性问题，尚未确定原因。

测试子集经过按类抽样，类别比例与原完整测试集不同，所以这里 CNN 的 35.95% 与原项目的 94.38% **不能直接比较**。这是回顾性基准；原项目已经观察过官方测试集。CNN 与传统模型训练数据量不同，bootstrap 不涵盖训练随机性或外部工厂变化。没有同条件 CNN 延迟测量，不宣称更快。

## 文件导航

- `notebooks/01_complete_study.ipynb`：完整已执行记录，按顺序阅读。
- `data/study_data.npz`：可移植的预处理数据及 CNN 对应预测，NumPy 加载无需 pickle。
- `data/manifest.csv`：原始索引、批次、类别及哈希。
- `data/audit.json`：抽样、去重、来源、校验和。
- `src/prepare_data.py`：从原始文件和原项目清单重新制作数据。
- `src/study.py`：拟合、验证选参、冻结测试评价、保存模型。
- `results/`：CSV、结论、模型和摘要。
- `figures/`：图表。
- `predict.py`：原始 NumPy wafer map 的候选模型推理工具。
- `run_notebook.py`：用当前 Python 完整执行 Notebook。

如果需要从原数据重新制作（正常阅读/重跑 Notebook 不需要）：

```bash
python src/prepare_data.py --raw /path/to/LSWMD.pkl --original /path/to/wafer-map-defect-classification --output data
```

原项目需包含三份 manifest、`models/test_predictions.csv`、`models/final_metrics.json`。仅加载可信来源的 pickle/joblib。原始数据不要加入版本控制。

## 提交前

本项目完成实验记录，不是最终英文 PDF 报告。未来报告需按作业要求整理为 4–5 页正文并附程序。不得将较早的合成数据预研结果混入本晶圆实验。

参考文献：M.-J. Wu, J.-S. R. Jang, J.-L. Chen, IEEE TSM, 28(1), 1–12, 2015；X. Jiang, IEEE SPM, 28(2), 16–26, 2011。

# GHGCM

Official PyTorch implementation of **GHGCM**, a component-wise gated recurrent
framework for estimating directed effective connectivity from multivariate time
series. The model assigns an independent GRU predictor to each target variable
and applies Group Lasso to the input weights, allowing target-specific directed
connections to be estimated from the learned sparse input groups.

This repository contains the core component-wise GRU implementation, training
utilities, a Lorenz-96 demonstration, and an example workflow for resting-state
fMRI data.

## Method overview

Given a multivariate time series with \(p\) variables, GHGCM trains \(p\)
separate GRU networks. The \(i\)-th network predicts the future value of target
variable \(i\) from the history of all variables. A Group Lasso penalty is
applied to the input-weight columns of each network. If the input-weight group
associated with variable \(j\) is zero for the \(i\)-th predictor, variable
\(j\) is treated as having no directed predictive effect on variable \(i\).

The resulting connectivity matrix follows the convention:

```text
connectivity[target, source]
```

## Repository structure

```text
GHGCM_code/
├── cgru_demo.ipynb       # Lorenz-96 simulation example
├── cgru_fmri.py          # Example fMRI workflow
├── models/
│   ├── cgru.py           # Component-wise GRU and training algorithms
│   ├── clstm.py          # Component-wise LSTM implementation
│   └── model_helper.py   # Model helper functions
└── utils/                # Regularization, normalization, and visualization
```

## Requirements

- Python 3.9 or later
- PyTorch 2.2 or later
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- Jupyter Notebook or JupyterLab for the demonstration notebook

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy scipy scikit-learn matplotlib jupyter
```

For GPU training, install the appropriate PyTorch build for your CUDA version
according to the official PyTorch installation instructions.

## Lorenz-96 demonstration

The notebook `cgru_demo.ipynb` generates a nonlinear Lorenz-96 time series with
a known directed interaction graph, trains GHGCM, and visualizes the estimated
connectivity matrix.

Start Jupyter from the repository root:

```bash
jupyter lab
```

Open `cgru_demo.ipynb` and select a device available on your system. If CUDA is
not available, change the notebook device setting to:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Basic usage

```python
import torch
from models.cgru import cGRU, train_model_gista

# X must have shape [n_sequences, n_time_points, n_variables].
X = torch.tensor(data, dtype=torch.float32)

model = cGRU(num_series=X.shape[-1], hidden=100)

train_loss, train_mse = train_model_gista(
    model,
    X,
    context=10,
    lam=0.01,
    lam_ridge=1e-3,
    lr=1e-3,
    max_iter=25000,
    check_every=100,
)

# Continuous directed-connectivity scores with shape [target, source].
connectivity = model.GC(threshold=False).detach().cpu().numpy()

# Binary graph obtained from exactly zero/nonzero Group Lasso groups.
binary_graph = model.GC(threshold=True).detach().cpu().numpy()
```

The Group Lasso coefficient `lam` controls graph sparsity. It should be selected
using validation data rather than the test set.

## fMRI data format

The example fMRI workflow expects MATLAB files containing the variable:

```text
voxel_time_series_all
```

The expected structure is a MATLAB cell array with one element per ROI. Each
element contains a matrix of shape:

```text
[n_voxels_in_roi, n_time_points]
```

The example preprocessing procedure averages the voxel signals within each ROI,
discards the first 10 time points, and standardizes every ROI time series.

The data used in the associated study are not included in this repository.
Users should update the input and output paths in `cgru_fmri.py` before running
the fMRI workflow.

## Outputs

`model.GC(threshold=False)` returns continuous Group Lasso input-weight norms.
Rows correspond to affected target variables and columns correspond to source
variables. The fMRI example saves each estimated connectivity matrix as a NumPy
`.npy` file.

## Reproducibility recommendations

- Set NumPy and PyTorch random seeds before training.
- Divide time series chronologically into training, validation, and test data.
- Estimate normalization parameters using training data only.
- Select Group Lasso strength and graph thresholds using validation data only.
- When multiple scans belong to one participant, keep all scans from that
  participant in the same classification fold.
- Report results across multiple random seeds.

## Citation

If you use this code, please cite the associated paper. The complete citation
and BibTeX entry will be added after publication.

```bibtex
@article{GHGCM,
  title   = {TO BE ADDED},
  author  = {TO BE ADDED},
  journal = {TO BE ADDED},
  year    = {TO BE ADDED}
}
```

## License

Add the repository license before public release. If no license is provided,
GitHub users do not automatically receive permission to reuse, modify, or
redistribute the code.

## Contact

For questions about the implementation or data format, please open a GitHub
issue or contact the corresponding author listed in the paper.

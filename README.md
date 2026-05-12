# ADASECANT Optimizer for PyTorch

PyTorch implementation of the ADASECANT optimizer.

This repository contains a custom implementation of the ADASECANT optimization algorithm compatible with the standard `torch.optim.Optimizer` interface. The implementation is based on the ADASECANT method described in:

- Caglar Gulcehre, Jose Sotelo, Marcin Moczulski, Yoshua Bengio, *A Robust Adaptive Stochastic Gradient Method for Deep Learning*, arXiv:1703.00788, 2017.
- Caglar Gulcehre, Marcin Moczulski, Yoshua Bengio, *ADASECANT: Robust Adaptive Secant Method for Stochastic Gradient*, arXiv:1412.7419, 2014.

The optimizer can be used as a drop-in replacement for common PyTorch optimizers such as SGD, Adam, or RMSprop in typical neural network training loops.

## Overview

ADASECANT is an adaptive stochastic optimization method designed to adjust update magnitudes using gradient statistics, curvature-related estimates, and variance reduction. The method uses stochastic curvature information to automatically tune learning rates during training.

This implementation keeps per-parameter running estimates of:

- gradient mean,
- squared gradient mean,
- update step statistics,
- curvature statistics,
- gamma variance-reduction terms,
- adaptive memory time scale `tau`.

The optimizer is implemented in `src/adasecant.py` and can be used directly with PyTorch models.

## Repository structure

```text
adasecant-pytorch/
├── README.md
├── testing.ipynb
└── src/
    ├── __init__.py
    └── adasecant.py

# ADASECANT Optimizer for PyTorch

PyTorch implementation of the ADASECANT optimizer, compatible with the standard `torch.optim.Optimizer` interface.

It can be used as a replacement for optimizers such as SGD, Adam, or RMSprop. ADASECANT is an adaptive method that tunes update magnitudes automatically using gradient statistics and stochastic curvature information, so it does not require a learning rate.

## Usage

See [demo.ipynb](/demo.ipynb).

## Installation

```bash
pip install -r requirements.txt
```

## Repository structure

```text
adasecant-pytorch/
├── README.md
├── requirements.txt
├── demo.ipynb              # quick usage example
└── src/
    ├── adasecant.py        # the optimizer
    ├── models.py           # models used in experiments
    ├── dataloaders.py      # data loading helpers
    └── experiments.ipynb   # benchmarks and analysis
```

## References

- Caglar Gulcehre, Jose Sotelo, Marcin Moczulski, Yoshua Bengio, *A Robust Adaptive Stochastic Gradient Method for Deep Learning*, arXiv:1703.00788, 2017.
- Caglar Gulcehre, Marcin Moczulski, Yoshua Bengio, *ADASECANT: Robust Adaptive Secant Method for Stochastic Gradient*, arXiv:1412.7419, 2014.

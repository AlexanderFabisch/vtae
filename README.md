# Variational Trajectory Autoencoder

Code corresponds to the paper

A. Fabisch, F. Kirchner: Sample-Efficient Policy Search with a Trajectory Autoencoder.
In: 4th Robot Learning Workshop at NeurIPS 2021.

The implementation is currently only available
[here](https://github.com/AlexanderFabisch/AlexanderFabisch.github.io/blob/source/content/downloads/vtae_code_example.zip?raw=true).

## Dependencies

We compiled a simple example that can be used to see that the
variational trajectory autoencoder (VTAE) actually works as described in
the paper. It comes in a jupyter notebook. The following libraries are
required to run it:

* Python (tested with version 3.6, should work with Python 2.7, 3.5+; we recommend to use anaconda: https://www.anaconda.com/distribution/)
* pytorch (tested with version 1.2; installation instructions: https://pytorch.org/get-started/locally/)
* matplotlib
* numpy
* seaborn
* tqdm
* jupyter lab

Most of the packages (except pytorch) should come with the anaconda
distribution. Otherwise typically `pip install <package>` or
`conda install <package>` should install them.

You can run the example on CPU but it should also work on a GPU that supports
CUDA. We can accelerate training by a factor of about 2 when running on a GPU.

## Quick Start

You should go to the directory in which the source code is located in a
terminal and run `jupyter lab`. Now you can open the notebook
`simple_training.ipynb`. It will contain precompiled figures but you
can also run all cells in sequential order to reproduce the figures
on your own. Some cells contain parameters that can be configured.

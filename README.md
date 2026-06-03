# Trident

Trident is the neural network design and training package in the Poseigen family.
It provides super-module generators, pre-assembled architectures, and a full training toolkit for biological sequences and beyond.

## Features

Trident is organized into five modules:

- prongs: super-module generators (Prongs) for composable neural network construction.
- preass: pre-assembled neural networks ranging from simple dense networks to convolutional autoencoders.
- deepstarr: DeepSTARR-based model implementations.
- othermodels: additional model architectures.
- utils: shared training and evaluation utilities including:
  - A Trainer with batch flipping and Epoch Sampling ([Binning Methods paper](https://doi.org/10.1101/2025.06.26.661884)).
  - A Predictor for producing predictions.
  - Candidate scorer and repeater for hyperparameter optimization with [poseigen_compass](https://github.com/Husam94/poseigen_compass).
  - Binned loss using bin metrics as a loss function.
  - Synthetic data generation.

## Installation

Install from PyPI:

```bash
pip install poseigen_trident
```

For local development, install from source using your preferred editable-install workflow.

## Usage

Import modules directly:

```python
import poseigen_trident.utils as tu
import poseigen_trident.prongs as prongs
import poseigen_trident.preass as preass
```

## Project Status

poseigen_trident is in active development and is intended to support neural network workflows across the Poseigen ecosystem.

## Related Projects

- poseigen_seaside: shared utilities and metrics foundation.
- poseigen_binmeths: binning and split-generation utilities.
- poseigen_compass: hyperparameter optimization and model evaluation.

## License

This project is released under the MIT License.
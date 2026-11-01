# Trident

[![PyPI version](https://img.shields.io/pypi/v/poseigen_trident)](https://pypi.org/project/poseigen_trident/)
[![Python versions](https://img.shields.io/pypi/pyversions/poseigen_trident)](https://pypi.org/project/poseigen_trident/)
[![License](https://img.shields.io/pypi/l/poseigen_trident)](https://pypi.org/project/poseigen_trident/)

Trident is the neural-network design and training package in the [Poseigen ecosystem](https://github.com/Husam94/poseigen). It provides composable architecture generators, ready-to-use models, and tools for training and evaluating neural networks on sequence, signal, and other data.

## Installation

Requires Python 3.12 or newer. Install from PyPI:

```bash
python -m pip install poseigen_trident
```

## Package

Trident is organized into five modules:

- **`prongs`** provides the `Prong_X`, `Prong_Y`, `Prong_Z`, and `Prong_W` generators for building configurable neural-network modules.
- **`preass`** provides pre-assembled architectures, including `SimpNet`, `StandardNN`, `RegularNN`, `ChampionNN`, and `CycleNN`, as well as truncated and dual-input variants.
- **`utils`** provides training, prediction, feature extraction, evaluation, and data utilities, including:
  - `TridentTrainer` for training with validation, optional batch flipping, weighting, and epoch sampling.
  - `TridentTrainer_BW` for bin-aware training with weighting and epoch sampling.
  - `TridentPredictor`, attention-pooling layers, and binned loss functions.
  - `TridentCanScorer` and `TridentCanRepeater` for evaluation workflows with [poseigen_compass](https://github.com/Husam94/poseigen_compass).
  - Synthetic data generation and other model utilities.
  - **`deepstarr`** provides a PyTorch implementation of DeepSTARR.
- **`othermodels`** provides models such as `BiChrom` and `iSEGnet` for combining DNA sequence and epigenomic signal inputs.

For background on binning and epoch sampling, see the [Binning Methods preprint](https://www.biorxiv.org/content/10.1101/2025.06.26.66188).

## Integrations

- [poseigen_seaside](https://github.com/Husam94/poseigen_seaside) provides shared utilities and metrics.
- [poseigen_binmeths](https://github.com/Husam94/poseigen_binmeths) provides binning and sampling utilities used by Trident's training workflows.
- [poseigen_compass](https://github.com/Husam94/poseigen_compass) supports candidate search and evaluation.
- [poseigen_mimic](https://github.com/Husam94/poseigen_mimic) and [poseigen_oyster](https://github.com/Husam94/poseigen_oyster) build sequence models using Trident.

## Usage

Import the architecture and training utilities you need:

```python
import poseigen_trident.preass as preass
import poseigen_trident.prongs as prongs
import poseigen_trident.utils as tu
```

After preparing a model, inputs, targets, and train/validation indices, train with `TridentTrainer`:

```python
metrics = tu.TridentTrainer(
    model,
    inps,
    out,
    Split=[train_indices, validation_indices],
    maxepochs=100,
)
```
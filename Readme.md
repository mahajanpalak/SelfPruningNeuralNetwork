# Self-Pruning Neural Network

This repository contains a PyTorch implementation of a self-pruning neural network. Instead of using traditional post-training pruning techniques, this network dynamically learns which connections are important and prunes the unimportant ones during the standard training loop.

## Overview

The core of this project is the `PrunableLinear` layer, which serves as a drop-in replacement for `nn.Linear`. 

Each weight is paired with a learnable gate. During the forward pass, these gates are passed through a sigmoid function to bound them between 0 and 1, and then multiplied element-wise with the weights. By adding an L1 regularization penalty on the gate values to our loss function, the network is pressured to drive gates to zero, effectively removing the corresponding weights.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

You can run the training script directly to replicate the experiments:

```bash
python self_pruning_network.py
```

This will automatically download the CIFAR-10 dataset (if not already present), train the model across different sparsity penalties ($\lambda$), and generate comparative plots in the `results/` directory.

### Custom Runs

You can easily adjust training parameters via the command line:

```bash
python self_pruning_network.py --epochs 30 --lambdas 0 1.0 5.0 20.0 --device mps
```

## Results Summary

Applying higher L1 penalties ($\lambda$) yields significantly sparser networks with only minimal drops in classification accuracy.

| Lambda ($\lambda$) | Accuracy | Sparsity |
| :--- | :--- | :--- |
| **0 (Baseline)** | 60.8% | 11.4% |
| **1.0** | 60.4% | 90.3% |
| **5.0** | 59.7% | 97.0% |
| **20.0** | 57.5% | 98.7% |

*Detailed analysis and training charts can be found in `REPORT.md`.*

## Structure

* `self_pruning_network.py`: The main script containing the custom layers, network architecture, and training pipeline.
* `REPORT.md`: A detailed write-up of the methodology, analysis of L1 regularization, and breakdown of experimental results.
* `results/`: Directory containing generated distribution plots and training curves.

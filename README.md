# Solving sudoku with reasoning, evolution and deep learning
Project for Knowledge Representation MSc. AI @VU, work by Matus Halak (DPLL, Transformer) and Thibault Giesbertz (CDCL, Evolutionary algorithm)

## Top-level directories
- `SAT-EA` directory contains all custom-written code for testing the SAT solvers and EA's across multiple datasets.
- `Transformer` directory contains all custom-written code for loading the [sapientinc/sudokuextreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme) dataset, defining and training the Transformer neural network.

## Reasoning - SAT

## Evolution - EA

## Deep Learning - Transformer
- `Transformer/transformer_env.yml` contains the conda environment that was used to develop the Transformer and run the experiments
- `Transformer/sudoku_extreme.py` handles (down)loading and preprocessing the dataset. Once the dataset is downloaded, it is stored in `Transformer/sudoku` folder. Furthermore, the script defines a PyTorch Dataset class, which handles train-validation-test splitting, as well as, random masking of specified number of empty cells during curriculum learning.
- `Transformer/SudokuTransformer.py` defines the transformer neural network.
- `Transformer/trainTransformer.py` is a script that can simply be run from the commandline via `python trainTransformer.py` and trains the transformer network. Unfortunately, hyperparameters are not yet integrated into command-line arguments, but are specified as arguments in the `curriculum()` function in the `__main__` block at the bottom of the script. This script automatically handles logging all relevant training, validation and testing data using Tensorboard. Logs and associated plots are stored in `Transformer/transformer_logs` and can be accessed via the commandline via `tensorboard --logdir transformer_logs`.
- `Transformer/transformer_logs` is provided for reproducibility and contains the logs for the experiments reported in the paper. Note that we performed many experiments in initial phases of developing the project which were not automatically logged and are not included here.

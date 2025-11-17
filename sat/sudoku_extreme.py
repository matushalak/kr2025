# Author: @matushalak
import os
from datasets import load_dataset, load_from_disk
from datasets import Dataset as hfDataset
from main import to_cnf, solve_cnf
from typing import Literal
from numpy import ndarray, array
from torch import tensor, long
from torch.utils.data import Dataset

def load_sudoku_extreme(DATA_DIR:str):
    'Loads sudoku problems into Hugging-face Dataset'
    if os.path.exists(DATA_DIR):
        print("Loading dataset from disk...")
        ds:hfDataset = load_from_disk(DATA_DIR)
    else:
        print("Downloading dataset...")
        ds:hfDataset = load_dataset("sapientinc/sudoku-extreme")
        ds.save_to_disk(DATA_DIR)
    return ds

def iter_sudoku_extreme(ds:hfDataset, split:Literal['train', 'test']):
    dsiterator = ds[split].iter(batch_size=1)
    return dsiterator, len(ds[split])

def preprocess_puzzle(puzzle:dict)->tuple[ndarray, ndarray]:
    problem:str = puzzle['question'][0] 
    solution:str = puzzle['answer'][0]
    problem = problem.replace('.', '0')
    return array(list(problem), dtype=int).reshape((9,9)), array(list(solution), dtype=int).reshape((9,9))

class SudokuDataset(Dataset):
    """
    Dataset taken from "https://huggingface.co/datasets/sapientinc/sudoku-extreme"
    Contains 3.8 Million training and 423 Thousand test 9x9 sudoku puzzles of varying difficulty. 
    Puzzles in the train set are mathematically inequivalent to those in the test set.
    Each puzzle has a guaranteed unique solution
    """
    def __init__(self, split:Literal['train', 'test', 'validate'], transform = None, root:str = "sudoku",
                 dataset_prop:float = 1.0):
        self.split = split
        proxy_split = 'train' if split != 'test' else 'test'
        self.dataset:hfDataset = load_sudoku_extreme(root)[proxy_split]
        if split != 'test':
            self.dataset = self.train_val_split(val_prop=0.1*dataset_prop, train_prop=0.9*dataset_prop)
        self.root_dir = root
        self.transform = transform
    
    def train_val_split(self, val_prop:float, train_prop:float):
        split_ds = self.dataset.train_test_split(test_size=val_prop, train_size=train_prop, 
                                                 seed=42)
        if self.split == 'train':
            return split_ds['train']
        if self.split == 'validate':
            return split_ds['test']

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        puzzle:dict = self.dataset[idx]
        problem:list[int] = [int(p) for p in puzzle['question'].replace('.', '0')]
        solution:list[int] = [int(a) for a in puzzle['answer']]
        return tensor(problem, dtype=long), tensor(solution, dtype=long)
    
if __name__ == '__main__':
    ds = load_sudoku_extreme('sudoku')
    dsi, niter = iter_sudoku_extreme(ds, 'test')
    for _ in range(niter):
        numpy_puzzle, numpy_solution = preprocess_puzzle(next(dsi))
        cnf, nvars = to_cnf(numpy_puzzle, nonconsecutive=False)
        sat, model = solve_cnf(cnf, nvars)
        print(sat)
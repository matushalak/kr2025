import os
from datasets import load_dataset, load_from_disk, Dataset
from main import to_cnf, solve_cnf
from typing import Literal
from numpy import ndarray, array

DATA_DIR = "sudoku"

def load_sudoku_extreme():
    if os.path.exists(DATA_DIR):
        print("Loading dataset from disk...")
        ds = load_from_disk(DATA_DIR)
    else:
        print("Downloading dataset...")
        ds = load_dataset("sapientinc/sudoku-extreme")
        ds.save_to_disk(DATA_DIR)
    return ds

def iter_sudoku_extreme(ds:Dataset, split:Literal['train', 'test']):
    dsiterator = ds[split].iter(batch_size=1)
    return dsiterator, len(ds[split])

def preprocess_puzzle(puzzle:dict)->tuple[ndarray, ndarray]:
    problem:str = puzzle['question'][0] 
    solution:str = puzzle['answer'][0]
    problem = problem.replace('.', '0')
    return array(list(problem), dtype=int).reshape((9,9)), array(list(solution), dtype=int).reshape((9,9))

if __name__ == '__main__':
    ds = load_sudoku_extreme()
    dsi, niter = iter_sudoku_extreme(ds, 'test')
    for _ in range(niter):
        numpy_puzzle, numpy_solution = preprocess_puzzle(next(dsi))
        cnf, nvars = to_cnf(numpy_puzzle, nonconsecutive=False)
        sat, model = solve_cnf(cnf, nvars)
        print(sat)
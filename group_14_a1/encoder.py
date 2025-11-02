"""
SAT Assignment Part 1 - Non-consecutive Sudoku Encoder (Puzzle -> CNF)

THIS is the file to edit.

Implement: to_cnf(input_path) -> (clauses, num_vars)

You're required to use a variable mapping as follows:
    var(r,c,v) = r*N*N + c*N + v
where r,c are in range (0...N-1) and v in (1...N).

You must encode:
  (1) Exactly one value per cell
  (2) For each value v and each row r: exactly one column c has v
  (3) For each value v and each column c: exactly one row r has v
  (4) For each value v and each sqrt(N)×sqrt(N) box: exactly one cell has v
  (5) Non-consecutive: orthogonal neighbors cannot differ by 1
  (6) Clues: unit clauses for the given puzzle
"""


from typing import Tuple, Iterable
import numpy as np

def to_cnf(input_path: str) -> Tuple[Iterable[Iterable[int]], int]:
    """
    Read puzzle from input_path and return (clauses, num_vars).

    - clauses: iterable of iterables of ints (each clause), no trailing 0s
    - num_vars: must be N^3 with N = grid size
    """
    # Load puzzle
    mat:np.ndarray = np.loadtxt(input_path, dtype=int)
    # Puzzle size
    N, _ = mat.shape
    nvars = int(N**3)
    # Unique values
    vals = np.arange(1, N+1)
    # Box size
    B = np.sqrt(N).astype(int)

    # Meshgrid - cols, rows for each entry
    c, r = np.meshgrid(np.arange(N), np.arange(N))

    # variable mapping
    rn2 = r * N**2
    cn = c * N
    # no values added
    no_val = rn2 + cn
    # encode all possible variables in 3d tensor (N, N, N)
    # third index shows the value of variable
    all_vars = rn2[..., None] + cn[..., None] + (np.ones((N, N))[..., None] * vals[None, :])
    print(len(min_one_per_cell(all_vars)))
    print(len(max_one_per_cell(all_vars)))
    print(len(one_per_cell(all_vars)))

    # CNF = (var1 OR ~var2) AND (~var3 OR var2) AND ...
    # each iterable has ORs inside and all the iterables are linked with ANDs
    # true variable = +varX, false variable = - varY
    # equivalent to above: (1, -2), (-3, 2), ... 
    breakpoint()

def clues(no_val:np.ndarray, mat:np.ndarray)->list[list[int]]:
    N, _ = mat.shape
    # encode observed values per cell -> variables
    true_var = no_val + mat
    # locations with clues
    nonzero = mat > 0
    # observed true variables (one clause per observed variable)
    constraint = np.array(true_var[nonzero])[:, None].tolist()
    return constraint

def one_per_cell(all_vars:np.ndarray)->list[list[int]]:
    '''
    More efficient to implement in one loop instead of 2
    '''
    N, _, _ = all_vars.shape
    min_one_constraint = []
    max_one_constraint = []
    for ri in range(N):
        for ci in range(N):
            # At least one constraint
            cell_options = all_vars[ri, ci, :]
            min_one_constraint.append(cell_options.tolist())
            
            # At most one constraint
            for val_index in range(N):
                val_options = cell_options.copy()
                negative = cell_options != cell_options[val_index]
                val_options[negative] *= -1
                max_one_constraint.append(val_options.tolist())
    
    constraint = min_one_constraint + max_one_constraint
    return constraint


def one_per_row(all_vars:np.ndarray)->list[list[int]]:
    pass

def one_per_col(all_vars:np.ndarray)->list[list[int]]:
    pass

def one_per_box(all_vars:np.ndarray, box_size:int)->list[list[int]]:
    pass

def non_consecutive(all_vars:np.ndarray)->list[list[int]]:
    pass

# Not used because unnecessarily loop over all cells twice
def min_one_per_cell(all_vars:np.ndarray)->list[list[int]]:
    N, _, _ = all_vars.shape
    constraint = []
    for ri in range(N):
        for ci in range(N):
            cell_options = all_vars[ri, ci, :].tolist()
            constraint.append(cell_options)
    return constraint

def max_one_per_cell(all_vars:np.ndarray)->list[list[int]]:
    N, _, _ = all_vars.shape
    constraint = []
    for val_index in range(N):
        for ri in range(N):
            for ci in range(N):
                cell_options = all_vars[ri, ci, :]
                negative = cell_options != cell_options[val_index]
                cell_options[negative] *= -1
                constraint.append(cell_options.tolist())
    return constraint

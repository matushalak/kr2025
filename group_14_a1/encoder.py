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
from itertools import combinations
import numpy as np

def to_cnf(input_path:str, 
           verbose:bool = False) -> Tuple[Iterable[Iterable[int]], int]:
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
    all_vars = (no_val[..., None]
                + (np.ones((N, N))[..., None] * vals[None, :])
                ).astype(int)

    # CNF = (var1 OR ~var2) AND (~var3 OR var2) AND ...
    # each iterable has ORs inside and all the iterables are linked with ANDs
    # true variable = +varX, false variable = - varY
    # equivalent to above: (1, -2), (-3, 2), ... 
    c1 = one_per_cell(all_vars)
    c2 = one_per_row(all_vars)
    c3 = one_per_col(all_vars)
    c4 = one_per_box(all_vars, B)
    c5 = non_consecutive(all_vars)
    c6 = clues(no_val, mat)

    constraints = c1 + c2 + c3 + c4 + c5 + c6
    if verbose:
        print(f'Total {len(constraints)} clauses (constraints) and {nvars} propositional variables')
    return constraints, nvars


def clues(no_val:np.ndarray, mat:np.ndarray,
          verbose:bool = False)->list[list[int]]:
    N, _ = mat.shape
    # encode observed values per cell -> variables
    true_var = no_val + mat
    # locations with clues
    nonzero = mat > 0
    # observed true variables (one clause per observed variable)
    constraint = np.array(true_var[nonzero])[:, None].tolist()
    if verbose:
        print(f'{len(constraint)} clue clauses')
    return constraint

def one_per_cell(all_vars:np.ndarray,
                 verbose:bool = False)->list[list[int]]:
    '''
    More efficient to implement in one loop instead of 2
    '''
    N, _, _ = all_vars.shape
    min_one_constraint = []
    max_one_constraint = []
    for ri in range(N):
        for ci in range(N):
            # At least one constraint (1 OR 2 OR 3)
            cell_options = all_vars[ri, ci, :]
            min_one_constraint.append(cell_options.tolist())
            # At most one constraint
            # DNF (1 AND ~2 AND ~3) OR (~1 AND 2 AND ~3) OR (~1 AND ~2 AND 3)
            # CNF (1 OR 2 OR 3) AND (~1 OR ~2) AND (~1 OR ~3) AND (~2 OR ~3)
            not_cell_options = -cell_options
            for comb in combinations(not_cell_options, r = 2): 
                max_one_constraint.append(list(comb))
    constraint = min_one_constraint + max_one_constraint
    if verbose:
        print(f'{len(constraint)} cell clauses')
    return constraint

def one_per_row(all_vars:np.ndarray,
                verbose:bool = False)->list[list[int]]:
    N, _, _ = all_vars.shape
    constraint = []
    for ri in range(N):
        for val_index in range(N):
            val_row = all_vars[ri, :, val_index]
            # At least one in row
            constraint.append(val_row.tolist())
            # At most one in row
            not_val_row = - val_row
            for comb in combinations(not_val_row, r = 2):
                constraint.append(list(comb))
    if verbose:
        print(f'{len(constraint)} row clauses')
    return constraint

def one_per_col(all_vars:np.ndarray,
                verbose:bool=False)->list[list[int]]:
    N, _, _ = all_vars.shape
    constraint = []
    for ci in range(N):
        for val_index in range(N):
            val_col = all_vars[:, ci, val_index]
            # At least one in col
            constraint.append(val_col.tolist())
            # At most one in col
            not_val_col = - val_col
            for comb in combinations(not_val_col, r = 2):
                constraint.append(list(comb))
    if verbose:
        print(f'{len(constraint)} column clauses')
    return constraint

def one_per_box(all_vars:np.ndarray, box_size:int,
                verbose:bool = False)->list[list[int]]:
    N, _, _ = all_vars.shape
    constraint = []
    boxes = [all_vars[bri:bri+box_size, bci:bci+box_size, :] 
             for bri in range(0, N, box_size) 
             for bci in range(0, N, box_size)]
    
    for box in boxes:
        for val_index in range(N):
            val_box = box[..., val_index]
            # At least one of these in box
            constraint.append(val_box.ravel().tolist())
            # At most one of these in box 
            not_val_box = -val_box.ravel()
            for comb in combinations(not_val_box, r=2):
                constraint.append(list(comb))
    if verbose:
        print(f'{len(constraint)} box clauses')
    return constraint

def non_consecutive(all_vars:np.ndarray, 
                    verbose:bool = False)->list[list[int]]:
    N, _, _ = all_vars.shape
    val_indices = np.arange(N)

    col_constraints = []
    for r in range(N):
        for ci in range(N-1):
            # Pairwise at most 1 arrays
            # First greater than second by 1
            fgs = [[-all_vars[r, ci, v1], -all_vars[r, ci+1, v2]] for v1, v2 in zip(val_indices[1:], val_indices)]
            # Second greater than first by 1
            sgf = [[-all_vars[r, ci, v1], -all_vars[r, ci+1, v2]] for v1, v2 in zip(val_indices, val_indices[1:])]

            col_constraints += fgs
            col_constraints += sgf

    row_constraints = []
    for ri in range(N-1):
        for c in range(N):
            # Pairwise at most 1 arrays
            # First greater than second by 1
            fgs = [[-all_vars[ri, c, v1], -all_vars[ri+1, c, v2]] for v1, v2 in zip(val_indices[1:], val_indices)]
            # Second greater than first by 1
            sgf = [[-all_vars[ri, c, v1], -all_vars[ri+1, c, v2]] for v1, v2 in zip(val_indices, val_indices[1:])]

            row_constraints += fgs
            row_constraints += sgf
    
    constraint = row_constraints + col_constraints
    if verbose:
        print(f'{len(constraint)} non-consecutive clauses')
    return constraint

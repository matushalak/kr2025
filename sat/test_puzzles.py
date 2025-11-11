import os
from main import to_cnf, solve_cnf, parse_dimacs
import re
from typing import Literal

def tester(folder:Literal['puzzles', 'test_bank', 'Jnh'], order:bool = False):
    n_puzzles = len(os.listdir(folder))-1
    
    if order:
        puzzle_names = ['' for _ in range(n_puzzles)] if order else []
        for puzzle in os.listdir(folder):
            if puzzle.endswith('csv'):
                continue
            puzzle_i = int(re.findall(r'\d+', puzzle)[0])
            puzzle_names[puzzle_i-1] = puzzle
    else:
        puzzle_names = [pn for pn in os.listdir(folder) if not pn.endswith('.csv')]

    for puzzle in puzzle_names:
        if folder == 'puzzles':
            clauses, num_vars = to_cnf(os.path.join(folder, puzzle))
        else:
            clauses, num_vars = parse_dimacs(os.path.join(folder, puzzle))
        status, _ = solve_cnf(clauses, num_vars)
        print(puzzle, status)

if __name__ == '__main__':
    tester(folder='Jnh', order=False)
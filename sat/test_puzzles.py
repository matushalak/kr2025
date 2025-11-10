import os
from main import to_cnf, solve_cnf

n_puzzles = len(os.listdir('puzzles'))-1
puzzle_names = ['' for _ in range(n_puzzles)]
for puzzle in os.listdir('puzzles'):
    if puzzle.endswith('csv'):
        continue
    puzzle_i = int(puzzle.split('.')[0].split('e')[1])
    puzzle_names[puzzle_i-1] = puzzle

for puzzle in puzzle_names:
    clauses, num_vars = to_cnf(os.path.join('puzzles', puzzle))
    status, _ = solve_cnf(clauses, num_vars)
    print(puzzle, status)
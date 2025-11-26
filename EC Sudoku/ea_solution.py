import math
import random
from typing import List, Sequence, Tuple


"""
EA improved to solve harder sudokus
"""

class Sudoku:
    def __init__(self,
                 values: Sequence[Sequence[int]] | Sequence[int],
                 use_non_consecutive: bool = False,
                 non_consecutive_weight: int = 1):
        """
        values: flat 81 or 9x9 grid (0 = empty, 1..9 digits)
        use_non_consecutive: whether to penalize consecutive neighbors
        non_consecutive_weight: how strongly to weight that penalty
        """
        self.grid = self._normalize(values)
        self.size, self.box_size = self._size_check()
        self.fixed = [
            [cell != 0 for cell in row]
            for row in self.grid
        ]

        self.use_non_consecutive = use_non_consecutive
        self.non_consecutive_weight = non_consecutive_weight

    # basic helpers
    def _normalize(self, values) -> List[List[int]]:
        """Convert input to a 9x9 list-of-lists with ints."""
        if not isinstance(values[0], (list, tuple)):
            vals = list(values)
            if len(vals) != 81:
                raise ValueError(f"Flat Sudoku requires 81 values, got {len(vals)}")
            return [vals[i * 9:(i + 1) * 9] for i in range(9)]

        grid = [list(row) for row in values]
        if len(grid) != 9 or any(len(row) != 9 for row in grid):
            raise ValueError("Grid must be 9x9")
        return grid

    def _size_check(self):
        """Generalized check (works for n^2 x n^2 Sudoku if we ever generalize)."""
        n_rows = len(self.grid)
        n_cols = len(self.grid[0])
        if n_rows != n_cols:
            raise ValueError("Sudoku must be square")

        size = n_rows
        box_size = int(math.isqrt(size))
        if box_size * box_size != size:
            raise ValueError("Side length must be a perfect square (for equal sub-boxes)")
        return size, box_size
    
    def set_non_consecutive(self, enabled: bool, weight: int | None = None):
        self.use_non_consecutive = enabled
        if weight is not None:
            self.non_consecutive_weight = weight

    # candidate initialization
    def random_candidate(self) -> List[List[int]]:
        """
        Create a random candidate solution:
        - Each row is a permutation of 1 size
        - All given clues are respected (fixed cells unchanged)
        - Empty cells (0) are filled with remaining digits in that row
        """
        size = self.size
        candidate = [row[:] for row in self.grid]  

        digits = list(range(1, size + 1))

        for r in range(size):
            present = {candidate[r][c] for c in range(size) if candidate[r][c] != 0}
            missing = [d for d in digits if d not in present]

            if len(missing) != sum(1 for c in range(size) if not self.fixed[r][c]):
                raise ValueError("Row has inconsistent fixed digits")

            random.shuffle(missing)
            it = iter(missing)

            for c in range(size):
                if not self.fixed[r][c]:
                    candidate[r][c] = next(it)

        return candidate

    # fitness / conflicts
    def _column_conflicts(self, candidate: List[List[int]]) -> int:
        """Sum of duplicate counts in each column."""
        size = self.size
        conflicts = 0
        for c in range(size):
            col = [candidate[r][c] for r in range(size)]
            conflicts += size - len(set(col))
        return conflicts

    def _box_conflicts(self, candidate: List[List[int]]) -> int:
        """Sum of duplicate counts in each box."""
        size = self.size
        b = self.box_size
        conflicts = 0
        for br in range(0, size, b):
            for bc in range(0, size, b):
                box_vals = []
                for r in range(br, br + b):
                    for c in range(bc, bc + b):
                        box_vals.append(candidate[r][c])
                conflicts += size - len(set(box_vals))
        return conflicts

    def fitness(self, candidate: List[List[int]]) -> int:
        """
        Lower is better.
        0 (with non-consecutive off) means: all columns and boxes have unique digits.
        With non-consecutive on, 0 means also no orthogonally adjacent consecutive numbers.
        """
        base_conflicts = self._column_conflicts(candidate) + self._box_conflicts(candidate)

        if not self.use_non_consecutive:
            return base_conflicts

        nc_conflicts = self._non_consecutive_conflicts(candidate)
        return base_conflicts + self.non_consecutive_weight * nc_conflicts
    
    def _non_consecutive_conflicts(self, candidate: List[List[int]]) -> int:
        """
        Count violations of the non-consecutive rule:
        For each cell, orthogonal neighbors (right, down) cannot differ by 1.
        Each violating pair is counted once.
        """
        size = self.size
        conflicts = 0

        for r in range(size):
            for c in range(size):
                v = candidate[r][c]

                # right neighbor
                if c + 1 < size:
                    v_r = candidate[r][c + 1]
                    if abs(v - v_r) == 1:
                        conflicts += 1

                # down neighbor
                if r + 1 < size:
                    v_d = candidate[r + 1][c]
                    if abs(v - v_d) == 1:
                        conflicts += 1

        return conflicts
    
    def column_conflict_indices(self, candidate: List[List[int]]) -> List[int]:
        """Return indices of columns that have duplicates."""
        size = self.size
        bad_cols = []
        for c in range(size):
            col = [candidate[r][c] for r in range(size)]
            if len(col) != len(set(col)):
                bad_cols.append(c)
        return bad_cols

    def box_conflict_indices(self, candidate: List[List[int]]) -> List[tuple[int, int]]:
        """
        Return list of (br, bc) box origins that have duplicates.
        br, bc are top left row, col of the box.
        """
        size = self.size
        b = self.box_size
        bad_boxes = []
        for br in range(0, size, b):
            for bc in range(0, size, b):
                vals = []
                for r in range(br, br + b):
                    for c in range(bc, bc + b):
                        vals.append(candidate[r][c])
                if len(vals) != len(set(vals)):
                    bad_boxes.append((br, bc))
        return bad_boxes

    # printing / debug
    def pretty_print(self, grid: List[List[int]] | None = None):
        """Nicely print a grid."""
        if grid is None:
            grid = self.grid

        for r in range(self.size):
            row_str = []
            for c in range(self.size):
                v = grid[r][c]
                ch = "." if v == 0 else str(v)
                row_str.append(ch)
                if (c + 1) % self.box_size == 0 and c + 1 < self.size:
                    row_str.append("|")
            print(" ".join(row_str))
            if (r + 1) % self.box_size == 0 and r + 1 < self.size:
                print("-" * (self.size * 2 + (self.box_size - 1)))


# GA helpers
def initialize_population(sudoku: Sudoku, pop_size: int) -> List[List[List[int]]]:
    return [sudoku.random_candidate() for _ in range(pop_size)]


def evaluate_population(sudoku: Sudoku, population: List[List[List[int]]]) -> List[int]:
    return [sudoku.fitness(ind) for ind in population]


def tournament_select(population: List[List[List[int]]],
                      fitnesses: List[int],
                      k: int = 3) -> List[List[int]]:
    """
    Select a single individual using k tournament, minimizing fitness.
    """
    size = len(population)
    best_idx = None
    for _ in range(k):
        i = random.randint(0, size - 1)
        if best_idx is None or fitnesses[i] < fitnesses[best_idx]:
            best_idx = i
    # return a deep copy so later modifications do not touch the parent
    return [row[:] for row in population[best_idx]]


def crossover_rows_uniform(parent1: List[List[int]],
                           parent2: List[List[int]]) -> List[List[int]]:
    """
    Row wise uniform crossover.
    For each row index, randomly choose row from parent1 or parent2.
    """
    size = len(parent1)
    child = []
    for r in range(size):
        if random.random() < 0.5:
            child.append(parent1[r][:])
        else:
            child.append(parent2[r][:])
    return child


def mutate_memetic(candidate: List[List[int]],
                   sudoku: Sudoku,
                   mutation_rate: float,
                   attempts: int = 40) -> None:
    """
    In place mutation with conflict guided local search.
    With probability mutation_rate, perform up to 'attempts' swaps.

    Move types:
    - row swap: swap two non fixed cells within a row
    - column swap (conflict guided): swap two non fixed cells within a conflicting column
    - box swap (conflict guided): swap two non fixed cells within a conflicting box

    Accept a swap if it does not worsen fitness (<=).
    """
    if random.random() >= mutation_rate:
        return

    size = sudoku.size
    current_fit = sudoku.fitness(candidate)

    for _ in range(attempts):
        move_type = random.choice(["row", "col", "box"])

        if move_type == "row":
            # plain row swap as before, but hill climbing
            r = random.randint(0, size - 1)
            mutable_indices = [c for c in range(size) if not sudoku.fixed[r][c]]
            if len(mutable_indices) < 2:
                continue
            c1, c2 = random.sample(mutable_indices, 2)

            candidate[r][c1], candidate[r][c2] = candidate[r][c2], candidate[r][c1]
            new_fit = sudoku.fitness(candidate)

            if new_fit <= current_fit:
                current_fit = new_fit
                if current_fit == 0:
                    break
            else:
                candidate[r][c1], candidate[r][c2] = candidate[r][c2], candidate[r][c1]

        elif move_type == "col":
            # focus on a column that actually has conflicts
            bad_cols = sudoku.column_conflict_indices(candidate)
            if not bad_cols:
                continue
            c = random.choice(bad_cols)
            # pick two non fixed rows in that column
            mutable_rows = [r for r in range(size) if not sudoku.fixed[r][c]]
            if len(mutable_rows) < 2:
                continue
            r1, r2 = random.sample(mutable_rows, 2)

            candidate[r1][c], candidate[r2][c] = candidate[r2][c], candidate[r1][c]
            new_fit = sudoku.fitness(candidate)

            if new_fit <= current_fit:
                current_fit = new_fit
                if current_fit == 0:
                    break
            else:
                candidate[r1][c], candidate[r2][c] = candidate[r2][c], candidate[r1][c]

        else:  # move_type == "box"
            bad_boxes = sudoku.box_conflict_indices(candidate)
            if not bad_boxes:
                continue
            br, bc = random.choice(bad_boxes)
            b = sudoku.box_size

            # all non fixed cells in this box
            cells = [
                (r, c)
                for r in range(br, br + b)
                for c in range(bc, bc + b)
                if not sudoku.fixed[r][c]
            ]
            if len(cells) < 2:
                continue

            (r1, c1), (r2, c2) = random.sample(cells, 2)

            candidate[r1][c1], candidate[r2][c2] = candidate[r2][c2], candidate[r1][c1]
            new_fit = sudoku.fitness(candidate)

            if new_fit <= current_fit:
                current_fit = new_fit
                if current_fit == 0:
                    break
            else:
                candidate[r1][c1], candidate[r2][c2] = candidate[r2][c2], candidate[r1][c1]
            
def evolve_ga(
    sudoku: Sudoku,
    pop_size: int = 500,
    max_generations: int = 10000,
    mutation_rate: float = 0.5,
    crossover_rate: float = 1.0,
    tournament_k: int = 3,
    elitism: int = 2,
    use_restarts: bool = False,
    stagnation_threshold: int = 1000,  # gens without improvement before hard restart
    max_restarts: int = 2,
    verbose: bool = True,
) -> Tuple[List[List[int]], int, int, bool]:
    """
    GA + memetic local search.
    Optional hard restarts:
    - If use_restarts is True and best fitness does not improve for
    'stagnation_threshold' generations, reinitialize the population.
    - Up to 'max_restarts' times.
    """
    # initial population
    population = initialize_population(sudoku, pop_size)
    fitnesses = evaluate_population(sudoku, population)

    # global best
    best_idx = min(range(pop_size), key=lambda i: fitnesses[i])
    best_fit = fitnesses[best_idx]
    best_ind = [row[:] for row in population[best_idx]]

    last_improvement_gen = 0
    restarts_done = 0

    if verbose:
        print(f"Initial best fitness: {best_fit}")

    if best_fit == 0:
        return best_ind, best_fit, 0, True

    for gen in range(1, max_generations + 1):
        # sort population by fitness ascending
        paired = list(zip(population, fitnesses))
        paired.sort(key=lambda p: p[1])
        population = [ind for ind, fit in paired]
        fitnesses = [fit for ind, fit in paired]

        # elitism
        new_population = []
        for i in range(elitism):
            new_population.append([row[:] for row in population[i]])

        # create rest of population
        while len(new_population) < pop_size:
            p1 = tournament_select(population, fitnesses, k=tournament_k)
            p2 = tournament_select(population, fitnesses, k=tournament_k)

            if random.random() < crossover_rate:
                child = crossover_rows_uniform(p1, p2)
            else:
                child = [row[:] for row in p1]

            mutate_memetic(child, sudoku, mutation_rate)
            new_population.append(child)

        population = new_population
        fitnesses = evaluate_population(sudoku, population)

        # generation best
        gen_best_idx = min(range(pop_size), key=lambda i: fitnesses[i])
        gen_best_fit = fitnesses[gen_best_idx]

        # update global best
        if gen_best_fit < best_fit:
            best_fit = gen_best_fit
            best_ind = [row[:] for row in population[gen_best_idx]]
            last_improvement_gen = gen

        # optional hard restart
        if (
            use_restarts
            and best_fit > 0
            and restarts_done < max_restarts
            and gen - last_improvement_gen >= stagnation_threshold
        ):
            if verbose:
                print(
                    f"Hard restart #{restarts_done + 1} at generation {gen} "
                    f"(best fitness so far: {best_fit})"
                )
            # fully reinitialize population
            population = initialize_population(sudoku, pop_size)
            # optionally inject best_ind as first elite
            population[0] = [row[:] for row in best_ind]
            fitnesses = evaluate_population(sudoku, population)
            restarts_done += 1
            last_improvement_gen = gen  # reset stagnation clock

        if verbose and gen % 50 == 0:
            print(f"Generation {gen}, best fitness: {best_fit}")

        if best_fit == 0:
            if verbose:
                print(f"Solved at generation {gen}")
            return best_ind, best_fit, gen, True

    if verbose:
        print(
            f"Finished {max_generations} generations, "
            f"best fitness: {best_fit}, restarts_done={restarts_done}"
        )
    return best_ind, best_fit, max_generations, (best_fit == 0)

# small runner
if __name__ == "__main__":
    random.seed(0)

    values_to_set = hard2

    sdk = Sudoku(values_to_set)
    print("Original puzzle:")
    sdk.pretty_print()

    best_ind, best_fit, gens, solved = evolve_ga(
        sdk,
        pop_size=500,
        max_generations=5000,
        mutation_rate=0.3,
        crossover_rate=1.0,
        tournament_k=3,
        elitism=2,
        use_restarts=False,
        verbose=True,
    )

    print("\nBest individual after evolution:")
    sdk.pretty_print(best_ind)
    print(f"\nBest fitness: {best_fit}, generations: {gens}, solved: {solved}")


# some puzzles to test on 
# !! These are just to make sure the program runs fine, these are NOT the puzzles the Paper is benchmarked on !!
example1 = [
    0, 8, 0, 0, 0, 0, 0, 9, 0,
    0, 0, 7, 5, 0, 2, 8, 0, 0,
    6, 0, 0, 8, 0, 7, 0, 0, 5,
    3, 7, 0, 0, 8, 0, 0, 5, 1,
    2, 0, 0, 0, 0, 0, 0, 0, 8,
    9, 5, 0, 0, 4, 0, 0, 3, 2,
    8, 0, 0, 1, 0, 4, 0, 0, 9,
    0, 0, 1, 9, 0, 3, 6, 0, 0,
    0, 4, 0, 0, 0, 0, 0, 2, 0,
]

easy1 = [
    0,8,0, 0,0,0, 0,9,0,
    0,0,7, 5,0,2, 8,0,0,
    6,0,0, 8,0,7, 0,0,5,

    3,7,0, 0,8,0, 0,5,1,
    2,0,0, 0,0,0, 0,0,8,
    9,5,0, 0,4,0, 0,3,2,

    8,0,0, 1,0,4, 0,0,9,
    0,0,1, 9,0,3, 6,0,0,
    0,4,0, 0,0,0, 0,2,0
]


easy2 = [
    8,0,2, 0,0,3, 5,1,0,
    0,6,0, 0,9,1, 0,0,3,
    7,0,1, 0,0,0, 8,9,4,

    6,0,8, 0,0,4, 0,2,1,
    0,0,0, 2,5,8, 0,6,0,
    9,2,0, 3,1,0, 4,0,0,

    0,0,0, 4,0,2, 7,8,0,
    0,0,5, 0,8,9, 0,0,0,
    2,0,0, 0,0,7, 1,0,0
]

hard1 = [
    0,0,6, 0,0,0, 0,0,0,
    0,8,0, 0,5,4, 2,0,0,
    0,4,0, 0,9,0, 0,7,0,

    0,0,7, 9,0,0, 3,0,0,
    0,0,0, 0,8,0, 4,0,0,
    6,0,0, 0,0,0, 1,0,0,

    2,0,3, 0,0,0, 0,0,1,
    0,0,0, 5,0,0, 0,4,0,
    0,0,8, 3,0,0, 5,0,2
]

hard2 = [
    0,0,2, 0,0,0, 0,0,0,
    0,0,3, 0,1,0, 0,0,6,
    0,4,0, 0,2,0, 0,3,0,

    1,0,0, 0,0,3, 0,0,9,
    0,0,5, 0,0,0, 4,0,0,
    2,0,0, 6,0,0, 0,0,8,

    0,9,0, 0,7,0, 0,4,0,
    7,0,0, 0,8,0, 5,0,0,
    0,0,0, 0,0,0, 3,0,0
]
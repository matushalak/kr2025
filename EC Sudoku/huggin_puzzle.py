import time
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from ea_solution import Sudoku, evolve_ga

"""
Code to run the EA on consecutive puzzles.
Loads the data from the hugging face dataset. 
"""

def convert_puzzle(puzzle: str) -> List[int]:
    """
    Convert string with digits and '.' to list of 81 ints (0 for empty).
    """
    puzzle = puzzle.strip()
    if len(puzzle) != 81:
        raise ValueError(f"Puzzle length != 81: {len(puzzle)}")
    return [0 if ch == "." else int(ch) for ch in puzzle]

def run_ea_sudoku(flat_values: List[int],
                  max_generations: int = 5000) -> Dict[str, Any]:
    """
    Run the EA on a single puzzle given as a flat list of 81 ints.
    Returns a dict with basic run info.
    """
    sdk = Sudoku(flat_values)
    best_ind, best_fit, gens, solved = evolve_ga(
        sdk,
        pop_size=500,
        max_generations=max_generations,
        mutation_rate=0.3,
        crossover_rate=1.0,
        tournament_k=3,
        elitism=2,
        use_restarts=False,
        verbose=False,
    )

    return {
        "solved": bool(solved),
        "generations": int(gens),
        "best_conflicts": float(best_fit),
        "history": None,           
        "solution_grid": best_ind, # 9x9 grid
    }


def main():
    # choose split and number of puzzles to test
    split = "test"          # or "train"
    max_puzzles = 100        
    max_generations = 5000  # per puzzle

    # HuggingFace dataset path
    splits = {"train": "train.csv", "test": "test.csv"}
    path = "hf://datasets/sapientinc/sudoku-extreme/" + splits[split]

    print(f"Loading dataset from {path}")
    df = pd.read_csv(path)

    # convert each puzzle to flat list with zeros
    df["puzzle_list"] = df["question"].apply(convert_puzzle)

    print("First converted puzzles:")
    print(df[["question", "puzzle_list"]].head())

    results = []

    print(f"\nRunning EA on first {max_puzzles} puzzles of split '{split}'\n")
    for idx, row in df.head(max_puzzles).iterrows():
        flat = row["puzzle_list"]

        t0 = time.perf_counter()
        res = run_ea_sudoku(flat, max_generations=max_generations)
        elapsed = time.perf_counter() - t0

        results.append(
            {
                "idx": idx,
                "puzzle": row["question"],
                "solved": res["solved"],
                "generations": res["generations"],
                "best_conflicts": res["best_conflicts"],
                "time_sec": elapsed,
            }
        )

        print(
            f"Puzzle {idx}: "
            f"solved={res['solved']}, "
            f"gens={res['generations']}, "
            f"best_conflicts={res['best_conflicts']}, "
            f"time={elapsed}s"
        )

    if results:
        out_df = pd.DataFrame(results)
        out_path = Path(f"ea_hf_results_{split}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"\nSaved results to {out_path.resolve()}")
    else:
        print("No puzzles processed, no results saved")


if __name__ == "__main__":
    main()
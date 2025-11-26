import os
import csv
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

from ea_solution import Sudoku, evolve_ga

"""
Code to run the EA on non-consecutive puzzles
"""

# Assuming you run this script from inside the NCSudoku directory
SUBDIRS = ["9_sat"]          
MANIFEST_NAME = "results.csv"
OUTPUT_CSV = "ea_nc_results.csv"   # EA benchmark results


def load_manifest(subdir: str) -> List[Dict[str, Any]]:
    """
    Load results.csv inside <subdir>/.

    CSV format:
        size,type,index,conflicts,decisions,time_sec,result,file
    Example row:
        9,sat,0,1538,5439,< 0.1,SAT,sat_000.txt
    """
    manifest_path = os.path.join(subdir, MANIFEST_NAME)
    rows: List[Dict[str, Any]] = []

    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["index"])
                filename = row["file"].strip()   # e.g. sat_000.txt
                rows.append(
                    {
                        "index": idx,
                        "file": filename,
                        "subdir": subdir,
                    }
                )
            except Exception as e:
                print(f"[{subdir}] Bad CSV row {row}: {e}")

    rows.sort(key=lambda r: r["index"])
    return rows


def resolve_path(subdir: str, filename: str) -> str:
    """
    Most CSVs use filenames like `sat_000.txt`.

    But if a row uses `9_sat/sat_000.txt`, detect that too.
    """

    # Case 1: filename already contains folder(s)
    if "/" in filename:
        p = os.path.join(filename)
        if os.path.exists(p):
            return p

    # Case 2: assume it's relative to subdir
    p = os.path.join(subdir, filename)
    return p


def load_puzzle_flat(path: str) -> List[int]:
    """
    Load a puzzle file as a 9x9 grid of ints via np.loadtxt,
    flatten row-major, and return as list of 81 ints.
    Zeros represent blanks.
    """
    mat = np.loadtxt(path, dtype=int)
    if mat.size != 81:
        raise ValueError(f"Expected 81 cells, got shape {mat.shape} from {path}")
    return mat.astype(int).ravel().tolist()


def run_ea_sudoku(flat_values: List[int],
                  max_generations: int = 5000) -> Dict[str, Any]:
    """
    Run the EA on a single puzzle given as a flat list of 81 ints.
    Returns a dict with basic run info.
    """
    sdk = Sudoku(flat_values,
                 use_non_consecutive=True,
                 non_consecutive_weight=2)
    best_ind, best_fit, gens, solved = evolve_ga(
        sdk,
        pop_size=500,
        max_generations=max_generations,
        mutation_rate=0.3,
        crossover_rate=1.0,
        tournament_k=3,
        elitism=2,
        use_restarts=True,
        verbose=True,
    )

    return {
        "solved": bool(solved),
        "generations": int(gens),
        "best_conflicts": float(best_fit),
        "solution_grid": best_ind,  # 9x9 grid (if you ever want to inspect)
    }


def run_suite(subdir: str,
              max_generations: int = 5000,
              max_puzzles: int | None = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run EA on all (or first max_puzzles) puzzles in one subdir.

    Returns:
      stats dict + list of per-puzzle result rows for writing to CSV.
    """
    rows = load_manifest(subdir)
    if not rows:
        print(f"[{subdir}] No rows found in CSV")
        return {}, []

    if max_puzzles is not None:
        rows = rows[:max_puzzles]

    total = 0
    solved_count = 0
    t0 = time.time()

    out_rows: List[Dict[str, Any]] = []

    for row in rows:
        total += 1
        idx = row["index"]
        file_field = row["file"]
        puzzle_path = resolve_path(subdir, file_field)

        try:
            flat = load_puzzle_flat(puzzle_path)

            start = time.perf_counter()
            res = run_ea_sudoku(flat, max_generations=max_generations)
            elapsed = time.perf_counter() - start

            solved = res["solved"]
            gens = res["generations"]
            best_conf = res["best_conflicts"]

            if solved:
                solved_count += 1

            print(
                f"[{subdir}] {file_field} (#{idx}): "
                f"solved={solved}, gens={gens}, "
                f"best_conflicts={best_conf}, time={elapsed:.2f}s"
            )

            out_rows.append(
                {
                    "idx": idx,
                    "subdir": subdir,
                    "puzzle": file_field,     # file name
                    "solved": solved,
                    "generations": gens,
                    "time_sec": elapsed,
                    "best_conflicts": best_conf,
                }
            )

        except Exception as e:
            print(f"[{subdir}] {file_field} (#{idx}): ERROR {e}")
            out_rows.append(
                {
                    "idx": idx,
                    "subdir": subdir,
                    "puzzle": file_field,
                    "solved": False,
                    "generations": 0,
                    "time_sec": 0.0,
                    "best_conflicts": float("inf"),
                }
            )

    dt = time.time() - t0
    print()
    print(f"[{subdir}] Total puzzles: {total}")
    print(f"[{subdir}] Solved: {solved_count}")
    print(f"[{subdir}] Time: {dt:.2f}s\n")

    stats = {
        "total": total,
        "solved": solved_count,
        "time": dt,
    }
    return stats, out_rows


def main():
    print("Starting")
    max_generations = 5000      # per puzzle
    max_puzzles: int | None = None  # set to e.g. 100 to cap

    grand_total = 0
    grand_solved = 0
    grand_time = 0.0
    all_rows: List[Dict[str, Any]] = []

    for subdir in SUBDIRS:
        stats, rows = run_suite(
            subdir,
            max_generations=max_generations,
            max_puzzles=max_puzzles,
        )
        if not stats:
            continue
        grand_total += stats["total"]
        grand_solved += stats["solved"]
        grand_time += stats["time"]
        all_rows.extend(rows)

    print("========== EA OVERALL ==========")
    print(f"Total puzzles: {grand_total}")
    print(f"Solved: {grand_solved}")
    print(f"Total time: {grand_time:.2f}s")

    # Save combined results
    fieldnames = ["idx", "subdir", "puzzle", "solved", "generations", "time_sec", "best_conflicts"]
    out_path = Path(OUTPUT_CSV)
    with out_path.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved EA results to {out_path.resolve()}")


if __name__ == "__main__":
    main()
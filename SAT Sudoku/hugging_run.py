# bench_hugging_sat_runner.py

import os
import csv
import time

from encoder import to_cnf      # encoder from encoder.py
from solver_dpll import solve_cnf   # Change to solver_cdcl to run with that solver

INPUT_CSV = "hugging_puzzles_prepared.csv"
OUTPUT_CSV = "results_sat_dpll.csv"
TMP_DIR = "hugging_tmp_puzzles"  # directory for temporary puzzle .txt files

def main():
    os.makedirs(TMP_DIR, exist_ok=True)

    results = []

    with open(INPUT_CSV, newline="") as f_in:
        reader = csv.DictReader(f_in)

        total = 0
        failed = 0
        t0 = time.time()

        for row in reader:
            total += 1
            idx = int(row["idx"])
            puzzle_text = row["puzzle"]

            tmp_path = os.path.join(TMP_DIR, f"hug_{idx}.txt")

            try:
                # puzzle_text is already 9 lines of "0 9 0 ..." etc.
                with open(tmp_path, "w") as f_tmp:
                    f_tmp.write(puzzle_text.strip() + "\n")

                clauses, num_vars = to_cnf(tmp_path)

                start = time.perf_counter()
                status, _ = solve_cnf(clauses, num_vars)
                elapsed = time.perf_counter() - start

                status_str = str(status).strip().upper()
                solved = status_str.startswith("SAT")

                results.append(
                    {
                        "idx": idx,
                        "solved": solved,
                        "status": status_str,
                        "time_sec": elapsed,
                    }
                )

                print(
                    f"Puzzle {idx}: status={status_str}, "
                    f"solved={solved}, time={elapsed}s"
                )

            except Exception as e:
                failed += 1
                print(f"Puzzle {idx}: ERROR {e}")

                results.append(
                    {
                        "idx": idx,
                        "solved": False,
                        "status": "ERROR",
                        "time_sec": 0.0,
                    }
                )

        dt = time.time() - t0
        print()
        print(f"Processed {total} puzzles in {dt}s")
        print(f"ERRORS: {failed}")

    fieldnames = ["idx", "solved", "status", "time_sec"]
    with open(OUTPUT_CSV, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote results for {len(results)} puzzles to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
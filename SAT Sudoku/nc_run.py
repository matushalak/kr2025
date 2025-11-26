import os
import csv
import time

from encoder import to_cnf      # encoder from encoder.py
from solver_dpll import solve_cnf   # SAT solver from solverM.py

ROOT_DIR = "NCSudoku"
SUBDIRS = ["9_sat"]          
MANIFEST_NAME = "results.csv"

# output CSV for all processed puzzles
OUTPUT_CSV = "nc_results_sat_dpll.csv"

def load_manifest(subdir: str):
    """
    Load the results.csv.
    CSV format:
        size,type,index,conflicts,decisions,time_sec,result,file
    """
    manifest_path = os.path.join(subdir, MANIFEST_NAME)
    rows = []

    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["index"])
                status = row["result"].strip().upper()
                filename = row["file"].strip()  
                rows.append({
                    "index": idx,
                    "status": status,      # expected result from manifest
                    "file": filename,
                    "subdir": subdir,
                })
            except Exception as e:
                print(f"[{subdir}] Bad CSV row {row}: {e}")

    rows.sort(key=lambda r: r["index"])
    return rows


def resolve_path(subdir: str, filename: str):
    # If filename already contains a folder:
    if "/" in filename:
        p = os.path.join(filename)
        if os.path.exists(p):
            return p
    p = os.path.join(subdir, filename)
    return p


def run_suite(subdir: str):
    """
    Run solver on all puzzles in one subdir using its results.csv manifest.
    Returns:
      stats dict + list of per-puzzle result rows for writing to CSV.
    """
    rows = load_manifest(subdir)
    if not rows:
        print(f"[{subdir}] No rows found in CSV")
        return {}, []

    total = matched = mismatched = failed = 0
    t0 = time.time()

    out_rows = []  # rows to write to final CSV

    for row in rows:
        total += 1
        idx = row["index"]
        expected = row["status"]      # "SAT" or "UNSAT" from manifest
        file_field = row["file"]

        puzzle_path = resolve_path(subdir, file_field)

        try:
            clauses, num_vars = to_cnf(puzzle_path)

            start = time.perf_counter()
            status, _ = solve_cnf(clauses, num_vars)
            elapsed = time.perf_counter() - start

            got = str(status).strip().upper()
        except Exception as e:
            failed += 1
            print(f"[{subdir}] {file_field} (#{idx}): ERROR {e}")

            out_rows.append({
                "idx": idx,
                "subdir": subdir,
                "puzzle": file_field,
                "expected": expected,
                "status": "ERROR",
                "solved": False,
                "time_sec": 0.0,
            })
            continue

        solved = (got == expected)

        if solved:
            matched += 1
            print(f"[{subdir}] {file_field} (#{idx}): got={got}, expected={expected} OK")
        else:
            mismatched += 1
            print(f"[{subdir}] {file_field} (#{idx}): got={got}, expected={expected} MISMATCH")

        # Collect row for final CSV
        out_rows.append({
            "idx": idx,
            "subdir": subdir,
            "puzzle": file_field,   # filename
            "expected": expected,   # ground truth SAT/UNSAT from manifest
            "status": got,          # solver status
            "solved": solved,       # True if status matches expected
            "time_sec": elapsed,
        })

    dt = time.time() - t0
    print()
    print(f"[{subdir}] Total: {total}")
    print(f"[{subdir}] OK: {matched}")
    print(f"[{subdir}] MISMATCH: {mismatched}")
    print(f"[{subdir}] ERRORS: {failed}")
    print(f"[{subdir}] Time: {dt}s\n")

    stats = {
        "total": total,
        "matched": matched,
        "mismatched": mismatched,
        "failed": failed,
        "time": dt,
    }
    return stats, out_rows


def main():
    grand_total = grand_matched = grand_mismatched = grand_failed = 0
    grand_time = 0.0
    all_rows = []

    for subdir in SUBDIRS:
        stats, rows = run_suite(subdir)
        if not stats:
            continue
        grand_total += stats["total"]
        grand_matched += stats["matched"]
        grand_mismatched += stats["mismatched"]
        grand_failed += stats["failed"]
        grand_time += stats["time"]
        all_rows.extend(rows)

    print("========== OVERALL ==========")
    print(f"Total puzzles: {grand_total}")
    print(f"Matched: {grand_matched}")
    print(f"Mismatched: {grand_mismatched}")
    print(f"Errors: {grand_failed}")
    print(f"Total time: {grand_time}s")

    # Write combined CSV
    fieldnames = ["idx", "subdir", "puzzle", "expected", "status", "solved", "time_sec"]
    with open(OUTPUT_CSV, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote benchmark results for {len(all_rows)} puzzles to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
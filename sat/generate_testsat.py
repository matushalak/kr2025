# Re-run the test bank builder. If an environment reset occurred, this will recreate everything.
from itertools import product
from pathlib import Path

def has_unit_clause(cnf):
    return any(len(cl) == 1 for cl in cnf)

def brute_solve(nvars, cnf):
    for bits in product([False, True], repeat=nvars):
        ok = True
        for cl in cnf:
            if not any((bits[abs(l)-1] if l>0 else (not bits[abs(l)-1])) for l in cl):
                ok = False
                break
        if ok:
            return "SAT", {i+1: bits[i] for i in range(nvars)}
    return "UNSAT", None

def to_dimacs(nvars, cnf):
    lines = [f"p cnf {nvars} {len(cnf)}"]
    for cl in cnf:
        lines.append(" ".join(map(str, cl)) + " 0")
    return "\n".join(lines) + "\n"

tests = []

# 1) SAT: XOR over 2 variables
cnf1 = [(1, 2), (-1, -2)]

# 2) UNSAT: full 2-CNF cube
cnf2 = [(1, 2), (-1, 2), (1, -2), (-1, -2)]

# 3) SAT: exactly one of three
cnf3 = [(1, 2, 3), (-1, -2), (-1, -3), (-2, -3)]

# 4) UNSAT: PHP(3,2)
def v(i,j): return (i-1)*2 + j
cnf4 = []
for i in range(1,4):
    cnf4.append((v(i,1), v(i,2)))
for j in range(1,3):
    for a in range(1,4):
        for b in range(a+1,4):
            cnf4.append((-v(a,j), -v(b,j)))

# 5) SAT backtrack: x1 must be False
cnf5 = [(-1,2), (-1,-2), (1,3), (-3,4), (-4,3)]

# 6) UNSAT backtrack both sides
cnf6 = [(1,2), (1,-2), (-1,3), (-1,-3)]

# 7) SAT deeper chain (no initial units)
cnf7 = [(1,2,3), (-1,4), (-4,5), (-5,2), (-2,3), (-3,6), (6,1)]

# 8) UNSAT 3-SAT mix
cnf8 = [(1,2,3), (1,2,-3), (1,-2,3), (-1,2,3), (-1,-2,-3)]

# 9) SAT two XORs linked
cnf9 = [(1,2), (-1,-2), (3,4), (-3,-4), (-1,3)]

# 10) UNSAT variant over-constrained
cnf10 = [(1,2,3), (-1,-2), (-1,-3), (-2,-3), (-1, -2), (1, -2), (-1, 2)]

tests_data = [
    ("01_sat_xor2", 2, cnf1, "SAT; exactly-one of {1,2}."),
    ("02_unsat_full2", 2, cnf2, "UNSAT; four 2-clauses contradict."),
    ("03_sat_exactly_one_3", 3, cnf3, "SAT; exactly one of {1,2,3}."),
    ("04_unsat_php_3_2", 6, cnf4, "UNSAT; Pigeonhole 3→2."),
    ("05_sat_backtrack_x1_false", 4, cnf5, "SAT; x1=True leads to conflict → must backtrack to x1=False."),
    ("06_unsat_branch_both_sides", 3, cnf6, "UNSAT; whichever way you branch on x1, unit conflict arises later."),
    ("07_sat_deeper_chain", 6, cnf7, "SAT; needs multiple decisions before units appear."),
    ("08_unsat_3sat_mix", 3, cnf8, "UNSAT; no initial units."),
    ("09_sat_two_xors_link", 4, cnf9, "SAT; coupled XORs."),
    ("10_unsat_exactly_one_break", 3, cnf10, "UNSAT; over-constrained variant."),
]

outdir = Path("test_bank")
outdir.mkdir(exist_ok=True, parents=True)

summary_lines = ["id,name,nvars,nclauses,status,example_model_DIMACS,notes"]
for i,(name, nvars, cnf, notes) in enumerate(tests_data, start=1):
    # ensure no initial units
    assert not has_unit_clause(cnf), f"{name} unexpectedly has unit clauses"
    status, model = brute_solve(nvars, cnf)
    # write DIMACS
    (outdir / f"{name}.cnf").write_text(to_dimacs(nvars, cnf), encoding="utf-8")
    model_str = ""
    if status == "SAT":
        model_str = " ".join(str(v if model[v] else -v) for v in range(1, nvars+1))
    summary_lines.append(f"{i},{name},{nvars},{len(cnf)},{status},{model_str},{notes}")

summary_path = outdir / "SUMMARY.csv"
summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

print("Wrote files to:", outdir.as_posix())
print("Summary at:", summary_path.as_posix())
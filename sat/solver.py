"""
SAT Assignment Part 2 - Non-consecutive Sudoku Solver (Puzzle -> SAT/UNSAT)

THIS is the file to edit.

Implement: solve_cnf(clauses) -> (status, model_or_None)"""


from typing import Iterable, List, Tuple, Literal
import numpy as np
from collections import defaultdict

def choose_literal(clauses: Iterable[Iterable[int]], 
                   assignment:dict[int, bool],
                   method:Literal['random', 'MOM'],
                   nvars:int)->int:
    nclauses = len(clauses)
    match method:
        case 'random':
            chosen = False
            while not chosen:
                icl = np.random.choice(nclauses)
                cl = clauses[icl]
                ivar = np.random.choice(len(cl))
                lit = cl[ivar]
                if abs(lit) not in assignment:
                    return lit
        case 'MOM':
            k = 5
            clause_lengths = np.array([len(c) for c in clauses])
            smallest = np.min(clause_lengths)
            small_clauses = np.where(clause_lengths == smallest)[0]
            
            unassigned_lits = []
            vars = np.arange(1, nvars+1)
            for var in vars:
                if var not in assignment:
                    unassigned_lits += [var, -var]
            unassigned_lits_arr = np.array(unassigned_lits).reshape(-1, 2)
            counters = np.zeros_like(unassigned_lits_arr)
            
            for ic in small_clauses:
                cl = clauses[ic]
                index_lits = np.array([unassigned_lits.index(l) for l in cl])
                index_col = index_lits % 2
                index_row = index_lits // 2
                counters[index_row, index_col] += 1
            
            MOM = (counters.sum(axis = 1) * 2**k) + counters.prod(axis = 1)
            var_idx = np.argmax(MOM)
            lit_idx = np.argmax(counters[var_idx,:])
            lit = unassigned_lits_arr[var_idx, lit_idx]
            return lit



def simplify(l:int, clauses: Iterable[Iterable[int]]
             )-> Tuple[bool, Iterable[Iterable[int]]]:
    new_clauses = []
    for c in clauses:
        # drop true clause
        if l in c:
            continue
        # drop negative from the clause
        if -l in c:
            assert len(c) != 0
            shorter = [p for p in c if p != -l]
            # if empty clause = conflict and stop!
            # Contains empty clause, unsat for this partial assignment
            if len(shorter) == 0:
                return False, []
            else:
                new_clauses.append(shorter)
        # unaffected clauses
        else:
            new_clauses.append(c)
    return True, new_clauses

def unit_propagation(clauses: Iterable[Iterable[int]],
                     assignment: dict[int, bool]
                     )->Tuple[bool, Iterable[Iterable[int]], dict[int, bool]]:
    new_clauses, new_assignment = clauses, assignment
    propagating = True
    while propagating:
        propagating = False
        unit = None
        for c in new_clauses:
            if len(c) == 1:
                unit = c[0]
                break
        if unit is None:
            break
        # print('UNIT:', unit)
        # Keep unit-propagating in the simplified cnf
        uvar = abs(unit)
        uval = unit > 0
        if uvar in new_assignment:
            if new_assignment[uvar] != uval:
                # contradiction
                return (False, new_clauses, new_assignment)
        else:
            new_assignment[uvar] = uval
        reducible, new_clauses = simplify(unit, new_clauses)
        # Reveals a problem with our partial assignment!
        # (attempting to set unit to True led to contradiction)
        if not reducible:
            return (False, new_clauses, new_assignment)
        else:
            propagating = True
    return True, new_clauses, new_assignment


def dpll(clauses: Iterable[Iterable[int]], 
         assignment:dict[int, bool]|None,
         nvars:int
         )->Tuple[bool, dict[int, bool]|None]:
    if assignment is None:
        assignment = dict()
    # Unit propagation - early feasibility check for partial assignment
    feasible, cl1, ass1 = unit_propagation(clauses, assignment)
    if not feasible:
        return False, None
    # All clauses have been successfully removed
    if len(cl1) == 0:
        return True, ass1
    # Choose branching literal to set to true
    l = choose_literal(cl1, ass1, method='MOM', nvars=nvars)
    # print('LIT:', l, flush=True)
    # First branch - literal is True
    model1 = dict(ass1)
    reducible, new_clauses = simplify(l, cl1)
    if reducible:
        model1[abs(l)] = l > 0
        sat, model = dpll(new_clauses, model1, nvars)
        if sat: 
            return True, model
    # Second branch - flip literal
    model2 = dict(ass1)
    reducible2, new_clauses2 = simplify(-l, cl1)
    if reducible2:
        model2[abs(l)] = l < 0
        sat, model = dpll(new_clauses2, model2, nvars)
        if sat: 
            return True, model
    # If all branches tried and still not converged
    return False, None

def solve_cnf(clauses: Iterable[Iterable[int]], num_vars: int
              ) -> Tuple[str, List[int] | None]:
    """
    Implement your SAT solver here.
    Must return:
      ("SAT", model)  where model is a list of ints (DIMACS-style), or
      ("UNSAT", None)
    """
    print(len(clauses), 'clauses')
    output = {True : "SAT",
              False : "UNSAT"}
    sat, model = dpll(clauses, None, num_vars)
    return output[sat], model

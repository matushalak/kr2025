from typing import List, Tuple, Dict
from collections import defaultdict

"""
A CDCL solver implementing clause learning, 
non-chronological backtracking, watched-literals, VSIDS-style branching, phase saving, binary implications, and Luby restarts.
"""

def normalize_clauses(clauses_in: List[List[int]]) -> List[List[int]]:
    """Remove tautologies and duplicate literals; detect empty clause."""
    clauses: List[List[int]] = []
    for c in clauses_in:
        s = set(c)
        if any(-x in s for x in s):
            continue          # tautology
        if not s:
            return [[]]       # immediate UNSAT
        clauses.append(list(s))
    return clauses

def luby(i: int) -> int:
    """Return the i-th Luby sequence value for restart scheduling."""
    k = 1
    while (1 << k) - 1 < i:
        k += 1
    if i == (1 << k) - 1:
        return 1 << (k - 1)
    return luby(i - ((1 << (k - 1)) - 1))

def vabs(l: int) -> int:
    """Absolute variable index for a literal."""
    return l if l > 0 else -l

def init_state(clauses: List[List[int]], num_vars: int) -> Dict:
    """Initialize all global solver structures and attach original clauses."""
    nvars = num_vars
    asg = [-1] * (nvars + 1)
    lvl = [0] * (nvars + 1)
    reason = [-1] * (nvars + 1)
    phase = [1] * (nvars + 1)

    watch_pos: List[Tuple[int, int]] = []
    watchlist: Dict[int, List[int]] = defaultdict(list)
    bin_implies: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    var_act = [0.0] * (nvars + 1)
    var_inc = 1.0
    var_decay = 0.95

    trail: List[int] = []
    trail_lim: List[int] = []
    qhead = 0

    state = {
        "clauses": clauses,
        "nvars": nvars,
        "asg": asg,
        "lvl": lvl,
        "reason": reason,
        "phase": phase,
        "watch_pos": watch_pos,
        "watchlist": watchlist,
        "bin_implies": bin_implies,
        "var_act": var_act,
        "var_inc": var_inc,
        "var_decay": var_decay,
        "trail": trail,
        "trail_lim": trail_lim,
        "qhead": qhead,
    }

    for cid in range(len(clauses)):
        attach_clause(cid, state)

    return state

def lit_val(l: int, state: Dict) -> int:
    """Return literal value under current assignment: 1/0/-1."""
    asg = state["asg"]
    a = asg[vabs(l)]
    if a == -1:
        return -1
    return 1 if (l > 0 and a == 1) or (l < 0 and a == 0) else 0

def attach_clause(cid: int, state: Dict) -> None:
    """Register clause watches and binary implications."""
    clauses = state["clauses"]
    watch_pos = state["watch_pos"]
    watchlist = state["watchlist"]
    bin_implies = state["bin_implies"]

    c = clauses[cid]
    if len(c) == 1:
        i0 = i1 = 0
    else:
        i0, i1 = 0, 1
    watch_pos.append((i0, i1))
    watchlist[c[i0]].append(cid)
    watchlist[c[i1]].append(cid)
    if len(c) == 2:
        a, b = c[0], c[1]
        bin_implies[-a].append((b, cid))
        bin_implies[-b].append((a, cid))

def enqueue(l: int, cid_reason: int, state: Dict) -> bool:
    """Assign literal l if unassigned; return False if conflict."""
    asg = state["asg"]
    lvl = state["lvl"]
    reason = state["reason"]
    phase = state["phase"]
    trail = state["trail"]
    trail_lim = state["trail_lim"]

    v = vabs(l)
    want = 1 if l > 0 else 0
    cur = asg[v]
    if cur != -1:
        return cur == want
    asg[v] = want
    lvl[v] = len(trail_lim)
    reason[v] = cid_reason
    phase[v] = 1 if want == 1 else 0
    trail.append(l)
    return True

def bcp(state: Dict) -> int:
    """Perform Boolean Constraint Propagation; return conflicting clause id or -1."""
    clauses = state["clauses"]
    watch_pos = state["watch_pos"]
    watchlist = state["watchlist"]
    bin_implies = state["bin_implies"]
    trail = state["trail"]
    qhead = state["qhead"]

    while qhead < len(trail):
        l = trail[qhead]
        qhead += 1

        # binary implications of l
        for m, cid_bin in bin_implies.get(l, []):
            if not enqueue(m, cid_bin, state):
                state["qhead"] = qhead
                return cid_bin

        neg = -l
        lst = watchlist.get(neg, [])
        i = 0
        while i < len(lst):
            cid = lst[i]
            c = clauses[cid]
            w0, w1 = watch_pos[cid]

            if c[w0] == neg:
                wi, wo = 0, 1
            elif c[w1] == neg:
                wi, wo = 1, 0
            else:
                i += 1
                continue

            other = c[watch_pos[cid][wo]]
            if lit_val(other, state) == 1:
                i += 1
                continue

            moved = False
            for k, lit in enumerate(c):
                if k == watch_pos[cid][wo]:
                    continue
                if lit_val(lit, state) != 0:
                    if wi == 0:
                        watch_pos[cid] = (k, watch_pos[cid][wo])
                    else:
                        watch_pos[cid] = (watch_pos[cid][wo], k)
                    watchlist[lit].append(cid)
                    lst[i] = lst[-1]
                    lst.pop()
                    moved = True
                    break

            if moved:
                continue

            if lit_val(other, state) == 0:
                state["qhead"] = qhead
                return cid

            if not enqueue(other, cid, state):
                state["qhead"] = qhead
                return cid
            i += 1

    state["qhead"] = qhead
    return -1

def var_bump(v: int, state: Dict) -> None:
    """Increase VSIDS activity of variable v."""
    var_act = state["var_act"]
    var_inc = state["var_inc"]
    var_decay = state["var_decay"]
    nvars = state["nvars"]

    var_act[v] += var_inc
    if var_act[v] > 1e100:
        for u in range(1, nvars + 1):
            var_act[u] *= 1e-100
    state["var_inc"] = var_inc / var_decay

def pick_branch_lit(state: Dict) -> int:
    """Pick next decision literal using VSIDS and phase saving."""
    asg = state["asg"]
    var_act = state["var_act"]
    phase = state["phase"]
    nvars = state["nvars"]

    best_v, best_a = 0, -1.0
    for v in range(1, nvars + 1):
        if asg[v] == -1 and var_act[v] > best_a:
            best_v, best_a = v, var_act[v]
    if best_v == 0:
        for v in range(1, nvars + 1):
            if asg[v] == -1:
                best_v = v
                break
    pol = 1 if phase[best_v] == 1 else -1
    return best_v if pol > 0 else -best_v

def analyze(conflict_cid: int, state: Dict) -> Tuple[List[int], int]:
    """Perform 1-UIP conflict analysis; return learned clause and backjump level."""
    clauses = state["clauses"]
    nvars = state["nvars"]
    trail = state["trail"]
    trail_lim = state["trail_lim"]
    lvl = state["lvl"]
    reason = state["reason"]

    current_level = len(trail_lim)
    seen = [False] * (nvars + 1)
    learnt: List[int] = []
    counter = 0

    c = clauses[conflict_cid]
    for lit in c:
        v = vabs(lit)
        var_bump(v, state)
        if not seen[v]:
            seen[v] = True
            if lvl[v] == current_level:
                counter += 1
            elif lvl[v] > 0:
                learnt.append(lit)

    if counter == 0:
        if not learnt:
            return [], -1
        back_lvl = max(lvl[vabs(l)] for l in learnt)
        return [learnt[0]] + [x for x in learnt[1:]], back_lvl

    q = len(trail) - 1
    p = 0
    while True:
        while True:
            if q < 0:
                return [], -1
            p = trail[q]
            q -= 1
            if seen[vabs(p)] and lvl[vabs(p)] == current_level:
                break

        vp = vabs(p)
        if reason[vp] != -1:
            cid = reason[vp]
            for lit in clauses[cid]:
                v = vabs(lit)
                if seen[v]:
                    continue
                seen[v] = True
                if lvl[v] == current_level:
                    counter += 1
                elif lvl[v] > 0:
                    learnt.append(lit)
            counter -= 1
            if counter == 0:
                break
        else:
            learnt.append(-p)
            counter -= 1
            if counter == 0:
                break

    asserting = -p
    seen_lits = set()
    normalized = [asserting]
    for lit in learnt:
        if vabs(lit) == vabs(asserting):
            continue
        if -lit in seen_lits:
            continue
        if lit not in seen_lits:
            seen_lits.add(lit)
            normalized.append(lit)

    if len(normalized) == 1:
        back_lvl = 0
    else:
        back_lvl = max(lvl[vabs(l)] for l in normalized[1:])
    return normalized, back_lvl

def backjump(to_level: int, state: Dict) -> None:
    """Undo trail assignments down to decision level to_level."""
    trail = state["trail"]
    trail_lim = state["trail_lim"]
    asg = state["asg"]

    target = trail_lim[to_level] if to_level < len(trail_lim) else 0
    while len(trail) > target:
        p = trail.pop()
        asg[vabs(p)] = -1
    while len(trail_lim) > to_level:
        trail_lim.pop()
    state["qhead"] = len(trail)

def add_learned_clause(cl: List[int], state: Dict) -> int:
    """Insert learned clause with optimized second watch selection."""
    if not cl:
        return -1

    clauses = state["clauses"]
    watch_pos = state["watch_pos"]
    watchlist = state["watchlist"]
    bin_implies = state["bin_implies"]
    lvl = state["lvl"]

    if len(cl) >= 2:
        best_j = 1
        best_level = lvl[vabs(cl[1])]
        for j in range(2, len(cl)):
            lvj = lvl[vabs(cl[j])]
            if lvj > best_level:
                best_level = lvj
                best_j = j
        if best_j != 1:
            cl[1], cl[best_j] = cl[best_j], cl[1]

    cid = len(clauses)
    clauses.append(cl)
    if len(cl) == 1:
        watch_pos.append((0, 0))
        watchlist[cl[0]].append(cid)
        watchlist[cl[0]].append(cid)
    else:
        watch_pos.append((0, 1))
        watchlist[cl[0]].append(cid)
        watchlist[cl[1]].append(cid)
        if len(cl) == 2:
            a, b = cl[0], cl[1]
            bin_implies[-a].append((b, cid))
            bin_implies[-b].append((a, cid))
    return cid

def solve_cnf(clauses_in: List[List[int]], num_vars: int) -> Tuple[str, List[int] | None]:
    """Main CDCL loop: propagation, learning, backjumping, branching, restarts."""
    clauses = normalize_clauses(clauses_in)
    if any(len(c) == 0 for c in clauses):
        return "UNSAT", None

    state = init_state(clauses, num_vars)

    for cid, c in enumerate(clauses):
        if len(c) == 1:
            if not enqueue(c[0], cid, state):
                return "UNSAT", None
    confl = bcp(state)
    if confl != -1:
        return "UNSAT", None

    base_conflicts = 256
    restart_i = 1
    next_restart_at = base_conflicts * luby(restart_i)
    conflicts = 0

    while True:
        confl = bcp(state)
        if confl != -1:
            conflicts += 1
            if len(state["trail_lim"]) == 0:
                return "UNSAT", None

            learnt, back_lvl = analyze(confl, state)
            if back_lvl == -1 and not learnt:
                return "UNSAT", None

            backjump(back_lvl, state)
            cid = add_learned_clause(learnt, state)
            if cid == -1:
                return "UNSAT", None
            if not enqueue(learnt[0], cid, state):
                return "UNSAT", None

            if conflicts >= next_restart_at:
                restart_i += 1
                next_restart_at += base_conflicts * luby(restart_i)
                backjump(0, state)
            continue

        asg = state["asg"]
        nvars = state["nvars"]
        all_assigned = True
        for v in range(1, nvars + 1):
            if asg[v] == -1:
                all_assigned = False
                break
        if all_assigned:
            model = [v if asg[v] == 1 else -v for v in range(1, nvars + 1)]
            model.sort(key=vabs)
            return "SAT", model

        lit = pick_branch_lit(state)
        state["trail_lim"].append(len(state["trail"]))
        if not enqueue(lit, -1, state):
            backjump(len(state["trail_lim"]) - 1, state)
            state["trail_lim"].append(len(state["trail"]))
            if not enqueue(-lit, -1, state):
                return "UNSAT", None
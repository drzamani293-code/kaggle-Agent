"""mb1loc_sat.py -- SAT/SMT formulations of the finite statement MB1-loc(a).

THE STATEMENT
-------------
MB1-loc(a):  for all t > a and all p >= 1,

    if  col_0(s) = col_0(s+p)  for every s in [t-a, t]      (H1, "agreement")
    and col_0(t) = 0                                        (H2, "centre zero")
    then col_1(t) = col_1(t+p).                             (C,  "conclusion")

A *counterexample* is a triple (a, t, p) with H1, H2 true and C false, in the
actual rule-30 orbit of the single-cell seed.

THREE MODELS (see SAT_MODEL_SPECIFICATION.md for the formal statements)
-----------------------------------------------------------------------
MODEL A  -- unrestricted rule-30 space-time patch.  The base row is FREE.
            Asks: does *some* rule-30 orbit segment satisfy H1 & H2 & not-C?
            * A-UNSAT  => THEOREM: no rule-30 patch at all can do it, so the
              real orbit cannot either; MB1-loc(a) holds for that p.
            * A-SAT    => says NOTHING about the real orbit.  It is a
              counterexample to the seed-free relaxation only.

MODEL B  -- the actual single-cell orbit, complete backward light cone to
            time 0, base row pinned to the seed.  Exact, but the patch has
            ~(t+p)^2 cells, so it is tractable only for small t.
            NOTE: model B has ZERO free variables -- the seed determines every
            cell.  A solver run on it is a *certificate check*, not a search.

MODEL B'  -- ("B-trunc") the light cone truncated at a base row pinned to the
            actual, independently verified orbit row at time t-a.  Same
            constraints as B, tractable for every witness.  Sound relative to
            the simulator that produced the base row (which is cross-checked
            by four independent engines in the Phase 1 package).

GEOMETRY (shared by all three)
------------------------------
A patch spans times [t_base, t_top] and, at time tau, sites

    [ -(t_top - tau) , 1 + (t_top - tau) ].

At the top this is exactly {0, 1}; it widens by one cell on each side per step
downwards.  This is the backward light cone of the two top cells, so *every*
cell of the patch above the base row is a function of the base row alone: no
unconstrained boundary can influence any queried cell.  (Proof: a cell (tau, j)
in the patch reads (tau-1, j-1), (tau-1, j), (tau-1, j+1); if
-(t_top-tau) <= j <= 1+(t_top-tau) then those three sites lie in
[-(t_top-tau+1), 1+(t_top-tau+1)], which is exactly the patch at time tau-1.)

TRANSLATION INVARIANCE OF MODEL A
---------------------------------
Model A contains no reference to absolute time: its constraints are
"agreement over the a+1 levels below the top-minus-p level", "zero at that
level", "disagreement between two named levels".  Shifting every time index by
a constant maps solutions bijectively to solutions.  Hence **Model A depends
only on (a, p), not on t** -- so the search over t is complete after one
solve per (a, p).  This is asserted as Lemma T in SAT_MODEL_SPECIFICATION.md
and is manifest in `build_patch(model="A")`, which never receives t.

ENCODINGS
---------
ENC1 "direct"  : 8 clauses of width 4 per transition, one forbidding each
                 wrong (l, c, r, y).  No auxiliary variables.
ENC2 "tseitin" : auxiliary u <-> (c OR r) [3 clauses], then y <-> l XOR u
                 [4 clauses].  7 clauses + 1 aux var per transition.
Z3   "native"  : Z3 Bool terms  y == Xor(l, Or(c, r)); Z3 performs its own
                 clausification, so it shares no encoding code with ENC1/ENC2.

Hypothesis constraints carry selector literals so that an UNSAT run yields an
unsat core naming which hypotheses conflict.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import rule30_lab as L

# --------------------------------------------------------------------------
# Rule 30 truth table, taken from the Phase 1 library (verified exhaustively
# against the rule *number* 30 by rule30_lab.verify_local_rule).
# --------------------------------------------------------------------------
RULE30_TABLE = L.rule_table(30)          # index 4l + 2c + r


def rule30(l: int, c: int, r: int) -> int:
    return RULE30_TABLE[4 * l + 2 * c + r]


# --------------------------------------------------------------------------
# Patch geometry
# --------------------------------------------------------------------------


@dataclass
class Patch:
    """Backward light cone of the two cells (t_top, 0) and (t_top, 1)."""

    t_base: int
    t_top: int
    j_top_lo: int = 0
    j_top_hi: int = 1

    def width_at(self, tau: int) -> Tuple[int, int]:
        d = self.t_top - tau
        return (self.j_top_lo - d, self.j_top_hi + d)

    def sites(self, tau: int):
        lo, hi = self.width_at(tau)
        return range(lo, hi + 1)

    def cells(self):
        for tau in range(self.t_base, self.t_top + 1):
            for j in self.sites(tau):
                yield tau, j

    def n_cells(self) -> int:
        return sum(len(self.sites(tau)) for tau in range(self.t_base, self.t_top + 1))

    def contains(self, tau: int, j: int) -> bool:
        if not (self.t_base <= tau <= self.t_top):
            return False
        lo, hi = self.width_at(tau)
        return lo <= j <= hi


def constrained_cells(a: int, p: int, t: int):
    """The cells named by H1, H2 and C, as (kind, tau, j) triples."""
    out = []
    for k in range(a + 1):
        s = t - a + k
        out.append(("H1_agree_%d" % k, (s, 0), (s + p, 0)))
    out.append(("H2_zero", (t, 0), None))
    out.append(("C_disagree", (t, 1), (t + p, 1)))
    return out


def make_patch(model: str, a: int, p: int, t: Optional[int] = None) -> Tuple[Patch, int]:
    """Returns (patch, t_used).

    model "A"  : translation invariant; t is ignored and normalised to t = a,
                 so the base row sits at time 0.
    model "B"  : base row at time 0 (the seed row); t must be given.
    model "Bt" : base row at time t - a, pinned to the real orbit; t required.
    """
    if model == "A":
        t_used = a                      # normalisation; see Lemma T
        return Patch(t_base=0, t_top=t_used + p), t_used
    if t is None:
        raise ValueError("models B and Bt need an explicit t")
    if model == "B":
        return Patch(t_base=0, t_top=t + p), t
    if model == "Bt":
        return Patch(t_base=t - a, t_top=t + p), t
    raise ValueError("unknown model %r" % model)


# --------------------------------------------------------------------------
# CNF construction
# --------------------------------------------------------------------------


@dataclass
class CNF:
    n_vars: int = 0
    clauses: List[List[int]] = field(default_factory=list)
    selectors: Dict[str, int] = field(default_factory=dict)

    def new_var(self) -> int:
        self.n_vars += 1
        return self.n_vars

    def add(self, *lits: int):
        self.clauses.append(list(lits))

    def stats(self) -> Dict[str, int]:
        return {"vars": self.n_vars, "clauses": len(self.clauses)}


def _add_transition_enc1(cnf: CNF, l: int, c: int, r: int, y: int):
    """8 clauses of width 4: forbid every wrong output."""
    for vl, vc, vr in itertools.product((0, 1), repeat=3):
        want = rule30(vl, vc, vr)
        # clause: NOT(l=vl AND c=vc AND r=vr AND y=1-want)
        cnf.add(
            -l if vl else l,
            -c if vc else c,
            -r if vr else r,
            -y if (1 - want) else y,
        )


def _add_transition_enc2(cnf: CNF, l: int, c: int, r: int, y: int):
    """Tseitin: u <-> (c OR r), then y <-> l XOR u."""
    u = cnf.new_var()
    cnf.add(-c, u)
    cnf.add(-r, u)
    cnf.add(c, r, -u)
    cnf.add(-l, -u, -y)
    cnf.add(-l, u, y)
    cnf.add(l, -u, y)
    cnf.add(l, u, -y)


def build_cnf(model: str, a: int, p: int, t: Optional[int] = None,
              encoding: str = "enc1", base_row: Optional[Dict[int, int]] = None):
    """Builds the CNF for one (model, a, p[, t]) instance.

    Returns (cnf, var_of, patch, t_used, selector_names).
    Hypothesis/conclusion constraints are guarded by selector variables so that
    an UNSAT answer produces a core naming the offending constraints.
    """
    patch, t_used = make_patch(model, a, p, t)
    cnf = CNF()
    var: Dict[Tuple[int, int], int] = {}
    for tau, j in patch.cells():
        var[(tau, j)] = cnf.new_var()

    add_tr = _add_transition_enc1 if encoding == "enc1" else _add_transition_enc2
    for tau in range(patch.t_base + 1, patch.t_top + 1):
        for j in patch.sites(tau):
            add_tr(cnf, var[(tau - 1, j - 1)], var[(tau - 1, j)],
                   var[(tau - 1, j + 1)], var[(tau, j)])

    # --- base row pinning -------------------------------------------------
    if model == "B":
        # the single-cell seed: x[0,0] = 1, x[0,j] = 0 for every other site
        for j in patch.sites(patch.t_base):
            v = var[(patch.t_base, j)]
            cnf.add(v if j == 0 else -v)
    elif model == "Bt":
        if base_row is None:
            raise ValueError("model Bt needs base_row")
        for j in patch.sites(patch.t_base):
            v = var[(patch.t_base, j)]
            cnf.add(v if base_row[j] else -v)
    # model A: base row free

    # --- hypotheses and negated conclusion, with selectors ----------------
    sel_names = []
    for name, cell_a, cell_b in constrained_cells(a, p, t_used):
        s = cnf.new_var()
        cnf.selectors[name] = s
        sel_names.append(name)
        if name.startswith("H1"):
            A, B = var[cell_a], var[cell_b]
            cnf.add(-s, -A, B)
            cnf.add(-s, A, -B)
        elif name == "H2_zero":
            cnf.add(-s, -var[cell_a])
        elif name == "C_disagree":
            A, B = var[cell_a], var[cell_b]
            cnf.add(-s, A, B)
            cnf.add(-s, -A, -B)
    return cnf, var, patch, t_used, sel_names


# --------------------------------------------------------------------------
# Solving
# --------------------------------------------------------------------------


def solve_pysat(cnf: CNF, solver_name: str = "cadical153", want_proof: bool = False):
    """Returns dict with status, model (if SAT), core (if UNSAT), proof."""
    from pysat.solvers import Solver

    assumptions = list(cnf.selectors.values())
    res = {"solver": "pysat:" + solver_name, "status": None,
           "model": None, "core": None, "proof_lines": None}

    with Solver(name=solver_name, bootstrap_with=cnf.clauses) as s:
        sat = s.solve(assumptions=assumptions)
        res["status"] = "SAT" if sat else "UNSAT"
        if sat:
            res["model"] = s.get_model()
        else:
            core = s.get_core()
            inv = {v: k for k, v in cnf.selectors.items()}
            res["core"] = sorted(inv.get(abs(x), "var%d" % abs(x)) for x in (core or []))

    if not sat and want_proof:
        # second pass with the selectors asserted as units, to get a DRUP proof
        from pysat.solvers import Cadical153
        hard = list(cnf.clauses) + [[v] for v in assumptions]
        try:
            with Cadical153(bootstrap_with=hard, with_proof=True) as s2:
                s2.solve()
                pr = s2.get_proof()
                res["proof_lines"] = pr if pr is None else len(pr)
                res["proof"] = pr
        except Exception as e:                      # pragma: no cover
            res["proof_error"] = str(e)
    return res


def solve_z3(model: str, a: int, p: int, t: Optional[int] = None,
             base_row: Optional[Dict[int, int]] = None, timeout_ms: int = 0):
    """Native Z3 formulation -- no CNF is shared with the PySAT path."""
    import z3

    patch, t_used = make_patch(model, a, p, t)
    x = {(tau, j): z3.Bool("x_%d_%d" % (tau, j)) for tau, j in patch.cells()}
    s = z3.Solver()
    if timeout_ms:
        s.set("timeout", timeout_ms)
    for tau in range(patch.t_base + 1, patch.t_top + 1):
        for j in patch.sites(tau):
            l, c, r = x[(tau - 1, j - 1)], x[(tau - 1, j)], x[(tau - 1, j + 1)]
            s.add(x[(tau, j)] == z3.Xor(l, z3.Or(c, r)))
    if model == "B":
        for j in patch.sites(patch.t_base):
            s.add(x[(patch.t_base, j)] if j == 0 else z3.Not(x[(patch.t_base, j)]))
    elif model == "Bt":
        for j in patch.sites(patch.t_base):
            s.add(x[(patch.t_base, j)] if base_row[j] else z3.Not(x[(patch.t_base, j)]))

    tracked = {}
    for name, ca, cb in constrained_cells(a, p, t_used):
        if name.startswith("H1"):
            f = x[ca] == x[cb]
        elif name == "H2_zero":
            f = z3.Not(x[ca])
        else:
            f = x[ca] != x[cb]
        tracked[name] = f
        s.assert_and_track(f, name)

    r = s.check()
    if r == z3.sat:
        status = "SAT"
    elif r == z3.unsat:
        status = "UNSAT"
    else:
        status = "UNKNOWN"
    out = {"solver": "z3", "status": status}
    if r == z3.sat:
        m = s.model()
        out["base_row"] = {j: (1 if z3.is_true(m.eval(x[(patch.t_base, j)], True)) else 0)
                           for j in patch.sites(patch.t_base)}
    elif r == z3.unsat:
        out["core"] = sorted(str(c) for c in s.unsat_core())
    return out


# --------------------------------------------------------------------------
# Independent checking of a SAT model
# --------------------------------------------------------------------------


def extract_base_row(cnf_model: Sequence[int], var: Dict[Tuple[int, int], int],
                     patch: Patch) -> Dict[int, int]:
    truth = {abs(v): (v > 0) for v in cnf_model}
    return {j: (1 if truth.get(var[(patch.t_base, j)], False) else 0)
            for j in patch.sites(patch.t_base)}


def simulate_patch(base_row: Dict[int, int], patch: Patch) -> Dict[Tuple[int, int], int]:
    """Evolve the base row upward with an independent rule-30 step, using the
    Phase 1 table.  Cells outside the narrowing cone are never needed."""
    grid = {(patch.t_base, j): base_row[j] for j in patch.sites(patch.t_base)}
    for tau in range(patch.t_base + 1, patch.t_top + 1):
        for j in patch.sites(tau):
            grid[(tau, j)] = rule30(grid[(tau - 1, j - 1)],
                                    grid[(tau - 1, j)],
                                    grid[(tau - 1, j + 1)])
    return grid


def check_patch_constraints(grid, a: int, p: int, t: int) -> Dict[str, bool]:
    """Re-check H1, H2 and not-C directly on a simulated grid."""
    res = {}
    ok = True
    for k in range(a + 1):
        s = t - a + k
        v = grid[(s, 0)] == grid[(s + p, 0)]
        res["H1_agree_%d" % k] = v
        ok &= v
    res["H2_zero"] = grid[(t, 0)] == 0
    ok &= res["H2_zero"]
    res["C_violated"] = grid[(t, 1)] != grid[(t + p, 1)]
    ok &= res["C_violated"]
    res["ALL"] = bool(ok)
    return res


def render_patch(grid, patch: Patch, a: int, p: int, t: int, max_width: int = 120) -> str:
    """Human-readable rendering of the patch, marking the constrained cells."""
    lines = []
    lo_all, hi_all = patch.width_at(patch.t_base)
    lo_all = max(lo_all, -max_width // 2)
    hi_all = min(hi_all, max_width // 2)
    hdr = "".join("0" if j == 0 else ("1" if j == 1 else ("|" if j % 10 == 0 else " "))
                  for j in range(lo_all, hi_all + 1))
    lines.append("          sites %d .. %d   (columns 0 and 1 marked in the header)"
                 % (lo_all, hi_all))
    lines.append("          " + hdr)
    for tau in range(patch.t_base, patch.t_top + 1):
        lo, hi = patch.width_at(tau)
        row = []
        for j in range(lo_all, hi_all + 1):
            if lo <= j <= hi:
                row.append("#" if grid[(tau, j)] else ".")
            else:
                row.append(" ")
        marks = []
        if t - a <= tau <= t:
            marks.append("H1")
        if tau == t:
            marks.append("H2/C")
        if t - a + p <= tau <= t + p:
            marks.append("H1'")
        if tau == t + p:
            marks.append("C'")
        lines.append("t=%7d %s   %s" % (tau, "".join(row), ",".join(marks)))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Real-orbit helpers (for models B and Bt, and for witness re-verification)
# --------------------------------------------------------------------------


def real_orbit_row(time: int, j_lo: int, j_hi: int) -> Dict[int, int]:
    """The actual rule-30 row at `time`, over sites [j_lo, j_hi], from the
    single-cell seed.  Uses the Phase 1 big-integer engine."""
    R = time + 2
    width = 2 * R + 1
    mask = (1 << width) - 1
    row = 1 << R
    for _ in range(time):
        row = ((row << 1) ^ (row | (row >> 1))) & mask
    return {j: (row >> (R + j)) & 1 for j in range(j_lo, j_hi + 1)}


def real_orbit_witness_check(a: int, t: int, p: int) -> Dict[str, object]:
    """Re-simulate the single-cell orbit and check H1, H2, not-C directly."""
    cols = L.columns(t + p + 2, (0, 1))
    c0, c1 = cols[0], cols[1]
    h1 = all(c0[s] == c0[s + p] for s in range(t - a, t + 1))
    return {
        "a": a, "t": t, "p": p,
        "H1_agreement_on_window": h1,
        "H2_centre_zero": c0[t] == 0,
        "col1_t": c1[t], "col1_t_plus_p": c1[t + p],
        "C_violated": c1[t] != c1[t + p],
        "is_counterexample": bool(h1 and c0[t] == 0 and c1[t] != c1[t + p]),
    }


# --------------------------------------------------------------------------
# UNSAT control instance (exercises cores and proofs)
# --------------------------------------------------------------------------
#
# Proposition 6 (Phase 1, WIDTH2_PROOF_RECONSTRUCTION.md Sec. 11) contains the
# case analysis:  if column 0 agrees at lag p at times t and t+1, and
# a_t(0) = 1, then  a_t(-1) = a_{t+p}(-1)  is FORCED, because
#
#     a_t(-1) = a_{t+1}(0) XOR (a_t(0) OR a_t(1))
#
# and  a_t(0) = 1  makes the OR equal to 1 irrespective of a_t(1)
# (Fact F4: rule 30 is not right-permutive).
#
# So the constraint set
#     col_0 agrees at lag p at t and t+1,  a_t(0) = 1,
#     col_(-1)(t) != col_(-1)(t+p)
# must be UNSATISFIABLE even in the *unrestricted* model A.  This is a control:
# it verifies that the encoding can detect genuine forcing, and it exercises
# the unsat-core and DRUP-proof paths that the MB1-loc sweep never reaches.


def build_cnf_prop6_control(p: int, encoding: str = "enc1", forced: bool = True):
    """Control instance.  forced=True  -> a_t(0) = 1, expected UNSAT.
                          forced=False -> a_t(0) = 0, expected SAT
                                          (the neighbour is then free)."""
    t = 1                      # translation invariant; base row at time 0
    patch = Patch(t_base=0, t_top=t + p + 1, j_top_lo=-1, j_top_hi=0)
    cnf = CNF()
    var = {}
    for tau, j in patch.cells():
        var[(tau, j)] = cnf.new_var()
    add_tr = _add_transition_enc1 if encoding == "enc1" else _add_transition_enc2
    for tau in range(patch.t_base + 1, patch.t_top + 1):
        for j in patch.sites(tau):
            add_tr(cnf, var[(tau - 1, j - 1)], var[(tau - 1, j)],
                   var[(tau - 1, j + 1)], var[(tau, j)])

    def sel(name):
        v = cnf.new_var()
        cnf.selectors[name] = v
        return v

    for k, s_ in enumerate((t, t + 1)):                 # agreement at t and t+1
        v = sel("H1_agree_%d" % k)
        A, B = var[(s_, 0)], var[(s_ + p, 0)]
        cnf.add(-v, -A, B)
        cnf.add(-v, A, -B)
    v = sel("H2_centre_%s" % ("one" if forced else "zero"))
    cnf.add(-v, var[(t, 0)] if forced else -var[(t, 0)])
    v = sel("C_left_disagree")
    A, B = var[(t, -1)], var[(t + p, -1)]
    cnf.add(-v, A, B)
    cnf.add(-v, -A, -B)
    return cnf, var, patch, t


# --------------------------------------------------------------------------
# Solver-independent decision by exhaustive enumeration (small instances only)
# --------------------------------------------------------------------------


def decide_by_enumeration(a: int, p: int, max_bits: int = 24):
    """Decide MODEL A for (a, p) by brute force over every possible base row.

    Model A's only free variables are the base row of the light cone, so the
    instance can be decided exactly by enumerating all 2^w rows and evolving
    each one with the Phase 1 rule-30 table.  This uses no SAT solver at all,
    so it is a fully independent decision procedure -- but the base row has
    w = 2(a+p)+2 bits, so it is affordable only for small a + p.

    Returns dict(status=SAT|UNSAT|SKIPPED, witness_base_row=..., n_rows=...).
    """
    patch, t = make_patch("A", a, p)
    sites = list(patch.sites(patch.t_base))
    w = len(sites)
    if w > max_bits:
        return {"status": "SKIPPED", "reason": "base row has %d bits > %d" % (w, max_bits),
                "n_rows": None}
    for code in range(1 << w):
        base = {j: (code >> k) & 1 for k, j in enumerate(sites)}
        grid = simulate_patch(base, patch)
        chk = check_patch_constraints(grid, a, p, t)
        if chk["ALL"]:
            return {"status": "SAT", "witness_base_row": base, "n_rows": 1 << w,
                    "found_at_code": code}
    return {"status": "UNSAT", "witness_base_row": None, "n_rows": 1 << w}


def decide_control_by_enumeration(p: int, forced: bool = True, max_bits: int = 24):
    """Same brute force for the Proposition 6 control instance."""
    t = 1
    patch = Patch(t_base=0, t_top=t + p + 1, j_top_lo=-1, j_top_hi=0)
    sites = list(patch.sites(patch.t_base))
    w = len(sites)
    if w > max_bits:
        return {"status": "SKIPPED", "reason": "%d bits" % w}
    for code in range(1 << w):
        base = {j: (code >> k) & 1 for k, j in enumerate(sites)}
        grid = simulate_patch(base, patch)
        if grid[(t, 0)] != (1 if forced else 0):
            continue
        if grid[(t, 0)] != grid[(t + p, 0)]:
            continue
        if grid[(t + 1, 0)] != grid[(t + 1 + p, 0)]:
            continue
        if grid[(t, -1)] != grid[(t + p, -1)]:
            return {"status": "SAT", "base_row": base, "n_rows": 1 << w}
    return {"status": "UNSAT", "n_rows": 1 << w}


# --------------------------------------------------------------------------
# MODEL B'' -- two-window variant of B' (used for witnesses with large p)
# --------------------------------------------------------------------------
#
# A single light cone spanning [t-a, t+p] has height a+p, so it costs
# ~(a+p)^2 cells: 7.7 million for the a=28 witness (p=2750).  But once the
# base rows are pinned to the (independently verified) real orbit, the two
# time windows [t-a, t] and [t+p-a, t+p] are separately determined and need
# not be joined by a single cone.  Two cones of height a cost ~2a^2 cells --
# 1568 instead of 7.7e6 for a=28.
#
# Constraints across the two windows are exactly the same H1/H2/C.


def build_cnf_bt2(a: int, t: int, p: int, encoding: str = "enc1",
                  base_rows=None):
    """Two-window MODEL B'.  Returns (cnf, var, (patch1, patch2), base_rows)."""
    patch1 = Patch(t_base=t - a, t_top=t)
    patch2 = Patch(t_base=t + p - a, t_top=t + p)
    if base_rows is None:
        lo1, hi1 = patch1.width_at(patch1.t_base)
        lo2, hi2 = patch2.width_at(patch2.t_base)
        base_rows = (real_orbit_row(patch1.t_base, lo1, hi1),
                     real_orbit_row(patch2.t_base, lo2, hi2))
    cnf = CNF()
    var = {}
    for w, patch in ((1, patch1), (2, patch2)):
        for tau, j in patch.cells():
            var[(w, tau, j)] = cnf.new_var()
    add_tr = _add_transition_enc1 if encoding == "enc1" else _add_transition_enc2
    for w, patch in ((1, patch1), (2, patch2)):
        for tau in range(patch.t_base + 1, patch.t_top + 1):
            for j in patch.sites(tau):
                add_tr(cnf, var[(w, tau - 1, j - 1)], var[(w, tau - 1, j)],
                       var[(w, tau - 1, j + 1)], var[(w, tau, j)])
        br = base_rows[w - 1]
        for j in patch.sites(patch.t_base):
            v = var[(w, patch.t_base, j)]
            cnf.add(v if br[j] else -v)

    def sel(name):
        v = cnf.new_var()
        cnf.selectors[name] = v
        return v

    for k in range(a + 1):
        s = t - a + k
        v = sel("H1_agree_%d" % k)
        A, B = var[(1, s, 0)], var[(2, s + p, 0)]
        cnf.add(-v, -A, B)
        cnf.add(-v, A, -B)
    v = sel("H2_zero")
    cnf.add(-v, -var[(1, t, 0)])
    v = sel("C_disagree")
    A, B = var[(1, t, 1)], var[(2, t + p, 1)]
    cnf.add(-v, A, B)
    cnf.add(-v, -A, -B)
    return cnf, var, (patch1, patch2), base_rows


def simulate_bt2(base_rows, patches):
    grids = []
    for br, patch in zip(base_rows, patches):
        grids.append(simulate_patch(br, patch))
    return grids


def check_bt2_constraints(grids, a: int, p: int, t: int):
    g1, g2 = grids
    res, ok = {}, True
    for k in range(a + 1):
        s = t - a + k
        v = g1[(s, 0)] == g2[(s + p, 0)]
        res["H1_agree_%d" % k] = v
        ok &= v
    res["H2_zero"] = g1[(t, 0)] == 0
    ok &= res["H2_zero"]
    res["C_violated"] = g1[(t, 1)] != g2[(t + p, 1)]
    ok &= res["C_violated"]
    res["ALL"] = bool(ok)
    return res


def real_orbit_rows_pair(time1: int, span1, time2: int, span2):
    """Both base rows in a single forward pass (time1 <= time2).

    real_orbit_row() restarts the O(t^2) evolution from the seed on every
    call; the two base rows of a two-window MODEL B' instance are only p
    steps apart, so one pass suffices."""
    assert time1 <= time2
    R = time2 + 2
    width = 2 * R + 1
    mask = (1 << width) - 1
    row = 1 << R
    out1 = None
    for tau in range(time2):
        if tau == time1:
            out1 = {j: (row >> (R + j)) & 1 for j in range(span1[0], span1[1] + 1)}
        row = ((row << 1) ^ (row | (row >> 1))) & mask
    if time1 == time2:
        out1 = {j: (row >> (R + j)) & 1 for j in range(span1[0], span1[1] + 1)}
    out2 = {j: (row >> (R + j)) & 1 for j in range(span2[0], span2[1] + 1)}
    return out1, out2


def build_cnf_a_at_t(a: int, p: int, t: int, encoding: str = "enc1"):
    """MODEL A built at an EXPLICIT absolute time t (no normalisation).

    Only used to test Lemma T (translation invariance) empirically: the
    verdict must not depend on t.  The production path normalises t = a."""
    patch = Patch(t_base=t - a, t_top=t + p)
    cnf = CNF()
    var = {(tau, j): cnf.new_var() for tau, j in patch.cells()}
    add_tr = _add_transition_enc1 if encoding == "enc1" else _add_transition_enc2
    for tau in range(patch.t_base + 1, patch.t_top + 1):
        for j in patch.sites(tau):
            add_tr(cnf, var[(tau - 1, j - 1)], var[(tau - 1, j)],
                   var[(tau - 1, j + 1)], var[(tau, j)])
    for name, ca, cb in constrained_cells(a, p, t):
        s = cnf.new_var()
        cnf.selectors[name] = s
        if name.startswith("H1"):
            A, B = var[ca], var[cb]
            cnf.add(-s, -A, B)
            cnf.add(-s, A, -B)
        elif name == "H2_zero":
            cnf.add(-s, -var[ca])
        else:
            A, B = var[ca], var[cb]
            cnf.add(-s, A, B)
            cnf.add(-s, -A, -B)
    return cnf, var, patch


def check_cone_containment(patch: Patch) -> bool:
    """Lemma C, verified cell by cell on a concrete patch."""
    for tau in range(patch.t_base + 1, patch.t_top + 1):
        for j in patch.sites(tau):
            for dj in (-1, 0, 1):
                if not patch.contains(tau - 1, j + dj):
                    return False
    return True

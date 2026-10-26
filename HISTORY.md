# History

How the package got where it is. The current design is in [CODE.md](CODE.md);
the sections here follow its headings. Dates are those of the commits that
recorded each item in the documentation.

- [Scope](#scope)
- [Data model and units](#data-model-and-units)
- [Type design](#type-design)
  - [The EOS interface](#the-eos-interface)
- [State policy](#state-policy)
- [Discipline in `core/`](#discipline-in-core)
  - [The one stack buffer](#the-one-stack-buffer)
  - [Two stages](#two-stages)
- [Deliberate deviations from the C++](#deliberate-deviations-from-the-c)
- [Testing](#testing)
  - [Static analysis](#static-analysis)
- [Precision and devices](#precision-and-devices)
  - [Not-quite-transcendental functions](#not-quite-transcendental-functions)

## Scope

2026-10-24. Added `bin/`, which runs the C++ tools that EntropyEOS_jll ships,
so that repairing a table no longer needs a C++ build. The tools are wrapped,
not translated; repair stays out of the package. A package extension was the
alternative, but it would have offered a Julia function around what is a
command-line step done once per table, and `bin/` needs no API.

## Data model and units

2026-10-26. Ported the C++ 1.2.0 ("M3k", PR eschnett/EntropyEOS#2, not yet
merged; ported at its head, 1743044): `BuildOptions.trim_t_hi_rows` and
`crop_table`, the reader keeping the repair's `eeos_t_hi_rows_dropped` record,
the u-high ratio lock on L's secondary tail track, and the cs² flags with
`signal_cs²`. The PR first (ad31730) dropped the hottest T row only from the
fit, with a build default of 1, and its `eos_repair` wrote that row back
unrepaired, since the causal cap ran only on the rows kept. Fitted whole, as
this package and 1.1.0 do, such a file was worse than a 1.1.0 one. Measured
2026-10-25 without the ratio lock, by the census of
[CODE.md](CODE.md#where-the-answer-is-not-unique) (32,768 states per table,
cold; failed / converged on another state):

| table | 1.1.0 repair | ad31730, fitted whole | ad31730, top row cropped | top row cropped from the 1.1.0 file |
| --- | --- | --- | --- | --- |
| LS220 | 0 / 17 | 4 / 29 | 0 / 2 | 0 / 2 |
| SRO LS220 | 0 / 13 | 1 / 35 | 1 / 7 | 0 / 8 |
| DD2 | 7 / 210 | 0 / 26 | 0 / 0 | 3 / 109 |
| SFHo | 5 / 179 | 0 / 2 | 0 / 1 | 2 / 109 |

So on LS220 and SRO LS220 dropping the row is the whole effect, while DD2 and
SFHo also need the seam blend. 1743044 moved the drop into the repair and
changed the build's default to 0, so that every row of a repaired file is a
repaired row; this port follows it. Tables repaired by 1743044 give exactly the
cropped ad31730 column, and the ratio lock changes none of those counts. On the
published and 1.1.0-repaired tables the lock moves DD2's and SFHo's counts by
at most 2 states (DD2 7 / 210 to 5 / 212).

The cs² flags are bits 18 and 19, one above the C++'s, because bit 17 is
`FLAG_POL_W_CEILED` here. The same change corrected CODE.md's claim that the
C++ numbers the policy bits the same way. It has not since this package's 2.0.0
replaced the policy layer.

2026-10-26, later. PR eschnett/EntropyEOS#2 merged as 94d09ec, tagged v1.2.0,
with the same tree as 1743044, and EntropyEOS_jll 1.2.0+0 ships it. `bin/` now
requires the JLL at 1.2. The JLL's `eos_repair` writes tables that differ from
the local 1743044 build's by at most 4e-14 relative, presumably because the JLL
is built with `-ffp-contract=off`. Re-measured on the JLL's tables, the census
changes by one state: SRO LS220 goes from 1 / 7 to 1 / 6 cold, and SFHo from
0 / 2 to 0 / 1 warm. CODE.md quotes the JLL's numbers.

## Type design

### The EOS interface

2026-09-24. Introducing `AbstractEOS` changed no arithmetic: a hash over 3,000
states of every entry point's output was bitwise unchanged.

2026-10-23. Added the optional `evaluate_windowed`. Every con2prim Newton step
called `srange_extended` at the density `evaluate` had just used, and on the
table that repeated the two samples bracketing its temperature solve: 2 of
about 8 spline evaluations per step. `Residuals` now carries the window. Cold,
warm and `con2prim_safe` outputs were bitwise unchanged on 32,768 states in
each of 18 cases (six tables; s or log T uniform at Float64, s uniform at
Float32). Warm calls with a 1e-3-perturbed guess got about 10% faster on the
host, single-threaded; single cases scatter from 0 to 28%.

## State policy

2026-10-21. The entropy-floor branch of `con2prim_safe` clamped w into
`[0, w_cap]` and ρ into `[ρ_atm, ρ_ceiling]` without setting any flag, because
the projection that follows finds the clamped state in range. That was found
in TreeGRRMHD, which has to put a point at rest whenever the repair adds rest
mass, and had to detect it from the returned state (`ρ·cosh w > D`). On 400
thin states driven into that branch, 115 had ρ raised and 79 had w capped, all
reported only as `S_FLOORED`; on 400 dense states with a low ceiling, 209 had ρ
lowered.

The fix splits `FLAG_POL_ρ_CLAMPED`, as was already done for s: bit 13, which
only the ceiling ever set, became `FLAG_POL_ρ_CEILED`; the new
`FLAG_POL_ρ_FLOORED` took bit 16, since bits 8–15 were full and 6–7 are
reserved for solver flags; and `FLAG_POL_ρ_CLAMPED` became the mask of both,
which is what its documentation had always said it meant. `FLAG_POL_ANY` grew
from `0xff00` to `0x1ff00`. The branch also sets `W_CAPPED`. Returned states are
bitwise unchanged on all 800 states; only the flags differ. Code that tests
`flags & FLAG_POL_ρ_CLAMPED` now also sees the floor; code that compares a flag
word with `==` against `FLAG_POL_ρ_CLAMPED` must switch to `FLAG_POL_ρ_CEILED`.
The same change was made in the C++. Version 1.1.0.

Putting the point at rest inside EntropyEOS was considered and left to the
caller, as it may be specific to one evolution scheme.

2026-10-22. Proposed the
[two-level projection](CODE.md#state-policy), prompted by
TreeGRRMHD's request EOS-9 (hot states written at the collapse ceiling). Within
it, keeping D_Y rather than Yₑ when D changes was considered: it would remove
the need for any default Yₑ, even for D ≤ 0. Rejected, because it treats
electrons and the implicit neutrons differently: matter added or removed would
carry no electrons.

The same day, still within the proposal, decided by Erik:

- At the ρ_min face with D < ρ_min, the state is put at rest. Keeping S was
  rejected: it is discontinuous at D = ρ_min, where a state just above is slowed
  almost to rest and one just below keeps its full momentum, and it adds more
  mass.
- At the w_cap face τ is kept, and the cell heats. Keeping s, as 1.1.0 does,
  was rejected as inconsistent with keeping every conservative a face does not
  free; the s_max face bounds the heating.
- Faces are tested with a slack, without which no output on a face would be a
  fixed point.
- `check_con_state` promises no false positives, rather than predicting the
  faces, which it cannot do exactly without running the projection.
- Level 2 keeps level 1's flag bits. EOS-9's Q4, an atmosphere carrying only its
  cause, was rejected because it loses what level 1 did; `CEILING` is retired
  as redundant with `ATMOSPHERE | ρ_CEILED`.

Measured the same day on the five real tables (`study/projection_faces.jl`),
which settled two more points: the face root find iterates on φ = tanh w − V,
since f₁ is not monotone where V > 1, and the s faces are diagnosed only below
ρ_max, since f₂ is not monotone in s in LS220's extrapolation above it. The
measurement also found that `inner_solve_w`'s unique root at fixed s does not
hold where a table has cs² ≥ 1.

For where a face has several roots, a rule was decided the same day and then
withdrawn: return the lowest w at which φ rises through zero, in the face find
and `inner_solve_w` alike, so that the answer depends on the state alone.
Finding the lowest root needs a scan, so it needed a cheap test for where
several roots are possible. One was found (`study/root_certificate.jl`): every
crossing rises wherever cs²·tanh² w < 1, and a per-table bound on cs² per cell
of (log ρ, Yₑ), about 1,700 numbers, certified 74–94% of sampled states (95–99.7%
at W < 2) with no false certificate, once it sampled the temperature seams
densely. Still, the rule was rejected as not worth it: several roots occur in
0–3.7% of states, mostly fast flow through acausal parts of the tables; it would
have needed the bound array, a scan, and an undesigned re-solve; and for
0.35–1.5% of valid states it would have returned another state than the one
whose conservatives went in, where a warm-started solve stays on its branch.
The fixed-point invariant, which motivated it, holds without it once stated
with the output as the guess. Considered and rejected along the way:

- the root closest in τ, the one conservative a face frees: it needs τ at every
  root, and it picks differently on the two faces;
- the root nearest the pre-projection w: after a failed solve it depends on that
  solve's history;
- replacing each s face by a monotone envelope of its isotherm, which would
  remove region (c)'s extra roots by construction: too much machinery (a
  precomputed window, a policy window separate from `srange`) for an edge case.

2026-10-23. Decided that the C++ follows the two-level projection, as it did
for 1.1.0.

The same day, rejected a rescue for `NOT_CONVERGED`: a guarded bisection in s,
bracketed by the diagnosis. Measured by `study/cold_failures.jl`, every cold
failure on a valid state lies in region (a) or (b), where the inner root is not
unique, and the bisection recovered none of them, because G jumps between
roots. The measurement also showed that the endpoint diagnosis misclassifies
most of these valid states, which qualifies the earlier finding that it is
reliable. Since (a) and (b) are wrong physics, to be fixed by repairing the
tables, this was accepted as a limitation rather than resolved. Leaning on warm
starts was not an option: running without one is a design goal. A rescue that
follows each root through an ordered sweep in s was considered and left as a
possible remedy, untested.

2026-10-23. Implemented the two-level projection, as version 2.0.0. Decided by
Erik while implementing it:

- **The face rule.** The active set is decided by the diagnosis along the
  entropy edges, always; `con2prim`'s result is used only when it is valid. The
  alternative was to take the converged solution, clamp the ρ and w faces from
  it first and the s faces after. It is closer to 1.x, but a failed solve needs
  the diagnosis anyway, so two neighbouring states, one converged and one not,
  could take different faces. Read as a rule, the diagnosis picks the active set
  in which every active face moves its freed conservative the way the face
  table says.
- **A converged root outside the range** where the diagnosis finds a valid one
  inside gives `NOT_CONVERGED`. Projecting the converged root instead would keep
  dense matter in region (a) out of the atmosphere, but needs a second decision
  path.
- **Names.** `ProjectionOptions(ρ_min, ρ_max, w_cap)` for level 1, built by
  `default_projection`, which replaces `default_policy`; `PolicyOptions` now
  holds level 2's `s_atm`, `ye_atm`, `atm_trigger` and
  `collapse_to_atmosphere`. `policy_derive_bounds`, `policy_atmosphere`, `D_max`
  and `τ_max` are gone.
- **s_min** for the fixed state of `NONFINITE` and `NOT_CONVERGED`, and as the
  default atmosphere entropy. 1.x used the middle of the window, which at ρ_min
  is radiation-dominated.
- **Two verbs for the face flags**, FLOORED and CEILED, and the policy bits
  reallocated into one block in the order of CODE.md's tables:

  | 1.x | bit | 2.0 | bit |
  | --- | --- | --- | --- |
  | `ATMOSPHERE` | 8 | `ATMOSPHERE` | 8 |
  | `CEILING` | 9 | removed; `ATMOSPHERE \| ρ_CEILED` says the same | |
  | `S_FLOORED` | 10 | `S_FLOORED` | 15 |
  | `S_CEILED` | 11 | `S_CEILED` | 16 |
  | `W_CAPPED` | 12 | `W_CEILED` | 17 |
  | `ρ_CEILED` | 13 | `ρ_CEILED` | 12 |
  | `YE_CLAMPED` | 14 | `YE_FLOORED`, `YE_CEILED` | 13, 14 |
  | `NONFINITE` | 15 | `NONFINITE` | 9 |
  | `ρ_FLOORED` | 16 | `ρ_FLOORED` | 11 |
  | | | `NOT_CONVERGED` | 10 |

  The mask `FLAG_POL_ρ_CLAMPED` was dropped, and `FLAG_POL_ANY` is `0x3ff00`.

Settled by the implementation:

- The ρ_max face is solved as a nested one-dimensional problem, outer in s,
  with the rapidity carrying S found by arithmetic at fixed (ρ, s), rather than
  by the two-dimensional con2prim at fixed ρ the proposal anticipated.
- A face state that solves the input to the tolerance is returned before
  `NOT_CONVERGED`. Found by the Float32 smoke test on `HybridEOS`: there G is
  flat to within the tolerance across the window, `con2prim` converged at s far
  outside it, and 46 of 1,000 valid states went to `NOT_CONVERGED`, with a
  density error of order 1, where 1.x had floored s and kept ρ.
- Each solver has one call site in `con2prim_safe`
  ([CODE.md](CODE.md#discipline-in-core)). The first version called the face
  solves from every branch of the diagnosis; its kernel crashed Apple's Metal
  shader compiler (XPC connection interrupted, after two minutes), where 1.1.0's
  compiled in 24 s. Its LLVM IR was 47,000 lines against 1.1.0's 35,000, and
  marking the solvers `@noinline` did not help, since the backend inlines them
  anyway. Restructured, the IR is 36,000 lines, and the kernel compiles in 15 s.
  The same restructuring removed a `throw` that `Base.sqrt` had left in
  `w_from_momentum`.
- On an H200, a kernel calling both levels faults with an illegal address on
  invalid states under CUDA.jl 6.4.2 ([CODE.md](CODE.md#open-items)). Tried
  without effect: `@noinline` on the face solvers, `@inline` on `apply_policy`
  and on `pol_package`, and a 64 kB stack limit. `julia -g2` cannot report the
  exception, because `ptxas` rejects the Unicode identifiers in debug PTX.
- The face slack was widened from 4·`con2prim_tol` (4·10⁻¹²) to 10⁻⁷ at
  Float64, decided by Erik. At 4·10⁻¹², an output fed back without a guess was
  flagged again in 21% of 4,094 projected synthetic-table states, because a
  cold solve returns a state on a face only to its conditioning. 1.x had no
  slack and no hysteresis on the faces, only the band between `atm_trigger`
  and ρ_atm, which made an atmosphere state check clean rather than stop it
  being reset again. On the same states 1.1.0 re-flagged 33% (54 of them
  atmosphere resets), and 110 even with the output's primitives as the guess,
  where 2.0 re-flags none. Writing states a little inside the range, a true
  hysteresis, was considered and not taken: a returned state would no longer
  lie on its face. The remaining re-flags are measured by
  `study/cold_resolve.jl` ([CODE.md](CODE.md#the-contract)).
- A review of the implementation (by a separate agent) found that
  `check_con_state` broke its no-false-positive promise. It tested the sign of
  the energy residual on the edges themselves, while `con2prim_safe` accepts
  states within the slack, so it reported 40% of states lying exactly on an edge.
  It now tests edges moved outward by the slack, against a per-type residual bar
  (`check_f2_tol`). Testing only against the tolerance, without the shift, still
  left 41 of 3,000 at Float64. The review also led to:
  - the fixed state taking the clamped Yₑ, so that a `NOT_CONVERGED` with a Yₑ
    flag lies on that face;
  - a face solve that never gets a finite residual reporting NaN, and so
    `NOT_CONVERGED`, instead of the window's lower edge;
  - step 5 considering the ρ_max face state as well;
  - the fast path returning the input's B² bit for bit, including −0.0;
  - one call site for `check_con_state`'s edge root find;
  - `policy_violations` bars on kept conservatives tightened from 10⁻⁸ to
    10⁻¹⁰, with kept S measured against z rather than D (on hot states z
    exceeds D by up to 10⁸), and a bar on S_perp widened by eps/sin² for the
    cos_vB representation limit rather than a hard exemption.
- The proposal said a lowered S "may turn slightly". It does not: taking the
  velocity direction from the input momentum at the new state's own enthalpy
  density makes the returned S exactly parallel to the input.

What 2.0 changes for existing inputs. On the converged path, 1.1.0 clamped
primitives and regenerated every conservative, where the face rules keep each
conservative a face does not free: w_cap keeps τ where 1.1.0 kept s, ρ_max keeps
S and τ where 1.1.0 kept w and s, and the s faces keep S where 1.1.0 kept w.
After a failed solve, a state at the ρ_min face with D ≥ ρ_min is slowed down
rather than given mass, and an energy above the hottest state keeps D and S and
lowers τ rather than going to ρ_max. Only the s_min branch of a failed solve is
essentially unchanged. τ no longer short-circuits the solve (`τ_max` came from a
5×3 grid and stopped states the table represents). TreeGRRMHD's EOS-9 request
is largely subsumed: its edge-consistent root find, its guards on the endpoint
solves, the shared diagnosis with `check_con_state`, and dropping the `τ_max`
test carry over; its flag rules are replaced by the face flags, and its Q4 by
level 2 keeping level 1's bits.

2026-10-24. Added `inside_window`, decided by Erik after TreeGRRMHD's upgrade to
2.0. TreeGRRMHD's reconstruction tested each face state against the exact
window and, where one failed, used the cell's own value. A cell that
`con2prim_safe` had returned unflagged 5.5·10⁻⁹ of the window below s_min (a
cold re-solve of a state written onto that face) failed the test, and so did
its fallback, so the face went to first order. Under ARS(4,4,3), whose fourth
explicit weight is negative, that injected mass, and 48 of 48 perturbed
cut-off atmosphere tubes blew up, against none on 1.1.0, which clamped such a
root onto s_min and flagged it (TreeGRRMHD's measurements). The Yₑ face has the
same exposure, since Yₑ one ulp out of range is accepted. Considered and
rejected:

- exporting `face_slack`: it carries two conventions, relative to the window's
  width for s and one ulp for Yₑ, and would make its value part of the API;
- clamping an accepted root onto the window, with flags 0 and the input
  conservatives: the primitives would no longer reproduce the returned
  conservatives to the tolerance. For an ideal-gas state half a slack above
  s_max the clamp moves τ by 1.3·10⁻⁶, relative, and fed back with the clamped
  output as its guess the solve needs a Newton step, where it needs none with
  the unclamped one.

## Discipline in `core/`

### The one stack buffer

2026-09-23. The `MVector{33,T}` buffer in `bracket_scan` was the design's
biggest open risk; it is resolved by the kernel compiling and running on Metal.

### Two stages

2026-10-23. Split `con2prim` and `con2prim_safe` each into a first and a
second stage, so that a GPU caller can run the fallback as a second kernel over
only the states that need it. Before the direct cold seed, a host model from
per-state spline counts on LS220 put the cost of warp divergence at 4.4 times
the work, and at 2.4 times with the fallback split off; after it the seed was
no longer what warps waited for, so the device gained 1.4–1.6 times where the
host gained 2.6.

The fused functions are now the two stages in sequence. Their outputs were
bitwise unchanged on the 18 cases of 32,768 states used for the seed. The
second stage rebuilds the first's iterate from `(s, w, u)`; the test suite
asserts that finishing any first stage, done or not, gives the fused result
bit for bit, on the synthetic table with starved options and on the real
tables' cold states. Storing the iterate itself was rejected: `Residuals` is
~280 bytes per state against 32, and rebuilding it costs one evaluation on a
pending state, against a fallback's hundreds.

Measured on an H200 by `study/device_parity.jl`, 2²⁰ states per table, the
fused and the two-kernel form in the same run (a repeat of LS220 and SRO LS220
reproduced every rate to 5%). Millions of states per second, and the number of
states the second stage ran on:

| Float64 | cold, fused | cold, 2 kernels | second stage | `con2prim_safe`, fused | 2 kernels | second stage |
| --- | --- | --- | --- | --- | --- | --- |
| synthetic | 64.2 | 67.6 | 0 | 14.7 | 41.9 | 122,170 |
| LS220 | 1.5 | 3.8–4.0 | 23,115 | 1.3 | 2.5 | 142,169 |
| SRO LS220 | 2.3 | 5.1 | 24,751 | 1.8 | 3.7 | 143,644 |
| DD2 | 1.8 | 4.8 | 34,648 | 1.5 | 3.1 | 150,458 |
| SFHo | 1.4 | 2.1 | 5,886 | 1.3 | 1.9 | 127,637 |

At Float32 cold went from 5.5 to 13.1 on LS220, 9.1 to 16.8 on SRO LS220, 8.3
to 15.7 on DD2 and 14.5 to 21.1 on SFHo. The compaction (`findall` on the
device) took 0.1–0.3 ms, against 80–160 ms for the first stage. On the device
the two-kernel results were bitwise those of the fused kernel for every state
of every table at both precisions, and every gate the fused kernel passed, it
passed too. Against the commit before the direct cold seed, run in the same
job, cold Float64 `con2prim` on LS220 went from 1.1 to 4.0.

What remains is the second stage itself. On SFHo its 5,886 states, 0.6%, of
which 1,853 end in failure, took 410 ms against the first stage's 82. In
`con2prim_safe` the second stage also carries every state outside a face,
12–14% of this sampling: w is uniform in [0, 6), and 11.7% of it lies above
the default cap acosh(100) ≈ 5.30.

2026-10-25. The confined search had made warm Float64 `con2prim` 2–4% slower
on an H200. In the parity sampling a warm start converges with no Newton step,
so a warm call is one residual evaluation plus the density floor. A dedicated
benchmark (CUDA.jl, 40 launches of 2²⁰ states, minimum time, the old and new
trees and variants alternating in one job on cn113) separated the parts. Times
in ms, warm, on the synthetic table / DD2 / LS220:

| | fused kernel | first stage alone |
| --- | --- | --- |
| before the confined search | 3.04 / 3.42 / 3.44 | 2.80 / 3.14 / 3.20 |
| confined search | 3.15 / 3.51 / 3.51 | 2.94 / 3.28 / 3.31 |
| `w_floor` ≡ 0 | 3.10 / 3.47 / 3.47 | |
| ρ_max precomputed | 3.14 / 3.49 / 3.51 | |
| floor known by a flag, `acosh` skipped | 3.21 / – / – | |
| floor recomputed, `acosh` always | 3.20 / 3.57 / 3.57 | 2.93 / 3.20 / 3.22 |
| floor recomputed, `acosh` skipped (kept) | 3.16 / 3.52 / 3.52 | 2.85 / 3.14 / 3.15 |

Repeats in the same jobs reproduced single entries to 0.3%. Every variant's
outputs, warm and cold, hashed identical to the confined search's. Findings:

- **Not the `exp10`.** Baking ρ_max in as a literal changed nothing.
- **The `acosh`, in the first stage.** 7.5–10.4% of the sampled states have
  D > ρ_max, which puts one in 92–98% of warps. Skipping `acosh` when a
  polynomial lower bound on cosh already shows the warm start inside the box
  made the first stage 3–5% faster, within 2% of before either way.
- **Code generation, in the fused kernel.** Every kernel uses 255 registers
  and spills, 3.2 KB per thread for the first stage and 8.5 KB fused. With
  `w_floor` forced to zero the new code is the old algorithm bit for bit (its
  cold outputs hashed identical to the old tree's), and the fused kernel was
  still 2.3% slower. No variant moved the fused kernel by more than this
  noise-like ±2%, in either direction. A call-site `@noinline` on the second
  stage changed neither time nor local memory.

The kept variant changes no result and leaves cold times unchanged; it helps
the first stage, which the two-kernel form runs over every state.

## Deliberate deviations from the C++

2026-10-23. Fixed cold `con2prim` failing on states on an edge of the entropy
window. The two-level projection's `study/cold_resolve.jl` found it: 43 of 49
projected outputs that a cold re-solve flagged were states put at rest at ρ_min
on the s_max face, reported as `failed_no_bracket` although the best iterate
returned was the state itself. It is not specific to the projection: on the
synthetic table, cold solves at rest on either edge failed 14 of 120 times
(22 of 50 at ρ_min), and 3 of 120 at w = 1e-3. The C++, run on the same
states, fails 17 and 2: the same defect. The analytic EOSs, which have no
iterative T-solve, never failed.

The suspected cause, a singular Newton Jacobian at w = 0, was wrong: the
determinant there is −0.43. Newton stalled because the T-solve returned a warm
start unchanged whenever it was within its residual tolerance, 1e-12·|s|. That
leaked up to 2.4e-12 into U, and some 20 times con2prim's tolerance into the
energy residual, so Newton's small steps in s left the residual frozen just
above its tolerance; at random interior states this sent 2% of cold solves to
the fallback. The fallback then failed because `bracket_scan` emits `s_phys_lo`
and `s_phys_hi` twice. With the root on an edge, the two copies' solves differed
only in warm start, could disagree in sign, and that empty pair won, since its
midpoint is the anchor; re-solving both ends from one warm start found no sign
change. 171 of 360 scans at edge states returned such a pair.

Three changes, all in `core/`
([CODE.md](CODE.md#deliberate-deviations-from-the-c)): a final Newton step in
the T-solve, skipping empty pairs in the scan, and accepting an endpoint within
tolerance as a bracket. The T-solve step alone removes every failure and every
fallback in the samples below; the other two harden the fallback, which on the
same edge states, with the Newton disabled and a perturbed warm start, now
fails 5 of 960 instead of 22. That remainder is a resolution floor, recorded
under [CODE.md § Open items](CODE.md#open-items).

| synthetic table, cold | before | after |
| --- | --- | --- |
| at rest on an edge, failed / fallback | 14 / 49 of 120 | 0 / 0 |
| w = 1e-3 on an edge, failed / fallback | 3 / 66 of 120 | 0 / 0 |
| random states, fallback | 44 of 2000 | 0 |
| random states warm, fallback | 2 of 2000 | 0 |

On the real tables no edge state failed before or after; there the change shows
as fewer fallbacks. With the audit's sampling (20,000 warm, 2,000 cold, a
separate harness and seed, so its quantiles do not reproduce the
[validation table](CODE.md#validation-against-the-c)'s):

| warm Newton / fallback / failed | before | after |
| --- | --- | --- |
| LS220 | 19950 / 50 / 0 | 20000 / 0 / 0 |
| SRO LS220 | 19949 / 51 / 0 | 20000 / 0 / 0 |
| DD2 | 19932 / 64 / 4 | 19978 / 18 / 4 |
| SFHo | 19925 / 74 / 1 | 19977 / 21 / 2 |

Cold failures were unchanged (LS220 1, SRO LS220 2, others 0), cold fallbacks
fell from 8–26 to 0–20, and the density round trip's medians and p99s moved by
less than 10%. Of the tails only SFHo's moved more, and down: p99.9 from 9.0e-9
to 6.5e-9, maximum from 0.26 to 0.15.

Rejected: tightening `tsolve_residual_tol` instead, which buys the same
accuracy with extra T-solve iterations on every hot-path `evaluate`, where the
final step costs none; and removing the duplicate candidates at emission, which
changes the scan's measured sampling where skipping empty pairs does not.

2026-10-23. The cold seed solves for log T directly, and makes 5 passes rather
than 3. On an H200, cold `con2prim` ran at 1.1 million states per second
against 324 warm ([Precision and GPUs](docs/src/precision.md#throughput)).
Counting spline evaluations on the host, on LS220 with the device-parity
sampling, a cold state that Newton solves cost 230 of them, and 185 of those
were the seed: three passes of a Newton in s, each step a full `evaluate` with
its own temperature solve. `entropy_from_energy` on the table instead solves
the energy field for log T, which is monotone for the same reason σ is, and
samples σ once at the root.

With three passes the direct seed made cold solves half as expensive, but on
LS220 (65,536 states) it sent 1,709 states past Newton where the nested seed
sent 1,531, took 5.31 Newton steps on average against 4.89, and its density
round trip's p99 was 9.7e-9 against 3.5e-9. For energies inside the table
the two solve U(s) = ε equally well, to about 1e-14; where their seeds part was
not established. With five passes: 1,498, 4.04 and 2.0e-9, still at less than half the cost. More
iterations of the nested seed (40 rather than 16) changed nothing.

Measured on the host against the previous commit, 32,768 states per table,
single-threaded, Float64. "s" samples the entropy uniformly in the physical
window (2% margins), "log T" the temperature; w is uniform in [0, 6) and half
the states are magnetized. Fallbacks count states that Newton did not solve and
the fallback did:

| cold, s / log T | µs per state | failed | fallbacks | Newton steps |
| --- | --- | --- | --- | --- |
| synthetic | 11.3→4.4 / 10.9→4.3 | 0→0 / 0→0 | 0→0 / 0→0 | 1.28→0.46 / 1.28→0.46 |
| LS220 | 22.3→9.5 / 51.7→14.5 | 720→679 / 1222→1224 | 42→36 / 712→456 | 4.89→3.74 / 5.26→3.10 |
| SRO LS220 | 23.7→9.0 / 59.5→15.8 | 797→754 / 1457→1472 | 17→11 / 881→563 | 4.93→3.78 / 6.33→3.18 |
| DD2 | 23.0→11.1 / 49.3→13.0 | 7→5 / 3→2 | 1286→1066 / 2333→2123 | 5.78→4.41 / 5.34→3.52 |
| DD2 repaired | 22.3→10.8 / 52.8→12.4 | 7→5 / 1→2 | 1308→1072 / 2330→2130 | 5.78→4.41 / 5.34→3.52 |
| SFHo | 22.4→9.1 / 51.9→14.3 | 65→66 / 250→237 | 227→137 / 710→507 | 4.73→3.40 / 4.10→2.20 |

At Float32 (s sampling) cold solves went from 15–18 µs to 4–6 µs, with
failures 0–378 per table moving by at most 35, all but one of them down.
`con2prim_safe` got 1.8–3.1 times faster and left the same states unsolved, with no
non-finite output. Warm solves are bitwise unchanged, since the seed runs only
without a guess.

The density round trip's p99 got worse on no real table and improved on most,
by up to a factor of 6.7 (LS220, log T: 1.2e-8 to 1.8e-9); its median grew by
up to a factor of 1.4 (LS220, s: 1.7e-13 to 2.4e-13). On
the synthetic table its tail grew: p99 7.3e-11 to 1.7e-10, maximum 1.5e-9 to
7.2e-9. There most seeds already meet the tolerance, and 0.46 Newton steps are
taken on average rather than 1.28. Forcing one Newton step after every seed
left those quantiles exactly where they were, so they are where the
tolerance's admissible set lands on ill-conditioned states, not unfinished
iterations; the forced step was not kept.

The analytic EOSs keep the nested seed. Five passes make their cold solves
17% (ideal gas) and 45% (hybrid SLy) more expensive, take fewer Newton steps,
and on the hybrid send 41 rather than 53 of 20,000 states to the fallback.

`test_state_policy.jl`'s `NOT_CONVERGED` test starved the solver with no Newton
steps, and with five direct passes the seed alone solved every state it drew.
It now also limits the seed to one pass.

The two studies that re-solve states cold were rerun, on the previous commit
and on this one. `study/cold_failures.jl`, 20,000 valid states per table: cold
failures went from 8, 16, 10 and 3 (LS220, SRO LS220, DD2 either version, SFHo)
to 4, 9, 10 and 7, all still in the acausal regions (a) and (b) of
[CODE.md](CODE.md#where-the-answer-is-not-unique); `con2prim_safe` had
projected 33 of the 37 onto a face and returned 4 as `NOT_CONVERGED`, and now
projects 22 of 30 and returns 8. `study/cold_resolve.jl`, 5,000 pushed-off
states per table, reproduced the table in [CODE.md](CODE.md#the-contract)
exactly before the change; after it, states re-flagged cold fell on every real
table (LS220 146 to 140, SRO LS220 163 to 148, DD2 137 to 134, SFHo 68 to 63,
DD2 repaired unchanged at 135), and so did the counts above the old slack. The
superseded rows:

| table | excess p99 | max | > 4·tol (old slack) | > 10⁻⁷ | re-flagged cold |
| --- | --- | --- | --- | --- | --- |
| synthetic | 6·10⁻⁹ | 6·10⁻⁷ | 930 of 4,638 | 1 | 1 |
| LS220 | 5·10⁻⁹ | 2·10⁻⁵ | 360 of 3,899 | 29 | 146 |
| SRO LS220 | 2·10⁻⁷ | 10⁻³ | 396 of 3,896 | 42 | 163 |
| DD2 | 4·10⁻³ | 0.8 | 513 of 4,066 | 128 | 137 |
| DD2 repaired | 4·10⁻³ | 0.8 | 570 of 4,067 | 127 | 135 |
| SFHo | 4·10⁻⁹ | 0.7 | 477 of 3,995 | 31 | 68 |

On the H200, with `study/device_parity.jl` (2²⁰ states per table), the previous
commit and this one ran in the same job on the same GPU. Every Float64 and
mixed gate passed for both; at Float32 both failed the same gates, the ones
the [precision page](docs/src/precision.md#float32-on-a-gpu) reports. Warm
throughput was unchanged, since there the warm start is the exact answer and no
Newton step is taken. States the host solved cold and the device did not fell
on every real table (LS220 84 to 60, SFHo 207 to 135, DD2 20 to 3). Cold
throughput, millions of states per second:

| cold `con2prim`, H200 | Float64 | Float32 | mixed |
| --- | --- | --- | --- |
| synthetic | 16.3→67.3 | 18.1→56.0 | |
| LS220 | 1.1→1.5 | 4.1→5.8 | 0.9→1.2 |
| SRO LS220 | 1.6→2.5 | 5.4→9.9 | 1.4→2.3 |
| DD2 | 1.3→1.9 | 4.9→9.1 | |
| SFHo | 1.0→1.4 | 7.8→15.3 | |

DD2 repaired matched DD2. In the same job the host, on eight threads, got 2.6
times faster on LS220. On the device the synthetic table gained 4.1 times, but
the real tables only 1.4–1.6 at Float64 and 1.4–2.0 at Float32. There about 2%
of cold states take the fallback or fail, about half the warps contain one,
and the warp waits for it: the seed was most of the work per state but not of
the work per warp.

The throughput table it superseded, H200, LS220, millions of states per second:

| | evaluate | con2prim cold | con2prim warm |
| --- | --- | --- | --- |
| Float64 | 318 | 1.1 | 324 |
| mixed | 370 | 0.9 | 335 |
| Float32 | 411 | 4.0 | 841 |

2026-10-24. The search stays inside the density box. Found while asking why a
formulation meant never to fail still did. The test suite and
`study/cold_failures.jl` sample 5% inside each end of the density range, which
leaves out the top of the tables. Over the whole box (`study/device_parity.jl`'s
sampling, 32,768 valid states per table) cold solves failed on 2.0% of LS220's
states and 2.2% of SRO LS220's, and a further 0.6–0.9% on every real table
converged on another exact solution than the one that went in. A scratch
classification put every such state where the EOS along the solver's path is
acausal (a crossing of φ = 0 can fall only where cs²·tanh² w > 1) or unstable
(cs² < 0). There were two sources. One is the tables' own acausal regions (a) and (b) of
[CODE.md](CODE.md#where-the-answer-is-not-unique). The other was the solver's:
at fixed s its bracket [0, w_max] sweeps ρ = D/cosh w up to D = ρW, and above
the box the table is extended only to stay finite. Its x_hi tail is acausal (the
C++'s class E map counts 98–99% of that band acausal), and beyond
x_ext_hi = x_hi + 0.44 it is frozen at the edge, where evaluations reported cs²
from −636 to 2,382.

- On the SLy hybrid, causal on its box but not above it (cs² 1.26 at
  3·10¹⁵), 184 cold solves of 32,768 failed. 178 of them were at the top of the
  box, with every extra root above ρ_max.
- On LS220 causally repaired by `eos_repair` (66,419 nodes capped, its acausal
  corner gone), 84 failed and 142 converged on another state, again with every
  extra root above ρ_max.
- A plain nested bisection restricted to ρ ≤ ρ_max found exactly one solution,
  the input, for all 84 and for 178 of the hybrid's 184.
- The hybrid's other 6 are cold crust states at W ≈ 100–175 whose thermal
  energy is 2·10⁻¹³–6·10⁻¹² of τ, below the tolerance, so the conservatives do
  not determine s. ρ comes back right to 10⁻¹², reported as `failed_no_bracket`.
  That class is unchanged.

The "DD2 repaired" table measured here did not have its acausal regions
repaired. Its log shows the causal-cap stage capping no node and leaving all
131,736 violations, since none is anchored at ρ_max, and every count is the
same for both versions of DD2.

The adapter design's rule for out-of-bounds handling, "extension plus flag,
never a hard clamp", had been read as a reason to let the search leave the box.
It compares an extension with an EOS evaluated at a clamped point, whose zero
derivatives stall Newton; a search kept inside the box had not been measured.
A copy of `con2prim` generated from the source, bitwise the package's when
unconfined, measured it first: the Newton converged on the same states in the
same number of steps inside the box or not, and only the fallback changed.
Decided by Erik: confine the seed, the Newton and the fallback at
w ≥ acosh(D/ρ_max), with a face test in the inner solve and one before the
fallback's scan, and keep the low-density side.

Settled by the implementation and its measurement:

- ρ_max is widened by `face_slack`. Without it, outputs that `con2prim_safe`
  had written onto the ρ_max face were no longer fixed points, which the
  projection tests caught on the synthetic table, on an analytic EOS and on
  four of the five real tables: `acosh` of their D/ρ_max rounded above their
  own rapidity, so the confined search excluded their exact root.
- The face test before the fallback is the scan's first candidate, its least
  entropy, rather than an evaluation of its own: on the face φ rises with s,
  so if that root lies beyond the face, every other does. With the inner
  solve's test it also reuses its loop's evaluation, for the call-site
  discipline of [CODE.md](CODE.md#discipline-in-core). The Metal kernels grew
  by 49 lines of IR, to 8,246 (`con2prim`) and 15,798 (`con2prim_safe`), and
  the GPU testset passes.
- Without the inner solve's face test, a confined inner solve whose root lay
  beyond the face bisected its whole budget away onto it, and LS220's states
  with D > ρ_max took 65% longer instead of less.
- Rejected: ending the Newton after two iterates held on the face with the
  root beyond it. It cut the cost of states above the box by 11–70%, but sent
  764 of 4,127 valid ideal-gas states from a guess 10% off to the fallback,
  3.4 times as costly: valid states are often held on the face for a few
  iterates and then recover.
- Rejected, on the generated copy: confining only the fallback, the Newton free
  but not accepted above the box. It fixed the same failures, with up to 23
  fewer Newton successes per table (9 more on the synthetic table). Also
  rejected: confining at ρ_min as well. It changed no cold result and moved
  warm ones both ways, and that extension is causal by design and is where
  converged states legitimately live.

Measured against the previous commit (scratch `measure.jl`, the same states
solved by both, single-threaded timing); [CODE.md](CODE.md#deliberate-deviations-from-the-c)
has the cold counts over the whole box:

- Every state with D ≤ ρ_max came back bit for bit, in every sampling and start.
- From a guess 1e-3 off, the hybrid's 19 failures among states with
  D > ρ_max went to 0. On LS220, SRO LS220 and SFHo, 11, 13 and 3 states that
  had converged on another state failed instead.
- From a guess 10% off, the hybrid's 251 failures and 6 other states went to 0,
  and the repaired LS220's went from 17 and 44 to 1 and 21.
- With the 5% margins of the test sampling, cold failures went from 4 to 6 on
  LS220, from 1 to 0 on the repaired LS220, and stayed put elsewhere.
- Invalid states with ρ up to ten times ρ_max (8,192 per EOS, cold): on the
  ideal gas and the synthetic table, where the old Newton found the root above
  the box, 2.4 and 4.1 µs went to 18 µs, with every state now unsolved. On the
  real tables, 164–228 µs went to 18–36 µs, the old search having mostly ended
  in the fallback anyway. The hybrid went from 10 to 12 µs.
- Warm and cold solves of states with D ≤ ρ_max, timed alternately against the
  previous commit, differed by no more than the noise, ±5%.

`study/cold_failures.jl` rerun: cold failures 5, 9, 10 and 7 (LS220, SRO LS220,
DD2 either version, SFHo); `con2prim_safe` projects 23 of the 31 onto a face
and returns 8 as `NOT_CONVERGED`. The new LS220 failure is a state the previous
commit converged to ρ = 3.0·10¹⁶, three times the box's top. The superseded figures from 2026-10-23 were 4, 9, 10
and 7, with 22 of 30 projected and 8 `NOT_CONVERGED`; before the con2prim fixes
of that day there had been 41, of which 1.x's diagnosis found 29 outside the
entropy window and sent 12 to the atmosphere.

`study/cold_resolve.jl` rerun: the states re-flagged cold stayed within 3 of
their previous counts, and SRO LS220's largest excess fell from 4·10⁻³ to
4·10⁻⁶. The rows of [CODE.md](CODE.md#the-contract)'s table it superseded:

| table | excess p99 | max | > 4·tol (old slack) | > 10⁻⁷ | re-flagged cold |
| --- | --- | --- | --- | --- | --- |
| synthetic | 5·10⁻⁹ | 6·10⁻⁷ | 770 of 4,638 | 1 | 1 |
| LS220 | 2·10⁻⁹ | 2·10⁻⁵ | 271 of 3,899 | 23 | 140 |
| SRO LS220 | 3·10⁻⁸ | 4·10⁻³ | 313 of 3,899 | 30 | 148 |
| DD2 | 4·10⁻³ | 0.8 | 479 of 4,069 | 127 | 134 |
| DD2 repaired | 4·10⁻³ | 0.8 | 492 of 4,068 | 127 | 135 |
| SFHo | 4·10⁻⁹ | 0.7 | 402 of 3,998 | 30 | 63 |

Fixed with it: `study/cold_failures.jl`'s bisection wrote
`cond ? (a, ra = m, r) : (b, rb = m, r)`, which in parentheses builds a
NamedTuple and assigns nothing. The rescue rejected on 2026-10-23
([State policy](#state-policy)) had therefore only evaluated the window's
midpoint. With the bracket updated it still recovers none of the cold failures,
before this change and after it, so that decision stands.

The C++ brackets [0, w_max] as well, and presumably fails the same states; that
was not measured.

2026-10-25. Measured the confined search on an H200 (Symmetry, cn113). Old and
new ran in one job, with `study/device_parity.jl` at Float64 and at Float32 on
every table, and the full test suite with `ENTROPYEOS_TEST_GPU=cuda` and the real
tables (34,830 passed). The results are in
[CODE.md](CODE.md#deliberate-deviations-from-the-c).

- **Cold failures at Float32.** They rose ten times more than at Float64 (0.75%
  and 0.91% of LS220's and SRO LS220's states, against 0.08% and 0.02%). The
  reason is that the old search converged far more often at Float32: on LS220,
  1,037,772 of 2²⁰ states against 1,025,528 at Float64.
- **What the lost states were.** A host run at 2¹⁸ states classified them by
  comparing the result with the input, within 1% in ρ and in s. Of 2,114 on
  LS220, 2,113 had converged on another state, and 2,031 of those had an acausal
  input. 1,700 of them sat at s < 10⁻³, the foot of the entropy extension, at a
  median 27% of the input's ρ. On SRO LS220 it was 2,610 of 2,639.
- **A metric that failed.** An error scaled by the conditioning
  cosh² w/(1 − cs²·tanh² w) had first passed nearly all of these lost states as
  correct. It does not hold where the input is acausal, because that
  denominator turns negative there.

Warm `con2prim` at Float64 got 2–4% slower on the device: the synthetic table
three times, DD2 twice and a Float32-stored LS220 once, each run alternating old
and new. The host measurement had not resolved this. The same day's
investigation is under [Two stages](#two-stages): the `exp10` was not the
cause, and what could be recovered was recovered in the first stage only.

Superseded by it: the two-kernel gain on the real tables was 1.5–2.7, from
SFHo's 1.4 → 2.1 to DD2's 1.8 → 4.8 million states per second; it is now
1.8–2.7.

## Testing

2026-10-24. Added the C++ cross-check through EntropyEOS_jll. Until then every
comparison with the C++ had been run by hand against a local build. The
JLL ships only executables and a static C++ library with no C interface, so
reports, not function values, are compared. The first run found that the JLL's
macOS build contracts `fd3`'s stencil into FMAs. That flipped `delta_T` on 290
of the 330 plateau nodes between "skipped" and ~1e17 (see
[CODE.md](CODE.md#validation-against-the-c)). EntropyEOS_jll 1.1.0+1 is built
with `-ffp-contract=off`, which leaves 2 such nodes on macOS; those come from a
one-ulp difference between macOS's `__exp10` and Julia's `exp10`, so the test
keeps the exemption.

2026-10-26. The cross-check reads the C++'s `--version`. From 1.2.0
`eos_test --level con2prim` repairs in memory and drops the hottest T row, so
Julia builds with `trim_t_hi_rows = 1` to solve the same states.
`ENTROPYEOS_EOS_TEST` points the test at another binary. Run against a local
build of 1743044, everything passed except the dirty table, whose 1.2.0 preset
adds a `SeamDefect` that the Julia generator does not have. That comparison is
marked broken.

2026-10-26, later. With EntropyEOS_jll 1.2.0+0 out, ported `SeamDefect` and the
dirty preset's seam (`SeamDefect(14, 1.35, 0.04)`, 1-based) and removed the
broken marker. The dirty tables agree to 2e-16. An `eos_test` older than 1.2.0
is matched by removing the seam from Julia's preset; the cross-check passes
against the cached 1.1.0 binary too.

### Static analysis

2026-09-23. `Aqua` is how the unused `PrecompileTools` entry was eventually
found.

## Precision and devices

2026-09-24. Products of two conserved quantities overflow Float32;
`seed_z_solve` did this until the study found it.

2026-10-25. The H200 figures on the "Precision and GPUs" page were remeasured
after the search was confined to the density box
([Deliberate deviations from the C++](#deliberate-deviations-from-the-c)). The
figures they replace:

- LS220 at Float64, host / device: cold converged 1025421 / 1025417; cold round
  trip ρ 3.37e-9 / 3.39e-9 at p99 and 33.2 / 33.5 at maximum; evaluate `p`
  maximum 1.56e-12; cold-result agreement 99.85%.
- Float32 cold ρ p99, host / device: LS220 0.24 / 0.28, SRO LS220 0.16 / 0.24,
  DD2 0.16 / 0.15, SFHo 0.10 / 0.10.
- On DD2 at Float32, the device had 6% more cold failures than the host.
- Throughput, measured 2026-10-23, in million states per second on LS220:

  | | evaluate | cold | warm |
  | --- | --- | --- | --- |
  | Float64 | 311 | 1.5 | 315 |
  | Float64, two kernels | | 4.0 | |
  | mixed | 357 | 1.2 | 336 |
  | Float32 | 474 | 5.8 | 859 |
  | Float32, two kernels | | 13.1 | |

- Synthetic table cold: 67.
- Two-kernel Float64 cold on the other real tables: 2.1–5.1.
- Pure Float32 against Float64: up to 3.9 times faster, on cold con2prim.

Several of these had drifted before the change. The previous commit, measured
in the same job, gave 5.79e-10 for the LS220 cold ρ p99, 99.989% agreement,
0.54 / 0.52 for LS220's Float32 cold ρ p99, and 10.5% for DD2's Float32
excess failures.

### Not-quite-transcendental functions

2026-10-25. Rejected replacing `log10`, `exp10` and the hyperbolic functions
with not-quite-transcendental (NQT) ones: the binary exponent plus a
polynomial in the mantissa, exactly invertible and cheaper than the real
functions (Miller, Dolence & Holladay,
[arXiv:2206.08957](https://arxiv.org/abs/2206.08957); Hammond et al.,
[arXiv:2501.05410](https://arxiv.org/abs/2501.05410)). The C++ design notes had
deferred them until an accuracy study (`eos-device-interface.md` §8); this is
that study. Measured by `study/nqt.jl` on one Apple M3 thread.

There is little to gain. On the five real tables, 100,000 states each, sampled
as `test/test_real_tables.jl` does, a cold `evaluate` takes 1.8–4.2 µs and
9.4–9.8 temperature-solve iterations; its one `log10` and three `exp10` take
13.4 ns, 0.3–0.7%. The temperature solve iterates in log T itself, so its loop
calls none. A `con2prim` warm-started from the truth perturbed by 1e-3 takes
3.0–3.7 µs and 4.2–4.3 residual evaluations; their `cosh`, `sinh`, `tanh` and
`sinh(w/2)` take 70–78 ns, 2.0–2.4%. The times moved by up to a factor of two
between runs, and with them the shares: an earlier run gave 1.0% and 2.6% on
LS220. `acosh` runs only on the projection's slow path and for the default
`w_cap`. The GPU was not measured; Hammond et al. found no gain on GPUs for EOS
lookups in singularity-eos, where a linear interpolation costs much less than a
tricubic spline with derivatives.

And accuracy is lost where the design needs it. The published NQT logarithms
are C⁰ (NQTo1; the 2022 paper's "C¹" is corrected in the 2025 one) and C¹
(NQTo2). That suffices for the linear interpolation they were made for, and
Hammond et al. say higher-order interpolation will not benefit. Here p and cs²
are derivatives of a C² spline ([CODE.md](CODE.md#type-design)). In a 1-D test,
`fit_bspline_1d` fitted log₂ U, for U = ρ^0.5 + 0.1 ρ^1.2 on ρ ∈ [1, 2²⁰], at
nodes uniform in log₂ ρ or in an NQT version of it. Maximum relative errors on
[2, 2¹⁹] at spacing h = 1/8, which puts the powers of two on grid points, and
for cs² also at h = 0.13, which does not; "wiggle" is the largest deviation of
the NQT log₂ from log₂:

| axis | wiggle | U | ρ²U_ρ | h·cs² | h·cs², h = 0.13 |
| --- | --- | --- | --- | --- | --- |
| log₂ | 0 | 4.4e-9 | 1.9e-7 | 1.1e-5 | 1.2e-5 |
| NQTo1: C⁰, degree 1 | 8.6e-2 | 6.7e-3 | 0.48 | 16 | 13 |
| NQTo2: C¹, degree 2 | 9.7e-3 | 3.8e-4 | 0.020 | 0.45 | 0.51 |
| C², degree 3 | 4.4e-3 | 6.1e-6 | 2.0e-4 | 7.4e-3 | 3.4e-2 |
| C³, degree 4 | 1.8e-3 | 4.0e-6 | 1.2e-4 | 3.0e-3 | 3.2e-3 |
| C⁴, degree 5 | 4.8e-4 | 5.9e-7 | 1.7e-5 | 8.8e-4 | 1.1e-3 |
| C², degree 7, fitted to log₂ | 7.9e-7 | 4.3e-7 | 1.3e-5 | 3.5e-4 | 5.5e-4 |
| C², degree 9, fitted to log₂ | 1.7e-8 | 2.0e-8 | 8.4e-7 | 3.4e-5 | 3.0e-5 |
| C², degree 11, fitted to log₂ | 4.0e-10 | 4.6e-9 | 1.9e-7 | 1.2e-5 | 1.2e-5 |

Halving h from 1/4 to 1/32, log₂ converges at orders 4, 3 and 2 in U, p and
cs². NQTo1 converges at order 1 in U and not at all in p, and its cs² error
grows as 1/h; NQTo2 converges at orders 2 and 1, and its cs² error stays at
0.43–0.48. The errors sit at the powers of two, so a table would have them once
per factor of two in ρ, in T, and in ε, which is fitted as log₁₀ ε. The tables
would also have to be resampled: their nodes are uniform in log₁₀ ρ and
log₁₀ T, and a cheap cell lookup needs nodes uniform in the NQT coordinate.

Smoother variants exist, but smoothness is not the main cost. Write the NQT
log₂ as p + f(t), with t = m − 1 for the mantissa m ∈ [½, 1). Then
lg(2n) = lg(n) + 1 makes it C^k if and only if f(0) = 0, f(−½) = −1 and
f⁽ʲ⁾(−½) = 2ʲ f⁽ʲ⁾(0) for j = 1…k, which fix a polynomial of degree k + 1:
k = 1 is NQTo2, k = 2 is f(t) = (10t − 4t² + 8t³)/7, and up to k = 4 all are
monotone. No polynomial is C^∞, since the condition on its highest derivative
forces the leading coefficient to zero. The C² cubic recovers orders 4, 3 and 2
when the powers of two are grid points; when they are not, cs² converges at
first order only. What remains is the wiggle. On a log axis a power law is a
straight line, which a cubic spline reproduces exactly; on an NQT axis it picks
up the periodic deviation from log₂, one period per factor of two, and the
spline has to resolve that. So the C², C³ and C⁴ polynomials stay 80–700 times
above log₂ in cs². Further degrees spent on a least-squares fit to log₂ shrink
the wiggle, and matching log₂ took degree 11: a polynomial approximation of the
logarithm of the kind libm already uses, with no speed left to gain. From
degree 3 on, the inverse, which the three `exp10` call sites would need, is an
iterative root solve as well.

The hyperbolic functions fare worse, because there the identities are physics.
NQT `cosh` and `sinh` built from an NQT exponential E give
cosh² − sinh² = E(w)E(−w), which is 1 only where w/ln 2 is an integer; it
reaches 1.125 for NQTo1 and 1.0066 for NQTo2, at w = ½ ln 2. W and v would then
disagree with W = 1/√(1 − v²) by up to 5.7% and 0.33%, and the energy
residual's cancellation-free cosh w − 1 = 2 sinh²(w/2) would fail by up to 100%
and 20%.
An NQT `acosh` paired with the true `cosh` puts the projection's rapidity bounds
off ρ_min and ρ_max by up to 5.8% and 0.67%. The one consistent use is a new
Newton variable, u = Wv = S(ŵ) for some cheap monotone S, with W = √(1 + u²)
and v = u/W, which would change the solver for a gain of at most 3%.

Also not pursued: if those 3% ever matter, all four functions follow exactly
from one e = expm1(w), as sinh w = e(e + 2)/(2(e + 1)),
cosh w − 1 = e²/(2(e + 1)) and tanh w = sinh w / cosh w. That would be a
deviation from the C++, and `prim2con` would have to follow it.

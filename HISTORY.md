# History

How the package got where it is. The current design is in [CODE.md](CODE.md);
the sections here follow its headings. Dates are those of the commits that
recorded each item in the documentation.

- [Type design](#type-design)
  - [The EOS interface](#the-eos-interface)
- [State policy](#state-policy)
- [Discipline in `core/`](#discipline-in-core)
  - [The one stack buffer](#the-one-stack-buffer)
- [Deliberate deviations from the C++](#deliberate-deviations-from-the-c)
- [Testing](#testing)
  - [Static analysis](#static-analysis)
- [Precision and devices](#precision-and-devices)

## Type design

### The EOS interface

2026-09-24. Introducing `AbstractEOS` changed no arithmetic: a hash over 3,000
states of every entry point's output was bitwise unchanged.

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

## Discipline in `core/`

### The one stack buffer

2026-09-23. The `MVector{33,T}` buffer in `bracket_scan` was the design's
biggest open risk; it is resolved by the kernel compiling and running on Metal.

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

## Testing

### Static analysis

2026-09-23. `Aqua` is how the unused `PrecompileTools` entry was eventually
found.

## Precision and devices

2026-09-24. Products of two conserved quantities overflow Float32;
`seed_z_solve` did this until the study found it.

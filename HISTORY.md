# History

How the package got where it is. The current design is in [CODE.md](CODE.md);
the sections here follow its headings. Dates are those of the commits that
recorded each item in the documentation.

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

## Testing

2026-10-24. Added the C++ cross-check through EntropyEOS_jll. Until then every
comparison with the C++ had been run by hand against a local build. The
JLL ships only executables and a static C++ library with no C interface, so
reports, not function values, are compared. The first run found that the JLL's
macOS build contracts `fd3`'s stencil into FMAs. That flips `delta_T` on
plateau nodes between "skipped" and ~1e17 (see
[CODE.md](CODE.md#validation-against-the-c)); it is a build setting, not a
translation difference.

### Static analysis

2026-09-23. `Aqua` is how the unused `PrecompileTools` entry was eventually
found.

## Precision and devices

2026-09-24. Products of two conserved quantities overflow Float32;
`seed_z_solve` did this until the study found it.

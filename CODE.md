# Code design

Companion to [README.md](README.md), which covers what the package is for and
how to call it. This document covers how it is built and why. The physics is
specified by the C++ repository's design notes — `eos-adapter-F-to-U.md`,
`con2prim-entropy-rapidity.md`, `eos-causal-tail.md` — which this port follows
rather than restates; [README § What it does](README.md#what-it-does) summarizes
the method. How the design got here is in [HISTORY.md](HISTORY.md).

- [Scope](#scope)
- [Environment](#environment)
- [Layout](#layout)
- [Data model and units](#data-model-and-units)
- [Type design](#type-design)
- [State policy](#state-policy)
  - [Level 1: projection](#level-1-projection)
  - [Level 2: physics](#level-2-physics)
  - [Flags](#flags)
- [Discipline in `core/`](#discipline-in-core)
- [Deliberate deviations from the C++](#deliberate-deviations-from-the-c)
- [Threading](#threading)
- [Testing](#testing)
  - [Static analysis](#static-analysis)
  - [Allocation gates](#allocation-gates)
- [Latency](#latency)
- [Precision and devices](#precision-and-devices)
- [Validation against the C++](#validation-against-the-c)
- [Open items](#open-items)

## Scope

Only the run-time path a hydro code calls: load a table, check it, evaluate it,
and invert it.

Table *repair* is deliberately not included — it is an offline activity
performed once before a simulation campaign, and the run-time path contains no
repair logic. Use the C++ `eos_repair` tool for that, and `check_table` here to
detect a table that has not been repaired.

The C++ tools (`eos_repair`, `eos_test`, `eos_crop`) are not translated
either; `bin/` runs the builds that EntropyEOS_jll ships. `bin/` is its own
Julia environment, so the package does not depend on the JLL, nor on the MPI
build of HDF5 that the JLL pulls in. `bin/run.jl` passes the tool's exit code
through, because those codes are verdicts (`eos_repair` exits 1 when it
changed the table). The three shell scripts beside it only spare typing
`julia --project=bin bin/run.jl TOOL`.

## Environment

- Julia 1.10 or newer. Developed on 1.13; CI covers 1.10 and 1.11 on Linux,
  macOS and Windows.
- Dependencies: `HDF5` (table I/O), `Adapt` (the entire GPU contract),
  `StaticArrays` (one stack buffer), `EnumX` (namespaced outcome enums).
  Deliberately **not** dependencies: `CUDA`, `Metal`, `KernelAbstractions`.
  A caller brings its own backend; see "GPU" below.

  **Inconsistent:** this list does not name `PrecompileTools`, while
  [Layout](#layout) and [Latency](#latency) describe `src/precompile.jl` as a
  `PrecompileTools` workload.
- Tests additionally use `ForwardDiff` (the derivative oracle) and `StableRNGs`
  (Julia's default streams are not stable across minor versions, and a test
  that silently samples different points after an upgrade is not a test).

## Layout

The `core/` versus `host/` split mirrors the C++. There it is a hard
requirement — `core/` must compile under `nvcc`. Here it is documentation of
the GPU boundary, and it keeps every file cross-referenceable against its C++
original. Both halves live in one flat module; submodules would complicate
`Adapt` rules, precompilation and inlining for no benefit.

```
src/core/            kernel-side: allocation-free, exception-free, generic in T
  defs.jl            flag bits, outcome enums, domain-safe math, per-type tolerances
  eos_interface.jl   AbstractEOS, its contract, and the EOSPoint/SRange results
  bspline_eval.jl    uniform cubic B-spline evaluation, value and derivatives
  adapter_eval.jl    the designed domain extensions, then evaluate() and srange()
  adapt.jl           Adapt rules and explicit scalar-type narrowing
  analytic_eos.jl    IdealGasEOS and HybridEOS: closed-form potentials
  prim2con.jl        primitives to conserved, closed form
  con2prim.jl        conserved to primitives: Newton, cold seed, bracket scan, fallback
  faces.jl           the projection's face solves
  state_policy.jl    the projection that never fails, and the physics policy above it
src/host/            owns memory, may throw, never needed on a device
  units.jl           physical constants and the unit conventions
  table.jl           RawTable: axes, named 3-D fields, scalar attributes
  bspline_fit.jl     banded LU and the not-a-knot fit
  adapter_build.jl   validate, fit, derive κ, audit monotonicity
  check.jl           table diagnostics
  synthetic.jl       an analytic ideal gas, and deliberate defect injectors
  analytic_eos.jl    the analytic EOSs' validating constructors and GPP constants
  io_stellarcollapse.jl   the stellarcollapse HDF5 reader (read side only)
src/precompile.jl    PrecompileTools workload over the whole public Float64 path
bin/                 the C++ tools from EntropyEOS_jll, in their own environment
```

Include order in `EntropyEOS.jl` is dependency order, and `test/runtests.jl`
follows the same order so that the first failing testset is the lowest broken
layer rather than the alphabetically first one.

## Data model and units

`RawTable` is the file's contents verbatim: every dataset is carried, not just
the interpreted ones, and nothing is converted on the way in. A field is an
`Array{T,3}` of size `(nρ, nT, nYₑ)`, whose column-major layout is byte-identical
to the C++'s flat `irho + nrho*(jT + ntemp*kYe)`. The stellarcollapse files
store 3-D fields C-order with dims `(nYₑ, nT, nρ)`, which HDF5.jl hands back as
exactly that Julia array — **no permutation is applied, and adding one is the
bug this note exists to prevent**.

Two format traps: `energy_shift`, `have_rel_cs2` and `points*` are one-element
*datasets*, not HDF5 attributes; and the baryon-mass convention is per table
family and is **not in the file** (amu for LS220, DD2, SFHo; the neutron mass
for the SRO tabulation).

Table-side units are cgs plus MeV plus k_B per baryon. The run-time path's
units, and why κ is part of the EOS identity, are in
[README § Units](README.md#units); the view carries the pre-computed scalars
of the conversion done when the adapter is built.

κ re-zeroes the energy so `U ≥ 0` — which is what lets `prim2con` build τ
without cancellation — and the rescaling is exact rather than approximate:
`ρ*(1+U) = ρ(1+ε)` holds bitwise, and is asserted.

The fitted box ends at the table's hottest T row, and the designed tail serves
states above it. The C++ 1.2.0 repair drops that row from the table, because the
published tables' top row is the one it cannot make causal, and records how many
rows it dropped in `eeos_t_hi_rows_dropped` (`T_HI_ROWS_DROPPED_ATTR`), which
the reader keeps as an attribute. `BuildOptions.trim_t_hi_rows` drops rows at
build time instead, for a table that has not been through that repair; it
defaults to 0, as in the C++, so that a repaired table is fitted whole. A top
row left in place unrepaired spoils the fit below it
([HISTORY](HISTORY.md#data-model-and-units)).

## Type design

**Storage type versus working type.** A view is parameterized on the type of
its coefficient arrays; the working type is whatever promotes from the call's
arguments. So `evaluate(v::EOSTableView{Float64}, ρ::Dual, ...)` returns
`EOSPoint{Dual}` against a plain `Float64` table. That is what lets ForwardDiff
differentiate through the whole adapter without a dual-valued table, and it is
why kernel-side result types are constrained to `Real` rather than
`AbstractFloat` (`ForwardDiff.Dual` is `Real` but not `AbstractFloat`). Storage
types keep `AbstractFloat`, since that is an array element type.

**The EOS interface.** `prim2con`, `con2prim` and the state policy take any
`AbstractEOS`. They reach it only through `evaluate`, `srange`,
`srange_extended`, `logρ_bounds` and `yₑ_bounds`, and every method is typed on
the abstract type. Two more methods are optional, because their defaults are
built from those: `evaluate_windowed`, which the table overrides to return the
extended window its temperature solve has already sampled, and
`entropy_from_energy`, which it overrides to invert its energy field directly
([Deliberate deviations](#deliberate-deviations-from-the-c)). The contract is
stated once, in `core/eos_interface.jl`. Introducing it changed no arithmetic
([HISTORY](HISTORY.md#the-eos-interface)).
`EntropyEOS.κ` is part of the interface but is not exported, since `κ` is too common a name to take from a caller.

**The analytic EOSs** have no C++ counterpart; what they are is in
[README § Analytic EOSs](README.md#analytic-eoss). The generalized form of
`HybridEOS`'s cold part was chosen over the classic one
(Read et al. 2009) because it makes `dp/dρ` continuous as well as `p` and `ε`.
Its `Kᵢ` are derived from continuity of `dp/dρ`, so classic Read-et-al.
parameter sets give a different EOS. Use the paper's own fits, and note that its
Table II misprints the second crust break (`1.826e6` should be `1.826e8`).
The classic form's jumps in `cs²` would put jumps into con2prim's Jacobian, which
the table path is C² precisely to avoid. Both EOSs are closed-form, so
`evaluate` neither clamps nor iterates; the box only sets flags, and ends
con2prim's search at its top ([Deliberate deviations](#deliberate-deviations-from-the-c)),
which matters because the causality check covers only the box. The terms are
written as one `exp` of a sum of logs, and each polytropic piece is scaled to a
reference density, so that cgs parameters (`ρ^Γ ~ 1e45`) do not overflow
`Float32` intermediates. Validation, including a causality check over the box,
happens once in the host constructors, so the kernels never throw.

**The isbits boundary for tables is one function**, `EOSTableView(::EOSTable)`. Everything
in `core/` is kernel-side; everything in `host/` is not.
`EOSTableView{T,Array{T,3}}` is deliberately *not* isbits — it holds a heap
array — and becomes isbits once `Adapt` has moved it to a device array type.

**GPU.** The coefficient arrays are ordinary array fields rather than raw
pointers, so the view keeps them alive by itself and the whole class of
dangling-pointer bug that the C++ device mirrors exist to prevent simply does
not arise. One `Adapt.adapt_structure` rule covers both hops: `adapt(CuArray, v)`
gives a device-resident host-callable view, and passing *that* into a kernel
triggers the backend's own adaptor. This is exactly the protocol
KernelAbstractions uses — verified by reading its `argconvert`, and then by
running `evaluate` and `con2prim` as KA kernels on a Metal GPU. Narrowing the
scalar type is a separate explicit step (`narrow`), because a device hop must
never change precision silently.

## State policy

The never-fails path has two levels. `con2prim_safe` projects a state onto the
valid range, which is mechanics, and `apply_policy` decides what a state should
become — the atmosphere, say — which is physics and depends on the evolution
scheme:

```julia
out = con2prim_safe(eos, in, opts, proj)       # level 1: projection
out = apply_policy(eos, in, out, proj, pol)    # level 2: physics, replaceable
```

They were one layer until 1.x, and mixing them cost: the flags reported facts
for some repairs and decisions for others, an energy above the hottest state
was treated as a collapse even in a hot near-vacuum cell (which blew up
TreeGRRMHD's magnetized atmosphere tubes, request EOS-9), and an atmosphere
reset discarded the clamps that preceded it
([HISTORY](HISTORY.md#state-policy)). This design is ahead of the C++, which
follows it, as it did for 1.1.0; it is not one of the
[deliberate deviations](#deliberate-deviations-from-the-c).

### Level 1: projection

`con2prim_safe` returns the valid state closest to the input and flags each
face of the valid range that was active. It never replaces a state by the
atmosphere and has no notion of collapse. `ProjectionOptions` describes the
range:

- ρ ∈ [ρ_min, ρ_max],
- Yₑ ∈ [y_lo, y_hi], the table's range,
- s ∈ [s_min(ρ, Yₑ), s_max(ρ, Yₑ)], the physical window,
- w ≤ w_cap.

w ≥ 0 is not a face, since w is a magnitude and the direction is carried
separately.

Each face is tested with a slack, `face_slack(T)`: 10⁻⁷ at Float64 and
4·`con2prim_tol` (2.4·10⁻⁴) at Float32, relative to the window's width for s
and to the value for ρ and w, and one ulp for Yₑ. Without it a state written
onto a face would not be a fixed point: the solver returns it only to its
conditioning, and even Yₑ·D/D can come back one ulp past y_hi, so the face
would fire again. States are written exactly onto the face, so the slack is a
one-sided band: it decides what counts as valid, not where a projection
lands. The Float64 value is sized for a re-solve without a guess (see
[The contract](#the-contract)).

`inside_window(eos, ρ, s, yₑ)` is the Yₑ and s part of that test, exported for
an application that makes states itself, as by reconstruction, and has to
accept every state `con2prim_safe` returns: tested against the exact window, a
state within the slack of an s face fails. It does not test ρ, since a
reconstructed state may lie below ρ_min. `inside_faces` and `project_prim_state`
share its pieces, so the three cannot disagree.

"Closest" has no metric: D, S and τ have different units, and a priority order
does not work either (keeping D at the density ceiling would mean raising w,
creating momentum). Instead each face fixes one primitive and frees one
conservative, and moves it one way. Change as little as possible, and prefer
removing over adding:

| bit | flag | face | what gives way |
| --- | --- | --- | --- |
| 11 | `ρ_FLOORED` | ρ_min | S lowered to w = acosh(D′/ρ_min), and D raised to D′ = max(D, ρ_min) |
| 12 | `ρ_CEILED` | ρ_max | D lowered |
| 13, 14 | `YE_FLOORED`, `YE_CEILED` | y_lo, y_hi | D_Y set to Yₑ·D with Yₑ clamped |
| 15 | `S_FLOORED` | s_min | τ raised (unavoidable) |
| 16 | `S_CEILED` | s_max | τ lowered |
| 17 | `W_CEILED` | w_cap | S lowered |

Quantities a row does not name are kept, to the solver's tolerance. Yₑ is kept
when D changes, so D_Y moves with D: matter added or removed has the cell's
composition. Where S is lowered, the velocity's direction relative to B is taken
from the input momentum at the new state's enthalpy density, which makes the
returned S exactly parallel to the input: it is scaled, not turned. (`prim2con`
carries that direction as `cos_vB` and recovers sin_vB from it, so where v is
nearly along B, or B² ≫ z, the perpendicular part loses its relative digits.)

The ρ_min row is one rule for both signs of D − ρ_min: a state with D ≥ ρ_min is
slowed down until ρ = ρ_min and gains no mass, and a state with D < ρ_min is put
at rest at ρ_min. The two meet continuously at D = ρ_min, and the second adds
the least mass possible. Capping w at fixed τ heats the cell: the kinetic energy
taken out of S stays in τ. A cold state at W = 1000 capped to W = 100 needs
h ≈ 10, and usually ends on the s_max face as well. That is deliberate: level 1
keeps what it can, and a hot near-vacuum cell is for level 2.
[HISTORY](HISTORY.md#state-policy) has the rejected alternatives.

Two outcomes are not projections:

| bit | flag | when |
| --- | --- | --- |
| 9 | `NONFINITE` | a non-finite input, D ≤ 0 (where Yₑ = D_Y/D is undefined), or B² < 0 |
| 10 | `NOT_CONVERGED` | the solve failed, but the diagnosis finds a valid state inside the range |

Both return a fixed valid state — ρ_min, at rest, s_min, and the input's
Yₑ, clamped, or the middle of the Yₑ range for `NONFINITE` — only so that a
kernel never propagates NaN. A clamped Yₑ keeps its flag, and the fixed state
lies on that face. Level 2 is expected to
replace it, so it takes no parameter. s_min is the least energy; the middle of
the window at ρ_min would be radiation-dominated. No rescue is attempted (see
the measurements below).

#### Which faces are active

Each set of active faces leaves a square system — every face fixes one primitive
(ρ, w, s, Yₑ) and frees one conservative (D, S, τ, D_Y) — so it determines the
state. The ρ_min face is the exception, fixing w = acosh(D/ρ_min) and freeing S
where D ≥ ρ_min, and fixing w = 0 and freeing D and S where D < ρ_min. It is
never active together with w_cap. Several sets can give a valid state; the rule
that picks one is this: **the active set is the one in which every active face
moves its freed conservative the way the table above says.** Where the inner
root is unique, that set is unique. Two faces can each fix a state alone — a
cold, fast state can be floored in s (τ raised, S kept) or capped in w (S
lowered, τ kept) — and the rule picks the one whose sign holds.

`con2prim_safe` decides it in the following steps. Only a state inside every
face is taken from `con2prim`; anything else is decided by the diagnosis, so
that a converged and a failed solve of neighbouring states project alike.

1. **NONFINITE** as above, then the Yₑ faces, which decouple from the rest.
2. **Faces decided by D alone.** D < ρ_min: every rapidity gives ρ < ρ_min, so
   the state is put at rest at ρ_min with τ kept. D > ρ_max·cosh(w_cap): none
   gives ρ ≤ ρ_max, so the ρ_max face is active.
3. **The solve.** `con2prim` runs; if it converged inside every face, the input
   comes back bit for bit (with D_Y = Yₑ·D if Yₑ was clamped).
4. **The diagnosis.** On the rapidities at which keeping D leaves ρ and w in
   range, w ∈ [w_lo, w_hi] with w_lo = acosh(max(D/ρ_max, 1)) and
   w_hi = min(w_cap, acosh(D/ρ_min)), the states keeping D and S form a curve
   that runs from cold and fast to hot and slow, along which the energy residual
   G = f₂ rises. Along an entropy edge s = s_edge(D/cosh w), the curve's
   crossing is a root of φ = tanh w − V. With φ evaluated on both edges at both
   ends of the interval:
   1. φ on the s_min edge > 0 at w_lo: the crossing lies above ρ_max, and every
      state keeping D and S is too cold or too dense. Solve on the ρ_max face.
   2. φ on the s_max edge < 0 at w_hi: every such state is too hot or too fast.
      Solve on the w_hi face (w_cap or ρ_min, whichever bounds the interval).
   3. Too cold at the cold end: the s_min edge's root if it is in the interval
      and f₂ > 0 there (`S_FLOORED`); else the w_hi face, if the state there
      carries less momentum than the input. If neither, go on.
   4. Too hot at the hot end: the s_max edge's root if it is in the interval and
      f₂ < 0 there (`S_CEILED`); else the ρ_max face, if that removes mass. If
      neither, go on.
   5. Neither. If one of the face states of steps 3 and 4 solves the input to the
      tolerance, it is a solution on the face and is returned. Otherwise
      `NOT_CONVERGED`.

   Where G is monotone, steps 3 and 4 cannot both succeed; testing cold first
   only makes the choice deterministic where it is not. The tolerance case in
   step 5 matters where the energy barely depends on s: for a cold state in
   Float32 on `HybridEOS`, G is flat to within the tolerance across the whole
   window, `con2prim` converges at any s, often outside the window, and without
   step 5 such a state went to `NOT_CONVERGED`.

The diagnosis evaluates the s faces only at densities the ρ_max face has
admitted: above ρ_max, where LS220 is extrapolated with cs² ≈ 1.4, f₂ can fall in
s by its whole span. Its edge root find is consistent with the window at the
density actually written; TreeGRRMHD's EOS-9 measured that endpoint solves taken
at the window of the failed solve's density miss the written state's window by
up to 0.27 of its width.

#### The face solves

`core/faces.jl` has the three one-dimensional problems a set of active faces
leaves:

- **Along an entropy edge** (`edge_root`): the w that keeps S, by Illinois on φ
  rather than on f₁ = cosh w·φ. The two have the same sign and roots, and φ is
  what `f1_converged` tests, but f₁ is not monotone wherever V > 1, even at fixed
  s on an ideal gas.
- **At fixed ρ and w** (`face_solve` without `free_w`): the s that keeps τ, for
  the w_cap and ρ_min faces. Endpoint signs at s_min(ρ) and s_max(ρ) give the s
  corners.
- **At fixed ρ, w following S** (`face_solve` with `free_w`): the same, for the
  ρ_max face, which keeps S and τ. At fixed (ρ, s) the enthalpy is fixed, so the rapidity carrying S is pure
  arithmetic (`w_from_momentum`, a concave root in sinh² w), and the solve is a
  nested one-dimensional one rather than the two-dimensional con2prim at fixed ρ
  first anticipated. If it needs w > w_cap, the w_cap face is active too.

None of them can fail to return a state. Each is a bracketed find whose bracket
lies inside the valid range, so every iterate is a valid state on its face. If
one exhausts its budget, it returns its iterate with the smallest residual: valid
and flagged, with only the conservative the face keeps off by more than the
tolerance. That is not reported as `NOT_CONVERGED`, which would send a correct
projection to the atmosphere. A non-finite result (an EOS evaluation returning
NaN) gives the fixed state with `NOT_CONVERGED`.

#### Where the answer is not unique

Termination and uniqueness rest on φ being monotone in w along a face, and on G
being monotone along the curve. Measured on 2026-10-22 by
`study/projection_faces.jl` on the five real tables: 2,000 states per table,
mode (hydrodynamic, MHD) and sampling (s uniform in the window, log T uniform),
with τ scaled to aim at either branch, counting only states inside the other
faces.

- **The root find terminates.** It converges for all but 16 of 80,000
  state–face pairs, all in region (c) below. There the cold seam leaves ~10⁻⁸ of
  noise in φ, above the 10⁻¹² tolerance, and in one DD2 state φ jumps by 0.19
  across its sign change. Illinois takes 6–18 / 17–25 / 49 evaluations
  (p50 / p99 / max) on φ, against 18–22 / 25–36 / 60 on f₁.
- **The diagnosis is reliable where the inner root is unique.** G is
  non-monotone for 0–2.9% of states, by at most 4·10⁻⁴ of its span, and in none
  of 49,305 cold or hot diagnoses does it take the wrong sign anywhere in the
  window.
- **The answer is not always unique.** The share of states where φ along a face
  is non-monotone, and where it has more than one root (ranges over mode and
  sampling):

  | table | s_min: non-monotone | >1 root | s_max: non-monotone | >1 root |
  | --- | --- | --- | --- | --- |
  | LS220 | 4.2–5.4% | 1.9–3.1% | 2.5–2.8% | 0.15–0.55% |
  | SRO LS220 | 4.1–5.1% | 1.6–3.2% | 2.7–3.3% | 0.3–0.65% |
  | DD2 | 0.9–1.0% | 0–0.25% | 1.6–12% | 0.05–0.2% |
  | DD2 repaired | 0.85–1.0% | 0–0.4% | 1.2–12% | 0–0.35% |
  | SFHo | 4.7–5.2% | 0.2–1.3% | 4.0–15% | 0–0.3% |

  The multiple roots lie in three table regions: (a) the top of LS220, SRO
  LS220 and SFHo, ρ ≳ 2·10¹⁵ on the cold edge, where cs² reaches 1.0–1.14;
  (b) DD2's and SFHo's hottest edge at ρ ≈ 1–3·10⁷, where cs² is 5–6 or
  negative; (c) DD2's and SFHo's cold edge at ρ ≈ 10⁹–10¹¹ and Yₑ ≈ 0.27–0.33,
  where s_min(ρ) dips below zero and h along the edge is not monotone.

(a) and (b) are table defects, which the C++ 1.2.0 repair removes (below). At a root of φ, `residuals`' z_w =
z·(1 − cs²)·tanh w gives dφ/dw·cosh² w ≥ 1 − cs²·tanh² w for any B², so at
fixed s every crossing rises, and the root is unique, wherever cs²·v² < 1. The
solve at fixed s has the same multiple roots there (see [Open items](#open-items)),
and they need fast flow through an acausal part of the table. The extension
above the box is acausal as well, and supplied extra roots of its own until
con2prim's search was kept inside the box
([Deliberate deviations](#deliberate-deviations-from-the-c)). Only (c), at most
0.4% of states, comes from the edge's density dependence. Where φ has several
roots, the face find returns whichever root its bracketed find reaches, as
`inner_solve_w` does; no rule picks among them
([HISTORY](HISTORY.md#state-policy)), and the fixed-point invariant below does
not need one.

Where the inner root is not unique, level 1 cannot keep its contract: a cold
start can fail, and the edge solves of the diagnosis can land on another root,
so that a valid state is projected onto a face. Measured on 2026-10-24 by
`study/cold_failures.jl`: of 20,000 valid states per table (the sampling of
`test/test_real_tables.jl`), a cold start failed on 5 (LS220), 9 (SRO LS220),
10 (DD2, either version) and 7 (SFHo), all in region (a) or (b); a warm start
failed on 2, both DD2. Of these 31 cold failures (counting DD2 once),
`con2prim_safe` projects 23 onto a face — all 14 of LS220's and SRO LS220's onto
ρ_max, which is the top of the table they lie at — and returns 8 as
`NOT_CONVERGED`. A bisection in s, bracketed by either window, recovered none,
because G jumps between roots.

That sampling keeps 5% inside each end of the density range, so it leaves out
most of region (a). Over the whole box, cold solves fail on 2.1% of LS220's
and 2.2% of SRO LS220's valid states and on 0.3% of SFHo's, and on every real
table a further 0.6–0.8% converge on another exact solution than the one that
went in, which no flag reports. Regions (a) and (b) are wrong physics, and
repairing the tables removes them. Measured on 2026-10-26: 32,768 valid states
per table over the whole box (s uniform in the inner 96% of the window, w
uniform in [0, 6], half with B² = 10⁻³–10 p), solved cold; failed / converged
on another state:

| table | as published | repaired by 1.1.0 | repaired by 1.2.0 |
| --- | --- | --- | --- |
| LS220 | 679 / 202 | 0 / 17 | 0 / 2 |
| SRO LS220 | 723 / 194 | 0 / 13 | 1 / 7 |
| DD2 | 5 / 212 | 5 / 212 | 0 / 0 |
| SFHo | 85 / 276 | 6 / 179 | 0 / 1 |

The 1.1.0 repair's causal cap fixes (a) on LS220 and SRO LS220 but does not
reach (b), whose violations are not anchored at ρ_max: on DD2 it caps no node.
The 1.2.0 repair adds a seam blend for (b), drops each table's hottest T row,
which it cannot repair, and runs the causal cap on what is left. All 11 states
it leaves are at ρ ≥ 10^15.5, at the top of the box. Warm-started 10⁻³ off, the
1.2.0 tables leave 1 failure (SRO LS220) and 2 other solutions (SFHo). On a
1.2.0 table the u-high ratio lock changes none of these counts; on the others it
moves DD2's and SFHo's by at most 2 states. So a table should be repaired by
1.2.0 before a run; on a table that is not, this remains a limitation
([Open items](#open-items)).

#### The contract

Where the inner root is unique:

- a set face flag means that face was active, and the returned state lies on it,
  to the slack;
- `policy_flags == 0` means `cons` is the input bit for bit;
- every output is a fixed point: fed back with its own primitives as the guess,
  it returns flags 0 and so, by the previous invariant, the same `cons` bit for
  bit. `con2prim` tests convergence at its guess before iterating, so this holds
  whichever root the output is on, even where the root is not unique. Without a
  guess it holds where the re-solve is well conditioned; see below;
- `check_con_state` never reports a face that `con2prim_safe` does not set. Its
  cheap mode is arithmetic, and each of its tests is sufficient: non-finite
  input, D < ρ_min, D > ρ_max·cosh(w_cap), and Yₑ out of range. With
  `check_endpoints` it also runs the edge part of the diagnosis, and reports an
  s face where that alone decides it and no ρ face is active. It takes
  `con2prim_safe`'s branch on the edges themselves, but tests cold or hot on
  the edges moved outward by the slack, and the energy residual against
  `check_f2_tol`. A state within the slack of an edge, which `con2prim_safe`
  returns unflagged, is then not reported either. Testing the edges themselves
  reported 40% of states lying exactly on an edge, which is every state
  `con2prim_safe` writes onto an s face, fed back. At Float32 the bar is 128
  times the solver's tolerance, since the residual is noise at up to 65 times
  it. Measured 2026-10-23 on states lying exactly on an edge: no false report in
  5,000 per case on the synthetic table (both precisions) and the ideal gas
  (Float32), nor in 3,000 per real table, except 6 on LS220. Those are on its
  cold edge at Yₑ ≈ 0.06, where G touches zero at s_min and `con2prim` converges
  on a second root inside the window: the root is not unique there.

A re-solve without a guess returns a state on a face only to its conditioning.
Measured on 2026-10-24 by `study/cold_resolve.jl`: 5,000 states per table,
pushed off a valid state by up to four decades in D, τ and S, each projected
output re-solved cold. "Excess" is how far the re-solved state lies past its
face, relative as for the slack:

| table | excess p99 | max | > 4·tol (old slack) | > 10⁻⁷ | re-flagged cold |
| --- | --- | --- | --- | --- | --- |
| synthetic | 5·10⁻⁹ | 6·10⁻⁷ | 771 of 4,638 | 1 | 1 |
| LS220 | 2·10⁻⁹ | 2·10⁻⁵ | 278 of 3,901 | 25 | 140 |
| SRO LS220 | 3·10⁻⁸ | 4·10⁻⁶ | 314 of 3,900 | 31 | 148 |
| DD2 | 4·10⁻³ | 0.8 | 477 of 4,067 | 127 | 136 |
| DD2 repaired | 4·10⁻³ | 0.8 | 490 of 4,065 | 127 | 138 |
| SFHo | 4·10⁻⁹ | 0.7 | 403 of 3,998 | 30 | 63 |

With a guess at the output none is re-flagged. Most of the excess is the
solver's conditioning, which the slack of 10⁻⁷ covers; the tail is not. Where
the excess is of order 1, as on DD2 and SFHo, the cold start found another
root, in regions (a)–(c). Before the con2prim fixes of
[Deliberate deviations](#deliberate-deviations-from-the-c), the synthetic table
re-flagged 49 rather than 1: 43 of them were states put at rest at ρ_min on the
s_max face, for which a cold `con2prim` failed although its best iterate was the
state. At Float32 the slack cannot help: the re-solved s is off by whole window
widths wherever τ cannot resolve the thermal energy
([Precision and devices](#precision-and-devices)).

`test/testutil.jl`'s `policy_violations` checks all of these, plus each row's
kept and freed conservatives. On 2,000 states per real table, pushed off a valid
state by up to four decades in D, τ and S, none is violated, even in regions (a)
and (b), where the test only promises validity and the fixed point.

`project_prim_state` is the same projection on primitives, where every face is
a plain clamp with the same slack. A non-finite state, ρ ≤ 0 or w < 0 has no
closest valid state and gives `NONFINITE` with the fixed state.

### Level 2: physics

`apply_policy(eos, in, out, proj, pol)` takes the input conservatives, level 1's
output, and the options of both levels (the atmosphere sits at level 1's
ρ_min), and may replace the state by another valid one. It needs no further
flags: it can compute ΔD, ΔS and Δτ from the two conservative states. It keeps
level 1's bits and adds its own, so the face bits describe level 1's output, and
`ATMOSPHERE` (bit 8) says that output was then replaced. The reference
implementation is close to 1.x's behaviour. The atmosphere — at rest, at
ρ_min, with `s_atm` (default s_min) and `ye_atm` (default the input's Yₑ) —
replaces:

- `ρ_FLOORED`, or D < `atm_trigger`·ρ_min;
- `ρ_CEILED` with `collapse_to_atmosphere`;
- `NONFINITE` and `NOT_CONVERGED`.

An application with its own rules (TreeGRRMHD puts any state with added rest
mass at rest) replaces this function rather than configuring it.

### Flags

The policy bits are 8–17, contiguous, and `FLAG_POL_ANY` masks them; bits 6–7
stay reserved for solver flags. Level 2's bit comes first, then the outcomes
that are not projections, then the faces in the order ρ, Yₑ, s, w. Every face
flag names the bound it hit, FLOORED for a lower and CEILED for an upper one.
The C++'s policy layer is a different design, and its bits from 8 up mean other
things.

Bits 18 and 19, `FLAG_CS2_ACAUSAL` (cs² ≥ 1) and `FLAG_CS2_NONPOSITIVE`
(cs² ≤ 0), say what the evaluation returned, like bits 0–5; a NaN sets both, and
`FLAG_CS2_OUT_OF_RANGE` masks them. `EOSPoint.cs²` itself is never clamped:
con2prim's Newton uses it with `U_ρ` and `U_ρρ`, and a clamped value belongs to
no potential. `signal_cs²` is the clamped value for wave speeds and CFL limits.
The table sets them in the box and in every extension band, and the analytic
EOSs set them too. The C++ has them at bits 17 and 18, above its policy group.

## Discipline in `core/`

No heap allocation, no `throw`, no `@fastmath`, no closures, no `String`; array
reads `@inbounds`, small helpers `@inline`. Kernel-side structs stay `isbits`;
a NaN sentinel is not replaced with `Union{Nothing,T}`, because NaN is
load-bearing and flows through arithmetic. Six consequences worth naming:

- **Domain-safe math.** `log10`, `sqrt` and `floor(Int, ·)` throw in Julia where
  C returns NaN or is merely unspecified. `core/defs.jl` provides `safe_log10`,
  `safe_sqrt` and `trunc_floor`. `Base.isnan` and `isfinite` are already exactly
  the self-comparison tricks the C++ hand-rolls, so they are used directly — but
  only because `@fastmath` is banned, which a test enforces by grep.
  `@fastmath` folds the NaN and finiteness probes to the wrong answer, exactly as
  `-ffast-math` does in the C++.
- **Per-type tolerances.** Every numeric constant is a function of the scalar
  type, not a literal, because a bare `Float64` literal silently widens `Float32`
  arithmetic. The `Float64` methods reproduce the C++'s measured values exactly;
  the generic ones scale with `eps(T)` so a narrower type at least runs.
- **One call site per solver.** A GPU compiler inlines every call, and each
  copy of a solver carries whole EOS evaluations. `con2prim_safe` therefore
  decides first which solves to try and then runs them through one call site
  each, and the face solves loop over their endpoints rather than evaluating
  them at separate sites. Written the obvious way, with a solver call in every
  branch, its Metal kernel crashed Apple's shader compiler after two minutes;
  written this way it compiles in 15 s, against 24 s for 1.1.0's
  ([HISTORY](HISTORY.md#state-policy)).
- **The one stack buffer.** `bracket_scan` needs 33 entries live at once and
  cannot recompute them (each costs an inner solve, and the warm-start chain
  fixes their order). It uses `MVector{33,T}`, local and non-escaping so it is
  promoted out of the heap. The cap is a `Val` parameter so a GPU caller can
  shrink per-thread local memory without touching the algorithm
  ([HISTORY](HISTORY.md#the-one-stack-buffer)).
- **Two stages.** A warp runs as long as its slowest state. On a real table
  about 2% of cold states need `con2prim`'s fallback or fail, at many times a
  Newton solve's cost, and about half the warps contain one. So `con2prim` and
  `con2prim_safe` are each a first stage (`*_start`: the cold seed and the
  Newton, and for `_safe` the check that the state lies inside every face) and
  a second (`*_finish`: the fallback, and for `_safe` the diagnosis and the
  face solves). A caller runs the first over every state, compacts the indices
  of those not `is_done`, and runs the second over only those. The fused
  functions are the two stages in sequence and give the same result bit for
  bit. The hand-off is not the iterate (`Residuals` is ~280 bytes) but the
  `(s, w)` and warm start that reproduce it: `residuals` is pure, so the second
  stage rebuilds it exactly, for one extra evaluation on a pending state. On an
  H200 this made cold `con2prim` 1.8–2.7 times faster on the real tables, and
  the device's two-kernel results are bitwise those of its fused kernel
  ([HISTORY](HISTORY.md#two-stages)). The package launches no kernels:
  `study/device_parity.jl` and the GPU testset show the launch and the
  compaction.
- **Register pressure.** On an H200 every `con2prim` kernel uses the full 255
  registers per thread and spills: 3.2 KB of local memory for the first stage
  alone, 8.5 KB fused. The fused kernel runs a warm call 6–9% slower than the
  first stage alone, although its fallback never runs, and its time moves by
  ±2% with rewrites that leave the arithmetic bit for bit unchanged. So a
  change to the hot path is timed on the first stage, where such effects are
  smaller. One such change is the density floor of the confined search:
  `con2prim_start` skips its `acosh` when `above_floor`, a polynomial lower
  bound on cosh, already places the warm start inside the box, and the Newton
  recomputes it each step rather than carry it. That made the warm first stage
  3–5% faster; the fused kernel did not change
  ([HISTORY](HISTORY.md#two-stages)).

## Deliberate deviations from the C++

Each of these is a case where a literal transliteration would have carried a
C++ artifact into Julia for no benefit.

| C++ | here | why |
| --- | --- | --- |
| `Bspline3` owning a vector that `BsplineView3` points into | only `BsplineView3` | Julia arrays own themselves; the split existed purely for the pointer |
| `BsplineView3` stores `nx, nu, ny` | derived from `size(c) .- 2` | removes the standing `n` versus `n+2` hazard |
| `detail::TailAxis` enum | `apply_x_tail` / `apply_u_tail` | the axis is always known at the call site |
| `EntropyEOS` (host class) | `EOSTable` | the C++ name collides with the module |
| `view()` / `view_with()` | `EOSTableView` constructors | avoids shadowing `Base.view` |
| out-parameters (`int &iters_out`, `real S_out[3]`) | extra tuple return values | tuples of isbits values return in registers |
| `detail::` namespace and `aeval_`/`c2p_`/`pol_` prefixes | descriptive unexported names | one module namespace, no collisions with Base |
| `crop_table(in, ia, ib, ja, jb, ka, kb, &fields)` | `crop_table(t, iρ, jT, kYₑ; fields)` with index ranges | the inclusive index pairs are what a range is |
| `flag_cs2_acausal`, `flag_cs2_nonpositive` at bits 17, 18 | bits 18, 19 | bit 17 is `FLAG_POL_W_CEILED` here ([Flags](#flags)) |
| cs² flags only from the table's `eval_at` | the analytic EOSs set them too | they have no C++ counterpart, and a flag should mean the same on every EOS |

Three numerical changes fix defects that the C++ shares. Together they made cold
solves of states on an edge of the entropy window converge, where the C++ fails
17 of 120 such states at rest on the synthetic table
([HISTORY](HISTORY.md#deliberate-deviations-from-the-c)):

- **The T-solve ends with a Newton step.** When `evaluate`'s residual test
  passes, it still takes the step that residual implies, if the step stays in
  the bracket. The test admits an entropy error of 1e-12·|s|, which is a
  relative error in U some 20 times con2prim's tolerance, and a warm start
  already within it used to come back unchanged. The energy residual then
  depended on the warm start and not only on s: Newton stalled just above its
  tolerance, and at a root the residual's sign was noise. The step costs no
  evaluation, because `eval_at` samples at the final u anyway, and it brings
  the warm start's effect on U down to a few units in the last place.
- **`bracket_scan` skips pairs of equal entropy.** Its uniform half's endpoints
  duplicate `s_phys_lo` and `s_phys_hi`. With the root on an edge, the two
  copies' solves could disagree in sign, and that empty pair, whose midpoint
  is the anchor itself, was the one picked.
- **An endpoint within tolerance is a bracket.** After the fallback re-solves
  the scan's interval, an endpoint whose residual meets the tolerance no longer
  fails the sign test.

One change keeps the search inside the density box. The C++ brackets the
rapidity on [0, w_max], so at fixed s its inner solve sweeps ρ = D/cosh w up
to D = ρW: for a fast state near the top of the table, far above the box. The
table is extended there only so that an evaluation stays finite. Its x_hi tail
is acausal (cs² ≈ 1.1 just above ρ_max even after a causal repair) and frozen
beyond x_ext_hi = x_hi + 0.44, where `locate` clamps the density; the hybrid's
last polytrope reaches cs² = 1.26 at 1.5 ρ_max. A crossing of φ can fall only
where cs²·tanh² w > 1, so the inner solve found extra roots there, and the
fallback failed on states whose solution inside the box was unique. Here every
iterate keeps w ≥ `w_floor` = acosh(D/ρ_max), the cold seed's passes, the
Newton, the inner solve and the scan alike, and the EOS is never evaluated
above the box:

- The inner solve, where it would bisect towards the face with the root below
  its iterate, tests the face instead, once, and stops there if the root lies
  beyond it.
- The scan solves its least entropy first. On the face φ rises with s, since z
  does, so if that root lies beyond the face every other does too, and the
  scan stops unbracketed.

Both reuse their loop's own evaluation ([Discipline in `core/`](#discipline-in-core));
the Metal kernels grew by 49 lines of IR. ρ_max is widened by `face_slack`, so
that a state `con2prim_safe` writes onto that face solves again: `acosh` of its
D/ρ_max can round to above its own rapidity. The low-density side stays open:
that extension is causal by design, and converged states legitimately live
there.

Measured on 2026-10-24 against the previous commit, on the same 32,768 valid
states per EOS over the whole density box (`study/device_parity.jl`'s
sampling), solved cold, single-threaded on the host. States with D ≤ ρ_max are
untouched, bit for bit. For the others:

| EOS | states with D > ρ_max | failed | converged on another state | µs per such state |
| --- | --- | --- | --- | --- |
| SLy hybrid | 3,531 | 178 → 0 | 0 → 0 | 3.4 → 3.1 |
| LS220, causally repaired by `eos_repair` | 2,512 | 84 → 0 | 142 → 17 | 58 → 21 |
| LS220 | 2,512 | 656 → 679 | 225 → 202 | 36 → 14 |
| SRO LS220 | 2,512 | 719 → 723 | 200 → 194 | 31 → 17 |
| SFHo | 2,460 | 52 → 79 | 132 → 100 | 25 → 8.5 |
| DD2, either version | 2,334 | 0 → 0 | 0 → 0 | 34–36 → 35–39 |
| ideal gas, synthetic table | 4,127; 3,282 | 0 → 0 | 0 → 0 | 2.6 → 2.5; 4.2 → 4.1 |

Over all states, cold solves got 18–32% cheaper on the four tables with
failures, and stayed within the noise (±5%) elsewhere. The Newton converges on
the same states, to within 2 per table, cold and from a guess 1e-3 off (the
repaired LS220 gains 23 cold), and in the same number of steps (±0.1). From a
guess 10% off it converges on up to 22 more states per table, and on up to 9
fewer (the synthetic table), which the fallback then solves. What changed is
the fallback: where the
box is causal its bracket holds a unique root again. Where it is not, at
the top of LS220, SRO LS220 and SFHo (region (a) of
[Where the answer is not unique](#where-the-answer-is-not-unique)), a state
that converged on a root above the box, which `con2prim_safe` could not use,
now reports failure instead.

A state whose solution lies above the box is not solved: `con2prim` reports
`failed_no_bracket` with its iterate at or below the face, and `con2prim_safe` decides
it by its diagnosis, as it did a converged root outside the range. On the
ideal gas and the synthetic table, whose harmless extension let the Newton find
that root in 2–4 µs, this costs 18 µs per such state; on the real tables, where
the search used to end in the fallback anyway, it is 5–9 times cheaper.

On an H200 (2026-10-25, `study/device_parity.jl`, 2²⁰ states per table) every
Float64 gate still passes, and the Float32 gates that failed before fail on the
same tables. At Float64 in one kernel, cold `con2prim` runs 2.1–3.4 times
faster on LS220, SRO LS220 and SFHo (5.1, 4.8 and 4.1 million states per
second, from 1.5, 2.3 and 1.4), `con2prim_safe` 1.6–2.5 times, and DD2 and the
synthetic table are unchanged. As many states reach the fallback as before,
2.2% on LS220, but the fallback costs less. Device and host now agree on the
cold result for 99.999% of LS220's states, up from 99.989%. Warm `con2prim`
in one kernel costs 2–4% more at Float64 and is within 1% at Float32; its first
stage alone is within 2% of before, either way
([Discipline in `core/`](#discipline-in-core)). Float32
follows the Float64 picture. Cold failures rise by 0.7–0.9% of the states on
LS220 and SRO LS220 and by under 0.1% on DD2 and SFHo. A host run of 2¹⁸ states
per table found that 99.9% of the lost states had converged on another state,
most of them on LS220's acausal top at the foot of the entropy extension. The
number solved to the input itself changed by −5 to +25 per table.

The adapter design's rule for out-of-bounds handling, extension plus flag and
never a hard clamp, is about an EOS evaluated at a clamped point, whose zero
derivatives stall Newton. It does not cover a search that evaluates the EOS
only inside the box ([HISTORY](HISTORY.md#deliberate-deviations-from-the-c)).

One numerical change is for speed. Each pass of the cold seed recovers the
entropy from an internal energy. The C++ does that by a safeguarded Newton in
s, each step a full `evaluate` and so a temperature solve of its own, which
made the seed 80% of a cold solve. The table here solves its energy field for
log T instead and samples the entropy there, one temperature solve per pass
(`entropy_from_energy`); the analytic EOSs, whose `evaluate` is closed-form,
keep the nested solve. The seed lands elsewhere than the C++'s, and with the
C++'s three passes it sent more states to the fallback, so `seed_passes`
defaults to 5. Cold solves then cost 2.1–4.3 times less on the host than
before, take fewer Newton steps and fall back less often on every real table.
On an H200 they gained 1.4–1.6 times on the real tables at Float64 and 4.1 on
the synthetic one, where no state falls back
([HISTORY](HISTORY.md#deliberate-deviations-from-the-c)); the fallback is why,
and running it as a second kernel is the remedy
([Discipline in `core/`](#discipline-in-core)).

Everything else follows the C++ decision for decision, including the parts that
look wrong until you read why: the Newton step is clamped and taken
unconditionally rather than backtracked, convergence is tested on the
cosh-normalized residual, comparisons use a scaled max-norm rather than an L2
norm, the sort is the hand-rolled insertion sort, the guard precedence in the
tails is kept, and a Newton iterate is substituted into a *failed* fallback
only. These measured-and-kept quirks are load-bearing.

## Threading

The two refined-grid scans in `build_eos` that derive κ dominate build cost
and run one Yₑ slice per thread. On twelve threads LS220 builds in 2.4 s against 15 s serial,
and the 391×163×66 SRO table in 7.9 s against 37 s.

The partition is one accumulator per slice — **fixed, not thread-dependent** —
merged in slice order, and the reduction uses the same `<` test as the inner
loop rather than `min()` so a NaN cannot displace a real minimum. The result is
therefore bitwise identical whatever the thread count, which matters because κ
is part of the EOS identity.

Each slice body is a separate function rather than an inline loop body. Inlined,
the scalar minimum was boxed into the *enclosing* frame and shared between
threads — a race that also allocated once per grid point, and that shifted κ by
~6e-8 between thread counts, small enough to pass a tolerance-based check. A
rebuild-and-compare-bitwise test guards against it returning, and CI sets
`JULIA_NUM_THREADS=4` because that test is inert on one thread.

`check_table` is still serial. Its `rms` accumulation is order-dependent, so
threading it would change reported values rather than being a free win.

## Testing

Bit-identical cross-language output is unreachable in general: Julia's `log10`
and `exp10` differ from the system libm by about one unit in the last place, and
the table is stored logarithmically. So correctness rests on oracles that need
no reference implementation, in this order:

1. **Mathematical invariants**, which check points the *code* chose rather than
   points chosen in advance. The sharpest is that the extension machinery is
   *bitwise* transparent inside the physical box — 14,014 field comparisons —
   which is what lets the tails exist without perturbing interior results.
   Others: the interpolation property, C² continuity at every seam as a
   convergence statement, exactly-exponential tail asymptotics, `U ≥ 0` over the
   extended box, σ monotone in u, prim2con→con2prim round trips, bitwise-idempotent
   projection, and the projection's own bar — *the returned conservatives must
   re-solve and reproduce the returned primitives*.
2. **Automatic differentiation** replaces every finite-difference check the C++
   has to use. The analytic `U_ρ`, `U_s`, `U_ρρ`, `U_ρs` and the con2prim
   Jacobian agree with ForwardDiff to 1e-14 through 3e-13, where the C++ must
   tolerate ~1e-6. A dropped chain-rule term cannot hide behind that.
3. **Closed-form ground truth**, the analytic ideal gas in `synthetic.jl`. It is
   shipped in `src/` rather than `test/` because it is the only way to get a
   table without a several-hundred-megabyte file — useful to a downstream hydro
   code's own tests, and the reason the C++ ships its equivalent too. The
   analytic EOSs are checked the same way:
   - against ForwardDiff;
   - by the generalized polytrope's continuity at every break, to 1e-13;
   - by reproducing the paper's published SLy crust constants;
   - and through a test-only `ToyEOS` that implements nothing but the interface
     contract and runs the whole solver.

   Their round trips are asserted *per condition number*: ρ scaled by `hW²`,
   and `s` by the fraction of τ and of U that entropy controls. When `U ≪ 1`,
   or on the hybrid's cold end, the entropy is physically ill-determined, and a
   flat 1e-12 bound would be either false or vacuous.
4. **Statistical agreement on real tables**, gated behind `ENTROPYEOS_TABLE_DIR`.
5. **The C++ itself**, run through EntropyEOS_jll on synthetic tables
   ([§ Validation against the C++](#validation-against-the-c)).

Gating is by environment variable so CI needs nothing: `ENTROPYEOS_TABLE_DIR`
for the five real tables, `ENTROPYEOS_TEST_GPU` for a GPU backend. Both skip
with an `@info` rather than failing. The C++ cross-check needs no gate, since
EntropyEOS_jll is a test dependency; it skips likewise on a platform the JLL
has no build for.

Iteration counts and `C2PResult` are never compared across platforms, only
values: a state near a decision boundary legitimately takes a different path.

### Static analysis

`Aqua` covers what unit tests structurally cannot — a declared but unused
dependency, a missing compat bound, a stale export, a method ambiguity. It
runs in the suite so that the next such problem is found immediately (for the
one that was found late, see [HISTORY](HISTORY.md#static-analysis)).

`JET` asserts the property the GPU path depends on and that `@allocated` can
only measure indirectly: no runtime dispatch and no type instability anywhere in
`bspline_eval3`, `evaluate`, `prim2con`, `con2prim`, `con2prim_safe` or
`apply_policy`. Analyse concrete argument types through a wrapper function — a
closure over non-const globals reports its own captures as dynamic dispatch and
buries the real signal.
JET tracks the compiler closely enough that its results can shift with a Julia
release, so it is pinned to 1.11 and newer rather than letting a new Julia turn
a green suite red.

### Allocation gates

These assert the property the whole GPU design rests on, and they are subtle to
measure. Two rules:

- Measure from a **top-level function taking concrete arguments**. A closure
  defined inside a `@testset`, or a varargs `f(args...)` splat, allocates 16
  bytes of its own on x86-64 and the measurement then reports the harness.
- They are skipped when `--check-bounds=yes` or coverage is active, because
  those disable `@inbounds` and block optimization, so the property genuinely
  cannot hold. `julia-runtest` turns both on by default, so CI carries a
  separate `allocations` job with them off. Keeping bounds checking on
  elsewhere is worth more than the gate: it catches real indexing bugs.

## Latency

`con2prim` inlines a deep tree: the spline contraction sits inside the designed
tails, inside the EOS evaluation, inside the residual, inside the Newton loop,
the inner solve, the bracket scan and the cold seed. Compiling that on first
call is not cheap, and a hydro code would pay it at startup.

`src/precompile.jl` runs the whole public `Float64` path over a five-point-per-axis
table with the refinement and extension widths turned down — enough to reach
every code path, since the types and therefore the specializations are identical
to a real build's. Measured:

| | precompile, once | cache | first call, every session |
| --- | --- | --- | --- |
| without the workload | 3.1 s | 1.6 MB | 3.65 s |
| with | 5.3 s | 5.3 MB | **0.68 s** |

and 0.65 s of that 0.68 s is `using` itself. `Float32` is deliberately not
precompiled: it would roughly double both figures for a path mostly taken on a
GPU, where the kernel is compiled separately anyway.

## Precision and devices

Measured by the scripts in `study/`, which are run by hand and are not part of
the test suite; the numbers are on the "Precision and GPUs" documentation page
(`docs/src/precision.md`), and summarized in [README § GPUs](README.md#gpus).

What is easy to get wrong here:

- **The Float32 limits are mostly the hydro state's, not the solver's.** The
  Float64 solver applied to Float32-rounded conservatives — the study's
  "floor" — already fails on 7–13% of cold states, because τ in Float32 cannot
  resolve their thermal energy. Judge any Float32 change against the floor, not
  against Float64.
- **Float32 derivatives lose ~3 digits** to cancellation between spline
  coefficients of size ~20. Device–host differences at Float32 are therefore
  ~1e-4, not ~1e-7, and are not a device defect.
- **Products of two conserved quantities overflow Float32** (they reach 1e20
  each); keep such terms in ratio form
  ([HISTORY](HISTORY.md#precision-and-devices)).
- **`con2prim_tol(Float32)` is measured**, at 512 eps, just above the Float32
  residual's noise floor. The generic 64-eps fallback was below it and made
  Newton stall.
- **Cheaper logarithms do not pay.** Not-quite-transcendental replacements for
  `log10`, `exp10` and the hyperbolic functions were measured and rejected:
  those calls are a few percent of the work, and an NQT axis spoils the C²
  fit's p and cs² near every power of two
  ([HISTORY](HISTORY.md#not-quite-transcendental-functions)).

## Validation against the C++

Where the two can be compared, they agree.

- **In the test suite**, `test/test_cpp_crosscheck.jl` runs the C++ `eos_test`
  from EntropyEOS_jll (1.1.0+1). The C++ writes its three synthetic tables
  (clean, seeded, dirty), and both implementations then read the same file. On this
  Mac (aarch64), measured:
  - the generators agree to 8e-16;
  - every `check_table` class count matches, and every node's metric agrees to
    4e-11 (the test allows 1e-9);
  - κ and `m_B*` agree to 2e-15;
  - the con2prim audit's quantiles agree to within 15% above rounding level.
    For that comparison the test ports the audit's per-state generator, so
    Julia solves exactly the C++'s 20,000 states.

  `ENTROPYEOS_EOS_TEST` names another `eos_test` binary to check against, such
  as a release the JLL does not ship yet. The test reads its `--version`. From
  1.2.0 `--level con2prim` repairs in memory and so drops the hottest T row, and
  Julia then builds with `trim_t_hi_rows = 1` to solve the same table. Against
  a local build of the C++ 1.2.0 candidate (PR eschnett/EntropyEOS#2 at
  1743044) everything passes except the dirty table, which its 1.2.0 preset
  gives a `SeamDefect` that the Julia generator lacks; that comparison is
  marked broken.

  The one expected difference is `delta_T` at the 330 nodes of the dirty table
  whose T stencil spans its planted logenergy plateau. There the ε values are
  equal, so the stencil sum is 0 in exact arithmetic, and what either side
  computes is a rounding residual that depends on the last bit of each ε. If
  the residual is exactly 0, δ_T overflows to Inf and the node is skipped as if
  its input were non-finite; otherwise it is reported as 1e16 to 1e17. Two
  things move that last bit. The FMA contractions clang makes by default made
  290 of the 330 nodes differ on this Mac; the JLL is built with
  `-ffp-contract=off` since 1.1.0+1, which leaves 2. The math library makes the
  rest: clang turns `pow(10, x)` into macOS's `__exp10`, which is one ulp from
  Julia's `exp10` at those 2 nodes. On Linux x86-64, where the build never
  contracted, the C++ skipped some nodes that Julia evaluates, presumably from
  glibc's rounding. The plateau itself is reported by `logenergy_nonmonotone_T`
  either way. The test exempts exactly these nodes, in both directions, and
  requires all others to agree.
- The fitted B-spline coefficients are **bitwise identical** when the C++ is
  built with `-ffp-contract=off`. At clang's default some differ in the last
  bits, because clang fuses the elimination update into an FMA and Julia does
  not — a compiler setting, not a translation difference.
- Reproducing the configuration of the C++ README's worked example returns every
  digit it publishes.
- Running both audit harnesses on the unrepaired LS220, with matched sampling
  and a 1e-3-perturbed warm start over 20,000 warm and 2,000 cold states: 0 warm
  failures each, 1 cold failure each, and density round-trip quantiles matching
  to ~10% at the median, p99 and p99.9:

  | | C++ | Julia |
  | --- | --- | --- |
  | warm Newton / fallback / failed | 19964 / 36 / 0 | 19958 / 42 / 0 |
  | cold failures | 1 | 1 |
  | density round trip, median | 1.89e-13 | 1.97e-13 |
  | p99 | 2.55e-09 | 2.43e-09 |
  | p99.9 | 1.03e-08 | 1.05e-08 |

  This table predates the [numerical deviations](#deliberate-deviations-from-the-c),
  when the two ran the same algorithm. With them, Julia's warm fallbacks on
  LS220 drop to 0 and its cold failure and quantiles stay put, so the warm row
  no longer matches the C++'s (measured in
  [HISTORY](HISTORY.md#deliberate-deviations-from-the-c)).
  Matching the sampling matters: `con2prim_audit.cpp` uses 5%-per-side margins
  on the density box and entropy window and perturbs the warm start by 1e-3.
  Sampling the corners instead, or warm-starting from the exact truth, changes
  the failure rate by more than an order of magnitude in either implementation.
- Both reproduce the documented accept-and-guard outlier tail on DD2 — a
  handful of states per 20,000 where the round trip is poor. That is a
  property of the tables, and a port showing *none* of them would be sampling
  differently rather than doing better. On DD2 the C++'s own worst warm density
  error is 1.6; this port's is 0.18.

## Open items

- **(open)** No golden-file cross-checks against `eos_test --csv`. The invariant and
  closed-form oracles turned out strong enough that this was never needed, but
  the check-class violation sets would be cheap and exact to compare.
- Float32 derivative accuracy (`p`, `cs²`, `μ̃`) is limited by the
  representation — spline coefficients of `log10(ε + shift)` — not by the code.
  Storing the coefficients with a per-cell offset and recovering ε with `expm1`
  could lift it (untested) **(open)**, at the cost of departing from the C++'s
  representation.
- `check_table` is serial; see "Threading".
- **(open)** On CUDA.jl 6.4.2, a kernel that calls `con2prim_safe` and then
  `apply_policy` faults with an illegal address on states pushed off the valid
  range (2²⁰ such states, H200, either precision); on valid states it runs. Each
  level alone runs on the same states, and so do two chained `con2prim_safe`
  calls, in 1.1.0's code as in this one; compiled with `always_inline=true` the
  fused kernel runs too (91 s to compile), and gives finite results, so the
  states raise no genuine exception. The fault is a 1-byte write to address 0,
  the signature of an exception reported through a null pointer, from inside
  the separately compiled device functions CUDA.jl 6.4 links (`.extern .func`).
  Inlining `apply_policy` or `pol_package`, and raising the stack limit to 64 kB,
  did not help. Until it is understood, call the two levels from separate
  kernels on CUDA. Metal is not affected.
- `inner_solve_w`'s root is unique only where cs²·v² < 1 along its path,
  which since the search stays in the box means inside the box. The real
  tables violate that in places: every one has cs² > 1 near its top density,
  at all temperatures (LS220 1.3–1.4, DD2 just above 1), and DD2 and SFHo have
  spikes at their hottest edge (cs² 4.3 at ρ ≈ 10⁷ in DD2, 1.17 within ~0.01 in
  log T of T_max at ρ ≈ 5·10¹²). There the inner solve can have several roots,
  in 0–3.7% of states sampled with w uniform in [0, 4], and returns whichever
  it reaches; for 0.35–1.5% the sampled root is not the lowest (measured by
  `study/projection_faces.jl` and `study/root_certificate.jl`, 2026-10-22,
  before the search was confined). The C++ 1.2.0 repair removes nearly all of
  it ([Where the answer is not unique](#where-the-answer-is-not-unique)); table
  repair itself is out of scope ([Scope](#scope)). On a cold start it
  makes the solve fail, or the projection's diagnosis misclassify a valid
  state ([Where the answer is not unique](#where-the-answer-is-not-unique)). A
  possible remedy, not proposed: on the failure path only, follow each root
  through an ordered sweep in s, seeding every inner solve from its neighbour,
  and diagnose and solve on those roots.
- **(open)** A `ProjectionOptions` ceiling above the EOS's box is not reached
  by a solve, since con2prim searches only inside the box. A state valid for
  such a projection but above the box then ends `NOT_CONVERGED`. No caller sets
  one; lowering `ρ_max` to the box's top in `proj_ρ_max` would make the state
  `ρ_CEILED` instead.
- **(open)** On a table's cold edge the energy residual cannot always meet the
  tolerance. `eval_at` forms U as (1 + Û)/κ − 1, so U is resolved to
  eps(1 + U) absolutely, and at rest f₂ to about eps/U: 1.2e-12 on the
  synthetic table's coldest states, where U ≈ 2e-4, against a tolerance of
  1e-12. Newton converges on every such state sampled, but the fallback
  re-solves its interval's ends from a fresh warm start, which can flip the sign
  of a residual at that floor, and a state whose root is that end then fails as
  `failed_no_bracket`. With the Newton disabled and the warm start perturbed,
  5 of 960 states on the edges of the window fail this way (22 before the
  [deviations](#deliberate-deviations-from-the-c) of 2026-10-23). Two possible
  remedies, both untested: a tolerance with a floor of a few eps·z/τ, or a scan
  that returns its ends' warm starts so that the re-solve reproduces them.

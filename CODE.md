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
- [State policy flags](#state-policy-flags)
  - [Two-level projection (proposed)](#two-level-projection-proposed)
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
  state_policy.jl    the never-fails layer
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
the abstract type. The contract is stated once, in `core/eos_interface.jl`.
Introducing it changed no arithmetic ([HISTORY](HISTORY.md#the-eos-interface)).
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
`evaluate` neither clamps nor iterates; the box only sets flags. The terms are
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

## State policy flags

`con2prim_safe` and `project_prim_state` report each repair as a bit in
`policy_flags` rather than leaving the caller to infer it from the returned
state. Every clamp sets its own bit, including those that leave a state the
following projection finds already in range. The bit values are the same as in
the C++.

| bit | flag | the repair |
| --- | --- | --- |
| 8 | `FLAG_POL_ATMOSPHERE` | replaced the whole state by the atmosphere at rest |
| 9 | `FLAG_POL_CEILING` | hit a collapse ceiling, then excised or projected onto the ceiling |
| 10, 11 | `FLAG_POL_S_FLOORED`, `_S_CEILED` | moved s onto the physical entropy window |
| 12 | `FLAG_POL_W_CAPPED` | clamped w into `[0, w_cap]`, preserving D |
| 13 | `FLAG_POL_ρ_CEILED` | lowered ρ to ρ_ceiling: rest mass removed |
| 14 | `FLAG_POL_YE_CLAMPED` | clamped Yₑ into the table |
| 15 | `FLAG_POL_NONFINITE` | saw non-finite input or D ≤ 0 (with `ATMOSPHERE`) |
| 16 | `FLAG_POL_ρ_FLOORED` | raised ρ to ρ_atm without an atmosphere reset: rest mass added |

`FLAG_POL_ρ_CLAMPED` is the two-bit mask `ρ_FLOORED | ρ_CEILED`, and
`FLAG_POL_ANY` covers bits 8–16. Bits 6–7 stay reserved for solver flags, which
is why the newest policy bit is 16.

`ρ_FLOORED` is the signal a hydro code may need to act on. It arises in the
entropy-floor repair, which keeps the inner solve's rapidity: for a thin, fast
state `D/cosh w` falls below ρ_atm, the density is raised, and the returned
conservatives carry more mass *and* more momentum than the input. Whether such a
point should also be put at rest is left to the caller, since it depends on the
evolution scheme (TreeGRRMHD does reset it). A rapidity cap also raises ρ, as
D/cosh(w_cap), but preserves D, so it does not set `ρ_FLOORED`.

### Two-level projection (proposed)

*Proposed 2026-10-22, not decided. It would replace the policy described above,
and be a breaking change (2.0.0).*

**Why.** `con2prim_safe` today does two jobs at once: it projects a state onto
the valid range, which is mechanics, and decides what a state should become —
the atmosphere, the collapse ceiling — which is physics and depends on the
evolution scheme. Mixing them has costs. The flags report facts for some repairs
and decisions for others. An energy above the hottest state is treated as a
collapse and written at ρ_ceiling even for a hot near-vacuum cell, which blew up
TreeGRRMHD's magnetized atmosphere tubes. An atmosphere reset discards the
clamps that preceded it. And the policy carries options (`collapse_to_atmosphere`,
`atm_trigger`, `s_atm`, `ye_atm`, `τ_max`) that only one of the two jobs needs.
The proposal separates them.

**Level 1: projection.** `con2prim_safe` returns the valid state closest to the
input and flags each face of the valid range that was active. It never replaces
a state by the atmosphere and has no notion of collapse. The valid range is

- ρ ∈ [ρ_min, ρ_max] (today ρ_atm and ρ_ceiling),
- Yₑ ∈ [y_lo, y_hi], the table's range,
- s ∈ [s_min(ρ, Yₑ), s_max(ρ, Yₑ)], the physical window,
- w ∈ [0, w_cap].

That is seven faces. w ≥ 0 is not one, since w is a magnitude and the direction
is carried separately. The two Yₑ faces share a bit; the returned Yₑ says which.

"Closest" has no metric: D, S and τ have different units, and a priority order
does not work either (keeping D at the density ceiling would mean raising w,
creating momentum). Instead each face has a rule. Change as little as possible,
and prefer removing over adding:

| face | flag | what gives way |
| --- | --- | --- |
| s_max | `S_CEILED` | τ lowered |
| s_min | `S_FLOORED` | τ raised (unavoidable) |
| w_cap | `W_CAPPED` | S lowered, D kept |
| ρ_max | `ρ_CEILED` | D lowered |
| ρ_min, with D ≥ ρ_min | `ρ_FLOORED` | S lowered: w reduced to acosh(D/ρ_min), D kept |
| ρ_min, with D < ρ_min | `ρ_FLOORED` | D raised to ρ_min·cosh w, w from S at ρ_min |
| y_lo, y_hi | `YE_CLAMPED` | D_Y set to Yₑ·D with Yₑ clamped |

Quantities a row does not name are kept, to the solver's tolerance. Yₑ is kept
when D changes, so D_Y moves with D: matter added or removed has the cell's
composition ([HISTORY](HISTORY.md#state-policy-flags) has the rejected
alternative). Two rows differ from 1.1.0: at the ρ_min face a state with
D ≥ ρ_min is slowed down rather than given mass, and an energy above the hottest
state keeps D and S and lowers τ rather than going to ρ_max. With D < ρ_min,
whether w should keep S (as above) or be set to zero, which adds less mass,
is **(open)**.

Two outcomes are not projections, and get a bit of their own:

- `NONFINITE`: a non-finite input, or D ≤ 0, where Yₑ = D_Y/D is undefined. There
  is no closest valid state.
- `NOT_CONVERGED` (new, bit 17): the solve failed but the diagnosis finds a valid
  state inside the range, so there is nothing to project onto.

Both return a fixed valid state (ρ_min, at rest, the middles of the Yₑ range and
the entropy window) only so that a kernel never propagates NaN. Level 2 is
expected to replace it, so it takes no parameter.

The faces interact: the entropy window depends on ρ, ρ = D/cosh w, and w follows
from S through h, which depends on s. Only Yₑ decouples, and is clamped first.
The rest is an active-set problem: solve with no face active, add the violated
face, re-solve in the remaining unknowns, repeat. The s-face solve is a root find
in w with f₁(w) = 0 at s = s_edge(D/cosh w), consistent with the window at the
density actually written (TreeGRRMHD's EOS-9 measured that the current
endpoint solves, taken at the window of the failed solve's density, miss the
written state's window by up to 0.27 of its width). That this terminates with a
unique answer rests on f₁ being monotone in w along a face and f₂ monotone in s,
measured only on the synthetic table so far **(open)**; the real tables, whose
windows are less smooth, are where it has to be shown.

The contract is a set of invariants:

- a set flag means that face was active, and the returned state lies on it;
- `policy_flags == 0` means `cons` is the input bit for bit, as today;
- every output is a fixed point: fed back, it returns flags 0 and the same state;
- `check_con_state` predicts the faces, sharing the diagnosis routine with
  `con2prim_safe`. Its cheap mode stays arithmetic: non-finite input, D < ρ_min,
  D > ρ_max·cosh(w_cap), and Yₑ out of range.

`project_prim_state` is the same projection on primitives, where every face is a
plain clamp.

**Level 2: the application's physics.** A separate kernel-side function takes the
input conservatives, level 1's output and flags, and a level-2 policy, and may
replace the state by another valid one. It needs no further flags: it can
compute ΔD, ΔS and Δτ from the two conservative states. EntropyEOS ships a
reference implementation that reproduces today's behaviour:

- `ρ_FLOORED`, or D < `atm_trigger`·ρ_min: the atmosphere at rest, at ρ_min, with
  `s_atm` and `ye_atm`; sets `ATMOSPHERE`;
- `ρ_CEILED` with `collapse_to_atmosphere`: the atmosphere; sets `CEILING`;
- `NONFINITE` and `NOT_CONVERGED`: the atmosphere.

An application with its own rules (TreeGRRMHD puts any state with added rest
mass at rest) replaces this function rather than configuring it.

**What moves.** Level 1's policy keeps ρ_min, ρ_max and w_cap. `s_atm`, `ye_atm`,
`atm_trigger` and `collapse_to_atmosphere` move to level 2's. `D_max` and
`τ_max` go: the cheap check computes ρ_max·cosh(w_cap) itself, and τ no longer
short-circuits the solve (`τ_max` came from a 5×3 grid and stopped states the
table represents). Bits 10–14 and 16 keep their numbers with the meaning "face";
`ATMOSPHERE` and `CEILING` become level-2 bits; `NONFINITE` stays at level 1.

**Consequences.** Outputs and flags change for existing inputs, and
`PolicyOptions` loses fields: 2.0.0. It is a departure from the C++ beyond the
table under [Deliberate deviations](#deliberate-deviations-from-the-c), so
either the C++ follows, as it did for 1.1.0, or this becomes a section of its
own there. TreeGRRMHD's EOS-9 request is largely subsumed: its edge-consistent
root find, its guards on the endpoint solves, the shared diagnosis with
`check_con_state`, and dropping the `τ_max` test carry over; its flag rules are
replaced by the face flags.

## Discipline in `core/`

No heap allocation, no `throw`, no `@fastmath`, no closures, no `String`; array
reads `@inbounds`, small helpers `@inline`. Kernel-side structs stay `isbits`;
a NaN sentinel is not replaced with `Union{Nothing,T}`, because NaN is
load-bearing and flows through arithmetic. Three consequences worth naming:

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
- **The one stack buffer.** `bracket_scan` needs 33 entries live at once and
  cannot recompute them (each costs an inner solve, and the warm-start chain
  fixes their order). It uses `MVector{33,T}`, local and non-escaping so it is
  promoted out of the heap. The cap is a `Val` parameter so a GPU caller can
  shrink per-thread local memory without touching the algorithm
  ([HISTORY](HISTORY.md#the-one-stack-buffer)).

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
   projection, and the policy layer's own bar — *the returned conservatives must
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

Gating is by environment variable so CI needs nothing: `ENTROPYEOS_TABLE_DIR`
for the five real tables, `ENTROPYEOS_TEST_GPU` for a GPU backend. Both skip
with an `@info` rather than failing.

Iteration counts and `C2PResult` are never compared across platforms, only
values: a state near a decision boundary legitimately takes a different path.

### Static analysis

`Aqua` covers what unit tests structurally cannot — a declared but unused
dependency, a missing compat bound, a stale export, a method ambiguity. It
runs in the suite so that the next such problem is found immediately (for the
one that was found late, see [HISTORY](HISTORY.md#static-analysis)).

`JET` asserts the property the GPU path depends on and that `@allocated` can
only measure indirectly: no runtime dispatch and no type instability anywhere in
`bspline_eval3`, `evaluate`, `prim2con`, `con2prim` or `con2prim_safe`. Analyse
concrete argument types through a wrapper function — a closure over non-const
globals reports its own captures as dynamic dispatch and buries the real signal.
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

## Validation against the C++

Where the two can be compared, they agree.

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
- **(proposed)** Splitting `con2prim_safe` into a projection that reports faces
  and a separate, replaceable physics layer; see
  [Two-level projection](#two-level-projection-proposed).

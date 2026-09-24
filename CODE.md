# Code design

Companion to [README.md](README.md), which covers what the package is for and
how to call it. This document covers how it is built and why. The physics is
specified by the C++ repository's design notes — `eos-adapter-F-to-U.md`,
`con2prim-entropy-rapidity.md`, `eos-causal-tail.md` — which this port follows
rather than restates.

## Environment

- Julia 1.10 or newer. Developed on 1.13; CI covers 1.10 and 1.11 on Linux,
  macOS and Windows.
- Dependencies: `HDF5` (table I/O), `Adapt` (the entire GPU contract),
  `StaticArrays` (one stack buffer), `EnumX` (namespaced outcome enums).
  Deliberately **not** dependencies: `CUDA`, `Metal`, `KernelAbstractions`.
  A caller brings its own backend; see "GPU" below.
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
  bspline_eval.jl    uniform cubic B-spline evaluation, value and derivatives
  adapter_eval.jl    the designed domain extensions, then evaluate() and srange()
  adapt.jl           Adapt rules and explicit scalar-type narrowing
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

Table-side units are cgs plus MeV plus k_B per baryon. The run-time path is
unit-free: conversion happens once, when the adapter is built, and the view
carries the pre-computed scalars. At that boundary ρ means **ρ\* = κ·ρ** in
κ-rescaled g/cm³, `s` is k_B per baryon, `w` is the rapidity, and `D`, `τ`,
`S_par`, `S_perp`, `B²` are all g/cm³.

κ is part of the EOS identity, not an internal detail. It re-zeroes the energy
so `U ≥ 0` — which is what lets `prim2con` build τ without cancellation — and
the rescaling is exact rather than approximate: `ρ*(1+U) = ρ(1+ε)` holds
bitwise, and is asserted. A table swap that changes κ changes `D`, so
checkpoints are not interchangeable across it.

## Type design

**Storage type versus working type.** A view is parameterized on the type of
its coefficient arrays; the working type is whatever promotes from the call's
arguments. So `evaluate(v::EOSTableView{Float64}, ρ::Dual, ...)` returns
`EOSPoint{Dual}` against a plain `Float64` table. That is what lets ForwardDiff
differentiate through the whole adapter without a dual-valued table, and it is
why kernel-side result types are constrained to `Real` rather than
`AbstractFloat` (`ForwardDiff.Dual` is `Real` but not `AbstractFloat`). Storage
types keep `AbstractFloat`, since that is an array element type.

**The isbits boundary is one function**, `EOSTableView(::EOSTable)`. Everything
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

## Discipline in `core/`

No heap allocation, no `throw`, no `@fastmath`, no closures, no `String`; array
reads `@inbounds`, small helpers `@inline`. Three consequences worth naming:

- **Domain-safe math.** `log10`, `sqrt` and `floor(Int, ·)` throw in Julia where
  C returns NaN or is merely unspecified. `core/defs.jl` provides `safe_log10`,
  `safe_sqrt` and `trunc_floor`. `Base.isnan` and `isfinite` are already exactly
  the self-comparison tricks the C++ hand-rolls, so they are used directly — but
  only because `@fastmath` is banned, which a test enforces by grep.
- **Per-type tolerances.** Every numeric constant is a function of the scalar
  type, not a literal, because a bare `Float64` literal silently widens `Float32`
  arithmetic. The `Float64` methods reproduce the C++'s measured values exactly;
  the generic ones scale with `eps(T)` so a narrower type at least runs.
- **The one stack buffer.** `bracket_scan` needs 33 entries live at once and
  cannot recompute them (each costs an inner solve, and the warm-start chain
  fixes their order). It uses `MVector{33,T}`, local and non-escaping so it is
  promoted out of the heap. The cap is a `Val` parameter so a GPU caller can
  shrink per-thread local memory without touching the algorithm. This was the
  design's biggest open risk; it is resolved by the kernel compiling and running
  on Metal.

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
norm, and a Newton iterate is substituted into a *failed* fallback only.

## Threading

The two refined-grid scans in `build_eos` dominate build cost and run one Yₑ
slice per thread. On twelve threads LS220 builds in 2.4 s against 15 s serial,
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
   code's own tests, and the reason the C++ ships its equivalent too.
4. **Statistical agreement on real tables**, gated behind `ENTROPYEOS_TABLE_DIR`.

Gating is by environment variable so CI needs nothing: `ENTROPYEOS_TABLE_DIR`
for the five real tables, `ENTROPYEOS_TEST_GPU` for a GPU backend. Both skip
with an `@info` rather than failing.

### Static analysis

`Aqua` covers what unit tests structurally cannot — a declared but unused
dependency, a missing compat bound, a stale export, a method ambiguity. It is
how the unused `PrecompileTools` entry was eventually found, and it now runs in
the suite so the next one is found immediately.

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
(`docs/src/precision.md`). In short: Float64 is validated on CUDA (H200) with
the device indistinguishable from the host; a Float32 table with Float64
arithmetic is validated too; pure Float32 is usable with measured limits.

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
  each). `seed_z_solve` did this until the study found it; keep such terms in
  ratio form.
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
  to ~10% at the median, p99 and p99.9.
- Both reproduce the documented accept-and-guard outlier tail on DD2. That is a
  property of the tables, and a port showing *none* of them would be sampling
  differently rather than doing better. On DD2 the C++'s own worst warm density
  error is 1.6; this port's is 0.18.

## Open items

- No golden-file cross-checks against `eos_test --csv`. The invariant and
  closed-form oracles turned out strong enough that this was never needed, but
  the check-class violation sets would be cheap and exact to compare.
- Float32 derivative accuracy (`p`, `cs²`, `μ̃`) is limited by the
  representation — spline coefficients of `log10(ε + shift)` — not by the code.
  Storing the coefficients with a per-cell offset and recovering ε with `expm1`
  could lift it (untested), at the cost of departing from the C++'s
  representation.
- `check_table` is serial; see "Threading".

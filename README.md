# EntropyEOS.jl

* [![Documentation](https://img.shields.io/badge/Docs-Dev-blue.svg)](https://eschnett.github.io/EntropyEOS.jl/dev/)
* [![GitHub CI](https://github.com/eschnett/EntropyEOS.jl/workflows/CI/badge.svg)](https://github.com/eschnett/EntropyEOS.jl/actions)
* [![codecov](https://codecov.io/gh/eschnett/EntropyEOS.jl/graph/badge.svg?token=2Z29OBMVB6)](https://codecov.io/gh/eschnett/EntropyEOS.jl)

Handling tabulated equations of state for general-relativistic hydrodynamics.

This is a Julia translation of the C++ library
[EntropyEOS](https://github.com/eschnett/EntropyEOS), covering the run-time
path a hydro code actually calls: loading a table, checking it, and using it.

[CODE.md](CODE.md) describes how the package is built and why.

## What it does

Real EOS tables store a bundle of columns `F(ρ, T, Yₑ)` on a rectangular grid.
A hydrodynamics solver wants something else: a single thermodynamic potential
with consistent derivatives. This package builds that potential,

```
U(ρ, s, Yₑ) = ε(ρ, T(ρ, s, Yₑ), Yₑ),    where  σ(ρ, T, Yₑ) = s
```

from which `p = ρ²U_ρ`, `T̂ = U_s`, `h = 1 + U + ρU_ρ` and
`h·cs² = 2ρU_ρ + ρ²U_ρρ` all follow. Consistency is then structural rather
than inherited: `U` is a genuine single function, so its mixed partials
commute whether or not the underlying table satisfies its own Maxwell
relations.

On top of that sits a `prim2con`/`con2prim` pair that iterates on **entropy**
and **rapidity**. Both choices remove constraints from the iterate space: any
`(s, w)` inside the table domain is a physical state, and rapidity is well
conditioned both as `v → 0` and as `v → 1`. A policy layer above it never
fails, returning a valid and exactly solvable state for any input at all.

## Usage

```julia
using EntropyEOS

# Load, check and build, once at startup.
table = read_stellarcollapse("LS220_repaired.h5")   # or make_synthetic_table()
report = check_table(table)
report.status === Status.fatal && error("broken table")

eos = build_eos(table)
v = EOSTableView(eos)                    # small, immutable, device-ready
pol = default_policy(v, 1e3 * v.κ)       # atmosphere and collapse ceilings

# A valid primitive state. Note that ρ is ρ* = κ·ρ, in κ-rescaled g/cm³.
ρ, yₑ, w = exp10(0.5 * (v.x_lo + v.x_hi)), 0.4, 0.8
sr = srange(v, ρ, yₑ)                    # the physical entropy window here
s = 0.5 * (sr.s_min + sr.s_max)

pt = evaluate(v, ρ, s, yₑ, NaN)          # NaN = no warm start
@show pt.p pt.T_MeV pt.cs²

# Forward, then back.
c = prim2con(v, ρ, s, yₑ, w, 0.1ρ, 0.3, pt.u_solved)
cons = Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²)
rec = con2prim(v, cons, Con2PrimOptions(), s, w, pt.u_solved)   # warm-started

# The path that never fails, for any input at all.
safe = con2prim_safe(v, cons, Con2PrimOptions(), pol)
safe.policy_flags & FLAG_POL_ANY == 0 || @info "state repaired; adopt safe.cons"
```

Warm-start state is threaded explicitly rather than cached, so evaluation is
pure and re-entrant and calls are safe to run in parallel across grid points.

### Units

The run-time path is unit-free; conversion happens once, when the adapter is
built. At that boundary ρ means ρ\* in **κ-rescaled** g/cm³, `s` is in k_B per
baryon, `w` is the rapidity, and `D`, `τ`, `S_par`, `S_perp` and `B²` are all
g/cm³. Feeding a raw table density where ρ\* is expected is the single easiest
way to misuse the library. κ is part of the EOS identity, not an internal
detail: a table swap that changes κ changes `D`, so checkpoints are not
interchangeable across it.

## GPUs

The kernels are allocation-free, exception-free and generic in the scalar type,
and the types carry `Adapt.jl` rules. That is the whole contract — the package
depends on no GPU package at all:

```julia
using Adapt, KernelAbstractions, Metal
dv = adapt(MtlArray, EntropyEOS.narrow(v, Float32))
# pass dv into your own kernel; KernelAbstractions adapts it the rest of the way
```

Verified by running `evaluate` and `con2prim` as KernelAbstractions kernels on
a Metal GPU. Enable the GPU testset with `ENTROPYEOS_TEST_GPU=metal` (or
`cuda`) and the corresponding package installed.

## Status

The run-time path is complete and tested: table loading and checking, the
B-spline fit, the adapter, `prim2con`, `con2prim`, and the never-fails policy
layer. Roughly 31,000 assertions pass.

All five real tables are exercised end to end — LS220, the SRO LS220
re-tabulation, DD2 (original and repaired) and SFHo — covering grids from
234×136×50 to 391×163×66. Set `ENTROPYEOS_TABLE_DIR` to enable those tests.

Building the adapter is dominated by the refined-grid scans that derive κ.
Those are threaded over the Yₑ slices, so the cost falls with
`JULIA_NUM_THREADS`: on twelve threads LS220 builds in 2.4 s and the 391×163×66
SRO table in 7.9 s, against 15 s and 37 s serially. The partition is fixed
rather than thread-dependent and the reduction is a minimum, so the result is
bitwise identical whatever the thread count — which matters, because κ is part
of the EOS identity.

Table *repair* is deliberately not included — it is an offline activity
performed once before a simulation campaign, and the run-time path contains no
repair logic. Use the C++ `eos_repair` tool for that, and `check_table` here to
detect a table that has not been repaired.

## Relationship to the C++ library

Where the two can be compared they agree. The fitted B-spline coefficients are
**bitwise identical** to the C++ when it is built with `-ffp-contract=off`
(at clang's default some coefficients differ in the last bits, because clang
fuses the elimination update into an FMA and Julia does not). Reproducing the
configuration of the C++ README's worked example returns every digit it
publishes.

Statistically the two agree on real tables. Running both audits on the
unrepaired LS220 with matched sampling and a 1e-3-perturbed warm start, over
20,000 warm and 2,000 cold states:

| | C++ | Julia |
| --- | --- | --- |
| warm Newton / fallback / failed | 19964 / 36 / 0 | 19958 / 42 / 0 |
| cold failures | 1 | 1 |
| density round trip, median | 1.89e-13 | 1.97e-13 |
| p99 | 2.55e-09 | 2.43e-09 |
| p99.9 | 1.03e-08 | 1.05e-08 |

Both also reproduce the documented accept-and-guard outlier tail on DD2 — a
handful of states per 20,000 where the round trip is poor. That is a property
of the tables, and a port showing *none* of them would be sampling
differently, not doing better. (On DD2 the C++'s own worst warm density error
is 1.6; this port's is 0.18.)

Exact agreement is not achievable everywhere and is not attempted: Julia's
`log10` and `exp10` differ from the system libm by about one unit in the last
place, and the table is stored logarithmically. So correctness rests on
mathematical invariants and closed-form ground truth rather than on
cross-language comparison — for instance the extension machinery is asserted to
be *bitwise* transparent inside the table, the analytic Jacobian is checked
against automatic differentiation rather than finite differences, and the
policy layer's returned states are re-solved to confirm they reproduce
themselves.

## References

- Evan O'Connor, Christian D. Ott. *A new open-source code for
  spherically symmetric stellar collapse to neutron stars and black holes*.
  Classical and Quantum Gravity 27:114103 (2010),
  [DOI:10.1088/0264-9381/27/11/114103](https://doi.org/10.1088/0264-9381/27/11/114103).

- James M. Lattimer, F. Douglas Swesty. *A generalized equation of state for
  hot, dense matter*. Nuclear Physics A 535:331 (1991),
  [DOI:10.1016/0375-9474(91)90452-C](https://doi.org/10.1016/0375-9474(91)90452-C).

- Matthias Hempel, Jürgen Schaffner-Bielich. *A statistical model for a
  complete supernova equation of state*. Nuclear Physics A 837:210 (2010),
  [DOI:10.1016/j.nuclphysa.2010.02.010](https://doi.org/10.1016/j.nuclphysa.2010.02.010).

- Andre da Silva Schneider, Luke F. Roberts, Christian D. Ott. *Open-source
  nuclear equation of state framework based on the liquid-drop model with
  Skyrme interaction*. Physical Review C 96:065802 (2017),
  [DOI:10.1103/PhysRevC.96.065802](https://doi.org/10.1103/PhysRevC.96.065802).

- Wolfgang Kastaun, Jay Vijay Kalinani, Riccardo Ciolfi. *Robust recovery of
  primitive variables in relativistic ideal magnetohydrodynamics*. Physical
  Review D 103:023018 (2021),
  [DOI:10.1103/PhysRevD.103.023018](https://doi.org/10.1103/PhysRevD.103.023018).

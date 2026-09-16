# EntropyEOS.jl

* [![Documentation](https://img.shields.io/badge/Docs-Dev-blue.svg)](https://eschnett.github.io/EntropyEOS.jl/dev/)
* [![GitHub CI](https://github.com/eschnett/EntropyEOS.jl/workflows/CI/badge.svg)](https://github.com/eschnett/EntropyEOS.jl/actions)

Tabulated equations of state for general-relativistic hydrodynamics.

This is a Julia translation of the C++ library
[EntropyEOS](https://github.com/eschnett/EntropyEOS), covering the run-time
path a hydro code actually calls: loading a table, checking it, and using it.

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

## Status

Under construction; see the milestone list in the development notes. Table
*repair* is deliberately not included — it is an offline activity performed
once before a simulation campaign, and the run-time path contains no repair
logic. Use the C++ `eos_repair` tool for that, and `check_table` here to
detect a table that has not been repaired.

## Precision

All defaults and all validation are at `Float64`. The kernels are generic in
the scalar type, type-stable and allocation-free at `Float32`, and the types
carry `Adapt.jl` rules so they can be moved to a GPU — but the numerics are
not validated below `Float64`, and some tolerances are not representable
there.

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

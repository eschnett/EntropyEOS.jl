# Working in this repository

A Julia translation of the C++ library [EntropyEOS](https://github.com/eschnett/EntropyEOS),
which lives at `/Users/eschnett/src/EntropyEOS` on this machine. Read
[CODE.md](CODE.md) for the design; this file is the short list of things that
are easy to get wrong.

## Scope

Only the run-time path a hydro code calls: load a table, check it, evaluate it,
and invert it. Table **repair is deliberately absent** — it is an offline step
performed once before a simulation campaign, done by the C++ `eos_repair` tool.
`check_table` is what detects a table that still needs it. Do not port
`repair.cpp`, the audit harnesses, or the `tools/` binaries.

## Commands

```bash
julia --project=. -e 'using Pkg; Pkg.test()'          # ~25 s
julia --project=docs docs/make.jl                      # build docs locally
```

Optional test gates, both skip cleanly when unset:

```bash
ENTROPYEOS_TABLE_DIR=/Users/eschnett/src/EntropyEOS/tables   # the five real tables, +25 s
ENTROPYEOS_TEST_GPU=metal                                    # or cuda; needs the backend installed
JULIA_NUM_THREADS=4                                          # exercises the threading determinism test
```

## Invariants that must not be broken

Everything in `src/core/` is kernel-side and runs on GPUs. It must stay
**allocation-free, exception-free, and generic in the scalar type**. Concretely:

- No `throw`, `error`, `@assert`, or bounds-checked indexing on a hot path.
  `log10`, `sqrt` and `floor(Int, ·)` all throw in Julia where C returns NaN —
  use `safe_log10`, `safe_sqrt`, `trunc_floor` from `core/defs.jl`.
- **Never `@fastmath`.** It folds the NaN and finiteness probes to the wrong
  answer, exactly as `-ffast-math` does in the C++. A test greps for it.
- No bare `Float64` literals in `core/` — they silently widen `Float32`
  arithmetic. Write `T(...)`, or add a per-type function in `core/defs.jl`.
- Kernel-side structs stay `isbits`. Do not replace a NaN sentinel with
  `Union{Nothing,T}`; NaN is load-bearing and flows through arithmetic.

## Traps

- **ρ is `ρ* = κ·ρ`**, not the raw table density, and `U` is re-zeroed so that
  `ρ*(1+U) = ρ(1+ε)` exactly. Passing a raw density is the easiest way to be
  wrong by tens of percent while looking plausible.
- **Measuring allocations**: use a *top-level* helper taking concrete arguments.
  A closure defined inside a `@testset`, or a varargs `f(args...)` splat, itself
  allocates 16 bytes on x86-64 and the measurement then reports the harness.
  See `test/testutil.jl`.
- `--check-bounds=yes` disables every `@inbounds` and makes the allocation gates
  genuinely fail. They are skipped there and run in a dedicated CI job instead.
- **Comparing against the C++**: match its sampling. `con2prim_audit.cpp` uses
  5%-per-side margins on the density box and entropy window and perturbs the
  warm start by 1e-3. Sampling the corners instead, or warm-starting from the
  exact truth, changes the failure rate by more than an order of magnitude in
  either implementation.
- **Analytic EOSs have κ = 1**, so `ρ★ = ρ` there. But `HybridEOS`'s cold part
  is the *generalized* piecewise polytrope (O'Boyle et al. 2020): its `Kᵢ` are
  derived from continuity of `dp/dρ`, so classic Read-et-al. parameter sets give
  a different EOS. Use the paper's own fits, and note that its Table II
  misprints the second crust break (`1.826e6` should be `1.826e8`).
- Never compare iteration counts or `C2PResult` across platforms; a state near a
  decision boundary legitimately takes a different path. Compare values.
- A local coverage run scatters `*.cov` files through `src/`. They are
  gitignored; delete them before grepping.

## Quality gates

`Aqua` and `JET` run in the suite (`test/test_quality.jl`). Aqua catches unused
deps, missing compat bounds and stale exports; JET asserts the kernels are free
of runtime dispatch. If you add a dependency or a kernel, they will tell you
before CI does.

## Style

Idiomatic Julia, not a transliteration. BlueStyle, 4-space indent, 132-column
margin (`.JuliaFormatter.toml`). Unicode identifiers for physics quantities
(`ρ`, `κ`, `σ`, `T̂`, `μ̃`, `cs²`, `U_ρρ`), ASCII for numerical-linear-algebra
names. Comments explain *why*, not what the C++ did.

When translating further C++, **translate faithfully rather than improving**.
The measured-and-kept quirks are load-bearing: the clamped-not-backtracked
Newton step, the cosh-normalized convergence test, the scaled max-norm, the
hand-rolled insertion sort, the guard precedence in the tails.

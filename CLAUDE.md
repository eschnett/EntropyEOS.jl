# Working in this repository

- [Read first](#read-first)
- [What this package is](#what-this-package-is)
- [Scope](#scope)
- [Commands](#commands)
- [Invariants that must not be broken](#invariants-that-must-not-be-broken)
- [Traps](#traps)
- [Quality gates](#quality-gates)
- [Style](#style)
- [Repository](#repository)

## Read first

Read [CODE.md](CODE.md) for the design; this file is the short list of things
that are easy to get wrong. Where new content goes:

| file | holds |
| --- | --- |
| [README.md](README.md) | for users: what the package does, installation, a short example, status, a pointer to `CODE.md` |
| `CLAUDE.md` | for agents: this file; short rules pointing at `CODE.md` |
| [CODE.md](CODE.md) | the current design and the reasons for it, measurements that support a current decision, limitations, open questions (marked **(open)** or **(proposed)**) |
| `PLAN.md` | a concrete plan, when there is one: steps, what each delivers, how one knows it is done |
| [HISTORY.md](HISTORY.md) | how the package got here: decisions with dates, rejected alternatives, superseded measurements, fixed bugs, releases |
| `TODO.md` | the owner's personal list — do not modify it, and move nothing into or out of it |

Every fact lives in exactly one of these files; elsewhere, link to it. When the
design changes, fix `CODE.md` to describe the current state, and record in
`HISTORY.md` what changed, when and why. Keep each file's table of contents
current.

## What this package is

[README.md](README.md) says what it does and [CODE.md](CODE.md) how. The C++
original it translates lives at `/Users/eschnett/src/EntropyEOS` on this
machine.

## Scope

Do not port `repair.cpp`, the audit harnesses, or the `tools/` binaries; table
repair is out of scope ([CODE.md § Scope](CODE.md#scope)).

## Commands

```bash
julia --project=. -e 'using Pkg; Pkg.test()'          # ~25 s
julia --project=docs docs/make.jl                      # build docs locally
```

Optional test gates ([CODE.md § Testing](CODE.md#testing)):

```bash
ENTROPYEOS_TABLE_DIR=/Users/eschnett/src/EntropyEOS/tables   # the five real tables, +25 s
ENTROPYEOS_TEST_GPU=metal                                    # or cuda; needs the backend installed
JULIA_NUM_THREADS=4                                          # exercises the threading determinism test
```

Precision studies are scripts in `study/`
([CODE.md § Precision and devices](CODE.md#precision-and-devices)).

## Invariants that must not be broken

Everything in `src/core/` is kernel-side and runs on GPUs. It must stay
**allocation-free, exception-free, and generic in the scalar type**
([CODE.md § Discipline in `core/`](CODE.md#discipline-in-core)):

- No `throw`, `error`, `@assert`, or bounds-checked indexing on a hot path. Use
  `safe_log10`, `safe_sqrt`, `trunc_floor` from `core/defs.jl`.
- **Never `@fastmath`.** A test greps for it.
- No bare `Float64` literals in `core/`. Write `T(...)`, or add a per-type
  function in `core/defs.jl`.
- Kernel-side structs stay `isbits`. Do not replace a NaN sentinel with
  `Union{Nothing,T}`.

## Traps

- **ρ is `ρ* = κ·ρ`**, not the raw table density. Passing a raw density is the
  easiest way to be wrong by tens of percent while looking plausible
  ([README § Units](README.md#units),
  [CODE.md § Data model and units](CODE.md#data-model-and-units)).
- **Measuring allocations**: use a *top-level* helper taking concrete arguments;
  see `test/testutil.jl` ([CODE.md § Allocation gates](CODE.md#allocation-gates)).
- `--check-bounds=yes` makes the allocation gates genuinely fail. They are
  skipped there and run in a dedicated CI job instead
  ([CODE.md § Allocation gates](CODE.md#allocation-gates)).
- **Comparing against the C++**: match its sampling
  ([CODE.md § Validation against the C++](CODE.md#validation-against-the-c)).
- **Analytic EOSs have κ = 1** ([README § Analytic EOSs](README.md#analytic-eoss)).
  For `HybridEOS`, use the O'Boyle et al. paper's own fits, not classic
  Read-et-al. parameter sets, and mind its Table II misprint
  ([CODE.md § Type design](CODE.md#type-design)).
- Never compare iteration counts or `C2PResult` across platforms; compare values
  ([CODE.md § Testing](CODE.md#testing)).
- A local coverage run scatters `*.cov` files through `src/`. They are
  gitignored; delete them before grepping.

## Quality gates

`Aqua` and `JET` run in the suite (`test/test_quality.jl`). If you add a
dependency or a kernel, they will tell you before CI does
([CODE.md § Static analysis](CODE.md#static-analysis)).

## Style

Idiomatic Julia, not a transliteration. BlueStyle, 4-space indent, 132-column
margin (`.JuliaFormatter.toml`). Unicode identifiers for physics quantities
(`ρ`, `κ`, `σ`, `T̂`, `μ̃`, `cs²`, `U_ρρ`), ASCII for numerical-linear-algebra
names. Comments explain *why*, not what the C++ did.

When translating further C++, **translate faithfully rather than improving**;
the measured-and-kept quirks are listed in
[CODE.md § Deliberate deviations from the C++](CODE.md#deliberate-deviations-from-the-c).

## Repository

- Remote: [github.com/eschnett/EntropyEOS.jl](https://github.com/eschnett/EntropyEOS.jl);
  default branch `main`.
- Do not modify `TODO.md`.

# Precision and GPUs

What each scalar-type configuration costs, measured on all five real tables
and the synthetic ideal gas, and how the device path compares with the host.
All GPU numbers are from an NVIDIA H200 (sm_90) on Perimeter's Symmetry
cluster; the host is AMD EPYC 7543.

## Summary

| configuration | how | status |
| --- | --- | --- |
| **Float64** | `adapt(CuArray, v)` | validated on CUDA: device indistinguishable from host on every table |
| **mixed**: Float32 table, Float64 arithmetic | `adapt(CuArray, narrow(v, Float32))`, called with `Float64` arguments | validated on CUDA; con2prim indistinguishable from Float64 beyond the table's own rounding |
| **Float32** throughout | `narrow(v, Float32)`, `Float32` arguments, `Con2PrimOptions{Float32}()` | usable, with the accuracy limits below; the only option on Metal |

For production on NVIDIA hardware use Float64. If table memory or bandwidth
matters, the mixed configuration halves it at essentially no cost to con2prim.
Pure Float32 runs, never produces a non-finite state through
[`con2prim_safe`](@ref), and on an H200 is 1.3× (evaluate) to 3.6× (cold
con2prim) faster — but its limits are real, and most of them come from storing
the *hydro state* in Float32, not from this package.

## Method

The scripts are in `study/` of the repository and are run by hand:

- `study/float32_accuracy.jl` compares the three configurations against Float64
  ground truth on the host, 10⁶ states per table.
- `study/device_parity.jl` runs the same kernels on a GPU and on the host at one
  configuration, 2²⁰ states per table. It mirrors the C++
  `tests/test_device_cuda.cu` — sampling, passes and gates — so its output can
  be read side by side with the C++ H200 runs.

Inputs are rounded to Float32 first and handed to every configuration as the
same exactly-representable values, so input rounding and computational error
are not confused. Two samplings are used for con2prim: the C++ audit's (5%
margins on the density box and on the pointwise entropy window, a 10%
zero-field coin, log-uniform magnetization up to σ = 10⁴, w uniform in
[0, 6), warm start perturbed by 1e-3), and the same but **uniform in log T**
instead of in entropy. Uniform in entropy puts ~80% of states above 80 MeV on a
real table; uniform in log T is where cold neutron-star matter lives.

The measurement that makes the rest interpretable is the **floor**: the
*Float64* solver applied to the Float32-rounded conservatives. It is what a
hydro code loses merely by storing its state in Float32, before any solver runs
at Float32.

## Float64 on CUDA

The full test suite passes on the H200 with `ENTROPYEOS_TEST_GPU=cuda` and all
five real tables. The parity run passes every gate on every table:

| LS220, 2²⁰ states | host | device |
| --- | --- | --- |
| cold converged | 1025421 | 1025417 |
| cold round trip ρ, p99 / max | 3.37e-9 / 33.2 | 3.39e-9 / 33.5 |
| warm round trip ρ, p99 | 1.60e-16 | 2.37e-16 |
| evaluate `p`, device − host, p99 / max | | 1.98e-13 / 1.56e-12 |
| `con2prim_safe` non-finite outputs | | 0 |

These match the C++ H200 run on LS220 (device − host `p` 1.98e-13 / 1.53e-12).
Cold-result agreement is 99.85%; the differing states sit on the solver's
failure boundary and flip in both directions, which is why the gate is on the
failure *rate*, not per state.

One C++ gate needed a wider floor. The warm start is the host's own exact
answer, so the host's maximum warm error is 2e-16, and the C++ gate holds the
device to 10× that plus 1e-12. On 1–2 states per million the device takes one
Newton step there instead of zero, landing 1e-11 to 5e-11 away. Diagnosed on
the device: its temperature solve lands 1.5e-13 away in log T — within the
T-solve's own step tolerance of 1e-13 — at T ≈ 250 MeV where U ∝ T⁴, which moves
the energy residual to 1.4× the con2prim tolerance. Both answers are correct to
the solver's nested tolerances; the maximum gate now has a floor of 100
con2prim tolerances. The p99 gate is unchanged.

The mixed configuration passes every gate on every table as well, with the same
device–host agreement as pure Float64.

## What Float32 costs

### Evaluation

Relative error against Float64 at p99, inside the audit box. "mixed" isolates
what rounding the *table* to Float32 costs; the Float32 column adds Float32
arithmetic.

| p99 | U | p | h | cs² | T |
| --- | --- | --- | --- | --- | --- |
| mixed, real tables | 1.3–1.8e-6 | 4.6–8.6e-5 | 1.2–2.2e-5 | 0.6–3.5e-3 | 0.08–1.0e-6 |
| Float32, real tables | 1.6e-5 | 1.4–2.0e-4 | 3.7–5.2e-5 | 1.2–6.4e-3 | 2.5–3.5e-6 |

Over the whole extended box the p99 errors are up to 5× larger. No state produces a
non-finite value and the temperature solve never hits its iteration cap.

Two structural causes, both properties of the representation rather than of the
code:

- **Derivatives are differences of large coefficients.** `p`, `cs²` and the
  `U_ρρ`, `U_ρs` the solver uses are spline derivatives, i.e. differences of
  neighbouring coefficients of `log10(ε + shift)` ≈ 19 and of the entropy. In
  Float32 that costs roughly three decimal digits against the value itself —
  about 1000 ulps for `p`, more for `cs²`.
- **The energy is fitted as `log10(ε + shift)`.** Recovering ε subtracts the
  shift, which dominates at the cold end: on the synthetic gas, where ε is 7% of
  the shift at the coldest points, U loses a factor of ~15 there.

The chemical potential `μ̃` is a difference of two such terms. Its Float32 error
is 7e-4 of `1 + U` at p99 (mixed: 9e-5) — in cold matter, where `μ̃` is 0.006
to 0.06, that is 1–10% of `μ̃` itself. Downstream physics that uses `μ̃`
should not run at Float32.

### Storing the state in Float32

This is the dominant limit and is independent of this package. With the Float64
solver on Float32-rounded conservatives:

- uniform in entropy, ρ is recovered to 2.9e-3 at p99 on LS220, and s to 1.8e-3;
  the error grows steeply with rapidity (s p99 1.3e-4 at w < 1, 5.6e-3 at
  5 < w < 6);
- **uniform in log T, 6.8–13% of states have no solution at all** (no bracket).
  The thermal energy of cold matter is below Float32's resolution of τ, so the
  rounded τ lies below the coldest state the table can express.

A Float32 hydro code therefore cannot represent the thermal state of cold matter
in its conserved variables, whatever con2prim does.

### con2prim

Failures per 10⁶ states, audit sampling, with the Float32 defaults:

| | Float64 warm / cold | floor | mixed warm / cold | Float32 warm / cold |
| --- | --- | --- | --- | --- |
| LS220 | 5 / 530 | 12 | 11 / 588 | 46 / 220 |
| SRO LS220 | 8 / 949 | 22 | 7 / 970 | 65 / 385 |
| DD2 | 48 / 390 | 247 | 252 / 578 | 136 / 2689 |
| SFHo | 24 / 333 | 75 | 80 / 389 | 97 / 631 |

The recovered ρ at p99 is 2.9–3.9e-3 for the floor and mixed, and 4.7–5.8e-3
(warm) and 1.6e-2–1.0e-1 (cold) at Float32. On LS220 the mixed solutions sit
within 1.7e-5 (p99) of the floor's: storing the table in Float32 costs con2prim essentially
nothing. The Float32 solver roughly doubles the floor's error on the warm path.

Uniform in log T, the Float32 solver fails 0.08–0.95% warm and 0.05–0.8% cold —
fewer than the floor's 6.8–13%, because its tolerance accepts a nearby state where
Float64 finds none.

[`con2prim_safe`](@ref) at Float32 returns no non-finite state on any table. Its
acceptance bar — the returned conservatives re-solve to the returned primitives
— fails for at most 11 states per million.

### Two defects this study found and fixed

**The cold seed overflowed Float32.** `seed_z_solve` formed `B²·S⊥²` and `q³`,
which reach 1e50 and more on real tables, far beyond `floatmax(Float32)`. The
residual became `Inf`, bisection drove `z` to its lower bound, and every
magnetized Float32 cold start began at `w_max`: 16.6% of cold starts failed on
LS220, independent of field strength (even at B² = 1e-6·ρh). The residual is now
carried as `½B²(S⊥/q)²`, where `S⊥/q` is a velocity. This is algebraically
identical, and the Float64 results are unchanged to the digits the real-table
tests report. LS220 Float32 cold failures: 16.6% → 0.49%.

**The Float32 tolerance was below the noise floor.** The default was the generic
`64 eps`, 7.6e-6. Evaluated in Float32, the energy residual has a p99 noise
floor of 2.0e-5 to 3.8e-5 on the real tables, so that tolerance could not be
met: Newton stalled, the fallback took over, and 0.4–0.7% of warm starts failed. A
sweep over 16–1024 eps showed failures falling monotonically and the density
error flattening from 512 eps, the smallest power of two above the noise floor
on every table; `Con2PrimOptions{Float32}()` now uses 512 eps ≈ 6.1e-5. LS220
Float32 warm failures: 3932 → 46 per million, cold 4889 → 220.

## Float32 on a GPU

The device matches the host at Float32 statistically, not state by state:

| 2²⁰ states | LS220 | SRO LS220 | DD2 | SFHo |
| --- | --- | --- | --- | --- |
| cold ρ p99, host / device | 0.24 / 0.28 | 0.16 / 0.24 | 0.16 / 0.15 | 0.10 / 0.10 |
| evaluate `p`, device − host, p99 | 1.5e-4 | 2.0e-4 | 1.7e-4 | 1.7e-4 |
| `con2prim_safe` non-finite | 0 | 0 | 0 | 0 |

(These cold errors are larger than in the table above because this harness's
cold starts carry no warm information and it samples w up to 6 with B²/p up to
10, as the C++ device test does.)

Device − host differences in `evaluate` are ~1250 ulps at Float32, the same
~1000× amplification of ULP-level libm and FMA differences that gives 2e-13 at
Float64: it is the derivative cancellation above, not a device defect. Two of
the C++ gates therefore do not hold at Float32 and are reported rather than
loosened: the warm-start maximum (up to 0.9 on the device, from a start that is
the host's exact answer but a ~1e-4-noisy residual to the device); and on DD2,
6% more cold failures on the device than on the host, just over the gate's
5% + 10.

## Throughput

H200, 2²⁰ states, millions of states per second, LS220. The node is shared, so
treat these as ±20%.

| | evaluate | con2prim cold | con2prim warm |
| --- | --- | --- | --- |
| Float64 | 318 | 1.1 | 324 |
| mixed | 370 | 0.9 | 335 |
| Float32 | 411 | 4.0 | 841 |
| C++, Float64 (repaired LS220) | 381 | 0.6 | 529 |

The H200's FP64 throughput is unusually high; on consumer GPUs the Float32
advantage is much larger.

## Reproducing

```bash
julia --project=study -e 'using Pkg; Pkg.instantiate()'
julia -t 64 --project=study study/float32_accuracy.jl --n=1000000 --tables=/path/to/tables
julia -t 16 --project=study study/device_parity.jl --backend=cuda --type=Float64 --tables=/path/to/tables
julia -t 16 --project=study study/device_parity.jl --backend=cuda --type=Float64 --storage=Float32 --tables=/path/to/tables
julia -t 16 --project=study study/device_parity.jl --backend=cuda --type=Float32 --tables=/path/to/tables
```

`--only=NAME` restricts to tables whose name contains `NAME`; `--backend=metal`
runs the parity script on Apple GPUs (Float32 only).

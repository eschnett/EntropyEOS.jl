# History

How the package got where it is. The current design is in [CODE.md](CODE.md);
the sections here follow its headings. Dates are those of the commits that
recorded each item in the documentation.

- [Type design](#type-design)
  - [The EOS interface](#the-eos-interface)
- [State policy flags](#state-policy-flags)
- [Discipline in `core/`](#discipline-in-core)
  - [The one stack buffer](#the-one-stack-buffer)
- [Testing](#testing)
  - [Static analysis](#static-analysis)
- [Precision and devices](#precision-and-devices)

## Type design

### The EOS interface

2026-09-24. Introducing `AbstractEOS` changed no arithmetic: a hash over 3,000
states of every entry point's output was bitwise unchanged.

## State policy flags

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
[two-level projection](CODE.md#two-level-projection-proposed), prompted by
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

## Discipline in `core/`

### The one stack buffer

2026-09-23. The `MVector{33,T}` buffer in `bracket_scan` was the design's
biggest open risk; it is resolved by the kernel compiling and running on Metal.

## Testing

### Static analysis

2026-09-23. `Aqua` is how the unused `PrecompileTools` entry was eventually
found.

## Precision and devices

2026-09-24. Products of two conserved quantities overflow Float32;
`seed_z_solve` did this until the study found it.

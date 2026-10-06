# History

How the package got where it is. The current design is in [CODE.md](CODE.md);
the sections here follow its headings. Dates are those of the commits that
recorded each item in the documentation.

- [Type design](#type-design)
  - [The EOS interface](#the-eos-interface)
- [Discipline in `core/`](#discipline-in-core)
  - [The one stack buffer](#the-one-stack-buffer)
- [Testing](#testing)
  - [Static analysis](#static-analysis)
- [Precision and devices](#precision-and-devices)

## Type design

### The EOS interface

2026-09-24. Introducing `AbstractEOS` changed no arithmetic: a hash over 3,000
states of every entry point's output was bitwise unchanged.

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

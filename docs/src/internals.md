# Internals

Flag bits, outcome enums, domain-safe arithmetic, the scalar-type-parametric
tolerances, and the rules that move a view onto a device.

## Outcome enums

These are `EnumX` modules, so `Status.ok` names a member and `Status.T` the
type. They are listed explicitly because `@autodocs` does not collect module
bindings.

```@docs
EntropyEOS.Status
EntropyEOS.C2PResult
```

## Everything else

```@autodocs
Modules = [EntropyEOS]
Pages = ["core/defs.jl", "core/adapt.jl"]
```

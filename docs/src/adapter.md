# The adapter

Turning a tabulated `F(ρ, T, Yₑ)` into the single thermodynamic potential
`U(ρ, s, Yₑ)`, and evaluating it.

Note the density convention: the adapter's `ρ` is `ρ* = κ·ρ`, in κ-rescaled
g/cm³, and its energy is re-zeroed so that `ρ*(1 + U) = ρ(1 + ε)` exactly.

```@autodocs
Modules = [EntropyEOS]
Pages = ["core/bspline_eval.jl", "host/bspline_fit.jl", "core/adapter_eval.jl",
         "host/adapter_build.jl"]
```

# The solver

Converting between primitive and conserved variables, using entropy and
rapidity as the iteration variables. Above them, a projection onto the valid
range that never fails, and a replaceable physics policy on top of it.

```@autodocs
Modules = [EntropyEOS]
Pages = ["core/prim2con.jl", "core/con2prim.jl", "core/faces.jl", "core/state_policy.jl"]
```

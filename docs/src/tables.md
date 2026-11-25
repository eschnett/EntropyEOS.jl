# Tables

Loading a tabulated EOS, checking it, and generating an analytic one.

Table *repair* is not part of this package: it is an offline activity performed
once before a simulation campaign, and the run-time path contains no repair
logic. [`check_table`](@ref) is what detects a table that has not been repaired.

```@autodocs
Modules = [EntropyEOS]
Pages = ["host/table.jl", "host/io_stellarcollapse.jl", "host/check.jl",
         "host/synthetic.jl", "host/units.jl"]
```

## Cold β-equilibrium

Initial data for a cold neutron star: the states of least Gibbs energy at each
pressure among those with `μ_ν = 0` on the table's coldest row, with the phase
transitions between them explicit.

```@autodocs
Modules = [EntropyEOS]
Pages = ["host/beta_equilibrium.jl"]
```

## Composition

The chemical potentials, mass fractions and mean heavy nucleus of a table, for
neutrino rates, evaluated in a kernel. Build a [`CompositionTable`](@ref) from
the raw table and the EOS built from it, once; then call
[`composition`](@ref) next to `evaluate`, at the temperature the recovery
landed on:

```julia
ct = CompositionTable(table, eos)
matches(ct, eos) || error("composition and EOS are from different tables")
cv = CompositionView(ct)              # adapt(CuArray, cv) for a device
pt = evaluate(v, ρ★, s, yₑ, NaN)
c = composition(cv, ρ★, pt.T_MeV, yₑ)
η_ν = c.μ_ν / pt.T_MeV
η̂_nucleons = (c.μ̂ - q_np(cv)) / pt.T_MeV   # without rest masses
```

```@autodocs
Modules = [EntropyEOS]
Pages = ["core/composition_eval.jl", "host/composition_build.jl"]
```

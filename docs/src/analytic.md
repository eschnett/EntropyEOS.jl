# Analytic EOSs

The solver and the policy layer are written against an interface, not
against the table. Two closed-form equations of state implement it next to
[`EOSTableView`](@ref): an ideal gas and a hybrid EOS. These are what the standard
relativistic test problems need — shock tubes at `Γ = 2`, TOV stars on a
polytrope — and a tabulated nuclear EOS cannot host them.

```julia
using EntropyEOS

eos = IdealGasEOS(; Γ=2.0, K_ref=100.0, s_ref=5.0, s_window=(1.0, 20.0),
                  ρ_bounds=(1e-12, 1e-2), yₑ_bounds=(0.0, 1.0))
s = polytropic_entropy(eos, 100.0)          # the polytrope p = 100 ρ²
pt = evaluate(eos, 1.28e-3, s, 0.5, NaN)

pol = default_policy(eos, 1e-10)
c = prim2con(eos, 1.28e-3, s, 0.5, 0.8, 0.0, 0.0, NaN)
out = con2prim_safe(eos, Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²),
                    Con2PrimOptions(), pol)
```

Both are `isbits`, so they go into a GPU kernel as they are: they need no
`Adapt` rule. Use [`narrow`](@ref) for a `Float32` copy. Both have `κ = 1`, so
`ρ★ = ρ`, and `Yₑ` enters only the bounds: `μ̃ = 0`, and `T_MeV` and
`u_solved` are `NaN`. The units are the caller's, in any consistent system with
`U = ε/c²`.

## Choosing the entropy window

`s` is a label for the adiabats, fixed up to an additive constant by
`(K_ref, s_ref)`. For the ideal gas with `p = ρkT/m`, it is the entropy per
particle in units of `k_B`. The solver's bracket scan steps *relatively* in
`s`, and `s_min` must be positive, so choose `s_ref` so that the window is of
order 1 to 10, as for an entropy in `k_B` per baryon.

A barotropic EOS cannot be offered at all: the solver iterates on entropy, and
its Jacobian is singular when `U_s ≡ 0`. A polytrope is therefore an ideal gas
held at one entropy, [`polytropic_entropy`](@ref).

## Why a *generalized* piecewise polytrope

The hybrid EOS adds a thermal ideal gas to a cold piecewise polytrope. The
classic piecewise polytrope (Read et al. 2009) makes `p` and `ε` continuous at
each break, but not `dp/dρ`. `cs²` then jumps by the ratio of neighbouring
exponents, often by tens of percent. `con2prim`'s Jacobian contains `cs²`, so it
would jump too. In simulations the jumps cause reflections and limit
convergence at the breaks.

The generalized piecewise polytrope of O'Boyle, Markakis, Stergioulas & Read
(2020, [Phys. Rev. D 102, 083027](https://doi.org/10.1103/PhysRevD.102.083027))
adds a constant to each piece, `p = Kᵢρ^Γᵢ + Λᵢ`, and uses it to make `dp/dρ`
continuous as well. The number of free parameters is unchanged: `K₀`, the
exponents and the breaks. `U` is then C² in `ρ`, like the table path.

The `Kᵢ` are derived differently from the classic form, so **parameter sets
fitted for classic piecewise polytropes do not carry over**. That paper
tabulates its own fits: an SLy crust (its Table II) and about 25 nuclear-matter
cores matched to that crust (its Table III). Building SLY4:

```julia
crust_ρ = (6.285e5, 1.826e8, 3.350e11, 5.317e11)   # see the note below
crust_Γ = (1.611, 1.440, 1.269, -1.841, 1.382)
K₀ = 5.214e-9                                      # cgs, with c = 1

# Eq. B1: where the core (K₁, Γ₁) meets the crust with continuous dp/dρ.
K_crust = EntropyEOS.gpp_constants(crust_ρ, K₀, crust_Γ).K[end]
K₁, Γ₁ = 10^-31.350, 3.045
ρ₀ = (K₁ * Γ₁ / (K_crust * crust_Γ[end]))^(1 / (crust_Γ[end] - Γ₁))

sly = HybridEOS(; ρ_breaks=(crust_ρ..., ρ₀, 10^14.87, 10^14.99), K₀,
                Γs=(crust_Γ..., Γ₁, 2.884, 2.773),
                Γ_th=1.75, K_th_ref=2.4e-12, s_ref=20.0, s_window=(1.0, 20.0),
                ρ_bounds=(1e6, 2e15), yₑ_bounds=(0.0, 1.0))
```

!!! note "A misprint in Table II"
    Table II prints the second crust break as `1.826e6 g/cm³`. That is
    inconsistent with the table's own `K`, `Λ` and `a` columns, and makes the
    pressure negative. `1.826e8` reproduces all three columns, to within the
    rounding of the printed exponents. The test suite checks this.

## Low entropy on the hybrid

At the cold end of the window `U_th ≪ U_cold`, and the energy then determines
the entropy only weakly. `con2prim` still recovers the density, the velocity and
the pressure accurately, but `s` only to a relative accuracy of about
`ϵ·U/(s·U_s)`. This is physics, not a solver limitation, and a tabulated EOS
behaves the same way at low temperature. `s_min` sets how far into this regime
the EOS reaches.

## Reference

```@autodocs
Modules = [EntropyEOS]
Pages = ["core/eos_interface.jl", "core/analytic_eos.jl", "host/analytic_eos.jl"]
```

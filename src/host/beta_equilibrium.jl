# Cold, neutrinoless β-equilibrium: the Yₑ and entropy of a cold neutron star.
#
# Host-side, for initial data. The criterion is read from the table's own
# `munu` field rather than from the fitted potential's μ̃: at the cold edge μ̃
# carries the fit's error magnified by 1/σ_T, several MeV and more, which moves
# Yₑ by percent. The state found is then placed on the fitted EOS's cold edge,
# so that it is exactly a state the evolution can represent. CODE.md
# ("β-equilibrium") has the measurements behind both choices.

"""
    BetaEquilibrium{T}

The result of [`beta_equilibrium`](@ref).

  * `yₑ`: the electron fraction at which `μ_ν = 0`, clamped to the table's range.
  * `s`: the entropy of the coldest tabulated state at `(ρ★, yₑ)`, i.e.
    `srange(eos, ρ★, yₑ).s_min`.
  * `point`: the EOS evaluated there, giving `p`, `U`, `h`, `cs²`, ….
  * `flags`: `point.flags`, plus `FLAG_CLAMP_YE` if the root lay outside the
    table's `Yₑ` range.
  * `nroots`: how many interior roots `μ_ν = 0` with `μ_ν` rising through zero
    were found. More than one happens on the Hempel tables at subnuclear
    density (shell structure near `Yₑ ≈ 0.5`); the one returned minimizes the
    potential `∫ μ_ν dYₑ`. It is 0 when the result is clamped.
"""
struct BetaEquilibrium{T<:Real}
    yₑ::T
    s::T
    point::EOSPoint{T}
    flags::UInt32
    nroots::Int
end

"""
    beta_equilibrium(table::RawTable, eos::EOSTable, ρ★) -> BetaEquilibrium

The cold, neutrinoless β-equilibrium state at rescaled density `ρ★ = κρ`: the
`Yₑ` at which the table's neutrino chemical potential `munu` vanishes on its
coldest temperature row, and the entropy of the fitted EOS's cold edge there.

`table` must be the table `eos` was built from, and must carry `munu`.
`ArgumentError` is thrown otherwise.

`munu` is interpolated linearly in `log10 ρ` and `Yₑ`. Where it has several
upward zero crossings in `Yₑ` the global minimum of `Φ(Yₑ) = ∫ munu dYₑ` is
chosen, which is the equilibrium whatever rest-mass convention the table family
uses for its chemical potentials. Where the root lies outside the table's `Yₑ`
range (O'Connor–Ott LS220 near `10¹³–10¹⁴` g/cm³) `Yₑ` is clamped and
`FLAG_CLAMP_YE` is set. A density outside the table is looked up at the nearest
edge, and `evaluate`'s density flags pass through.

"Cold" means the table's lowest temperature, not `T = 0`.
"""
function beta_equilibrium(table::RawTable, eos::EOSTable, ρ★::Real)
    v = eos.view
    check_beta_inputs(table, v)
    munu = field(table, "munu")
    logρ, Yₑ = table.logρ, table.Yₑ
    n = length(Yₑ)
    R = promote_type(eltype(table), eltype(v), typeof(float(ρ★)))

    # The density cell on the raw axis, in the table's own (unrescaled) density.
    x = clamp(R(log10(ρ★)) - R(log10(v.κ)), R(logρ[1]), R(logρ[end]))
    i = clamp(searchsortedlast(logρ, x), 1, length(logρ) - 1)
    w = (x - R(logρ[i])) / (R(logρ[i + 1]) - R(logρ[i]))
    g(k) = (1 - w) * R(munu[i, 1, k]) + w * R(munu[i + 1, 1, k])
    for k in 1:n
        isfinite(g(k)) || throw(ArgumentError("beta_equilibrium: non-finite munu at log10(ρ) = $x"))
    end

    # Candidates are the minima of Φ: every upward crossing, and an endpoint at
    # which Φ points outward. Φ is accumulated cell by cell, exact for the
    # piecewise-linear g.
    best_y = R(NaN)
    best_Φ = R(Inf)
    clamped = false
    nroots = 0
    if g(1) > 0
        best_y, best_Φ, clamped = R(Yₑ[1]), zero(R), true
    end
    Φ = zero(R)
    for k in 1:(n - 1)
        y0, y1 = R(Yₑ[k]), R(Yₑ[k + 1])
        g0, g1 = g(k), g(k + 1)
        if g0 <= 0 < g1
            nroots += 1
            y = y0 + (y1 - y0) * (-g0 / (g1 - g0))
            Φy = Φ + (y - y0) * g0 / 2   # g(y) = 0
            if Φy < best_Φ
                best_y, best_Φ, clamped = y, Φy, false
            end
        end
        Φ += (y1 - y0) * (g0 + g1) / 2
    end
    if g(n) <= 0 && Φ < best_Φ
        best_y, best_Φ, clamped = R(Yₑ[n]), Φ, g(n) < 0
    end

    yₑ = best_y
    s = srange(v, R(ρ★), yₑ).s_min
    pt = evaluate(v, R(ρ★), s, yₑ, R(v.u_lo))
    flags = pt.flags | (clamped ? FLAG_CLAMP_YE : UInt32(0))
    return BetaEquilibrium{R}(yₑ, s, pt, flags, nroots)
end

# The table must carry `munu` and be the one the EOS was fitted to: a cropped
# or different table would silently read the wrong row.
function check_beta_inputs(table::RawTable, v::EOSTableView)
    has_field(table, "munu") || throw(ArgumentError("beta_equilibrium: table has no 'munu' field"))
    length(table.logρ) >= 2 && length(table.Yₑ) >= 2 ||
        throw(ArgumentError("beta_equilibrium: table needs at least two ρ and two Yₑ points"))
    lκ = log10(Float64(v.κ))
    same(a, b) = isapprox(Float64(a), Float64(b); atol=1.0e-8 * max(1.0, abs(Float64(b))))
    ok = same(table.logρ[1] + lκ, v.x_lo) && same(table.logρ[end] + lκ, v.x_hi) &&
         same(table.Yₑ[1], v.y_lo) && same(table.Yₑ[end], v.y_hi) && same(table.logT[1], v.u_lo)
    ok || throw(ArgumentError("beta_equilibrium: the table's axes do not match the EOS's box; pass the table the EOS was built from"))
    return nothing
end

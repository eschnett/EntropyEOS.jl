# Building the composition view from a raw table.
#
# The tables agree on what they tabulate but not on its conventions, and those
# are resolved here, once, so that a kernel sees one convention for every
# table:
#
#   * μ̂. Whether `muhat` includes the neutron–proton mass difference Q is read
#     off the table's own free nucleons where they are a hot, dilute ideal gas
#     (`muhat_offset`), not off its `munu`: O'Connor–Ott LS220's `munu`
#     subtracts a further Q above 10⁸ g/cm³. All four real tables' `muhat`
#     includes Q; a table's that lacks it gets Q added, so that μ̂ always
#     includes Q and μ_ν = μ_e − μ̂.
#   * The light clusters. The Hempel tables carry d, t, ³He and ⁴Li, SRO one
#     `Xl`; without them the four classic species miss up to 30% of the mass at
#     log ρ 11–13, T 3–10 MeV, Yₑ 0.1–0.3 on DD2. They are summed into X_l.
#   * Ā and Z̄ where there are no heavy nuclei. The tables fill them with
#     Ā = Z̄ = 1 (Hempel), 4 and 1.8 (O'Connor–Ott LS220), or 1 and Yₑ (SRO).
#     The kernel weights by X_h, so the fills never enter; they are stored as
#     1 and 0 only so that the array holds no arbitrary numbers.

# The datasets summed into X_l, by family: Hempel's light clusters, SRO's one.
const LIGHT_CLUSTER_FIELDS = ("Xd", "Xt", "X3he", "X4li", "Xl")

# How far the measured munu offset may be from 0 or −Q, in MeV.
const MUNU_OFFSET_TOL_MEV = 0.01

# The ideal-gas test of `muhat`: the nodes it reads, how many it needs, how
# alike their residuals must be for the gas to count as ideal, and how far
# their median may be from 0 or −Q, in MeV. On the real tables the residuals'
# 5–95 % range is 0.005–0.03 MeV (O'Connor–Ott LS220) or narrower.
const IDEAL_GAS_T_MEV = (2.0, 10.0)
const IDEAL_GAS_LOGRHO_MAX = 10.0
const IDEAL_GAS_X_MIN = 0.05
const IDEAL_GAS_MIN_NODES = 100
const IDEAL_GAS_IQR_MEV = 0.05
const IDEAL_GAS_TOL_MEV = 0.1

"""
    muhat_offset(table::RawTable, who; munu_offset = nothing) -> Float64

The constant `c` in `μ_ν = mu_e − muhat + c`, in MeV: 0 if the table's `muhat`
includes the neutron–proton mass difference `Q`, `−Q` if it does not. `who`
prefixes the error messages.

Where the free nucleons are a hot, dilute ideal gas (`T` 2–10 MeV,
`log10 ρ ≤ 10`, `Xn` and `Xp` above 0.05), a `muhat` that includes `Q` is
`Q + T ln[(Xn/Xp)(m_p/m_n)^{3/2}]`. The median of `muhat` minus that, over those
nodes, must be 0 or `−Q` to within 0.1 MeV, and is rounded to it; otherwise
this throws. The test reads only the table's own composition, not `munu`, whose
convention can differ from its `muhat`'s (O'Connor–Ott LS220's, above
10⁸ g/cm³). A table without such nodes, or whose residuals spread by more than
0.05 MeV (the synthetic tables, which are not physics), falls back to the
median of `munu − (mu_e − muhat)` on the coldest row, which must then be 0 or
`−Q` to within 10 keV.
"""
function muhat_offset(table::RawTable, who::AbstractString; munu_offset=nothing)
    munu_offset === nothing || return Float64(munu_offset)
    m = ideal_gas_muhat_residual(table)
    if m !== nothing
        abs(m) <= IDEAL_GAS_TOL_MEV && return 0.0
        abs(m + Q_NP_MEV) <= IDEAL_GAS_TOL_MEV && return -Q_NP_MEV
        throw(ArgumentError("$who: in the ideal-gas limit muhat differs from Q + T ln[(Xn/Xp)(m_p/m_n)^{3/2}] " *
                            "by $m MeV, neither 0 nor −Q = $(-Q_NP_MEV) MeV; pass munu_offset (MeV) explicitly"))
    end
    has_field(table, "munu") ||
        throw(ArgumentError("$who: the table has no ideal-gas region to read muhat's convention from, and no " *
                            "'munu'; pass munu_offset (MeV)"))
    mue, muh, munu = field(table, "mu_e"), field(table, "muhat"), field(table, "munu")
    d = [Float64(munu[i, 1, k]) - (Float64(mue[i, 1, k]) - Float64(muh[i, 1, k])) for i in axes(munu, 1),
         k in axes(munu, 3)]
    all(isfinite, d) || throw(ArgumentError("$who: non-finite mu_e, muhat or munu on the coldest row"))
    c = median_of(vec(d))
    abs(c) <= MUNU_OFFSET_TOL_MEV || abs(c + Q_NP_MEV) <= MUNU_OFFSET_TOL_MEV ||
        throw(ArgumentError("$who: munu − (mu_e − muhat) has median $c MeV on the coldest row, " *
                            "neither 0 nor −Q = $(-Q_NP_MEV) MeV; pass munu_offset (MeV) explicitly"))
    return c
end

# The median of `muhat − Q − T ln[(Xn/Xp)(m_p/m_n)^{3/2}]` over the ideal-gas
# nodes, or `nothing` where the test does not apply.
function ideal_gas_muhat_residual(table::RawTable)
    all(f -> has_field(table, f), ("muhat", "Xn", "Xp")) || return nothing
    muh, Xn, Xp = field(table, "muhat"), field(table, "Xn"), field(table, "Xp")
    m_n = M_NEUTRON_G * C_LIGHT_CM_S^2 / MEV_TO_ERG
    lr = 1.5 * log1p(-Q_NP_MEV / m_n)          # ln (m_p/m_n)^{3/2}
    r = Float64[]
    for k in axes(muh, 3), j in axes(muh, 2), i in axes(muh, 1)
        T = exp10(Float64(table.logT[j]))
        IDEAL_GAS_T_MEV[1] <= T <= IDEAL_GAS_T_MEV[2] && Float64(table.logρ[i]) <= IDEAL_GAS_LOGRHO_MAX || continue
        xn, xp = Float64(Xn[i, j, k]), Float64(Xp[i, j, k])
        xn > IDEAL_GAS_X_MIN && xp > IDEAL_GAS_X_MIN || continue
        v = Float64(muh[i, j, k]) - Q_NP_MEV - T * (log(xn / xp) + lr)
        isfinite(v) && push!(r, v)
    end
    length(r) >= IDEAL_GAS_MIN_NODES || return nothing
    s = sort(r)
    q(f) = s[clamp(round(Int, f * (length(s) - 1)) + 1, 1, length(s))]
    q(0.75) - q(0.25) <= IDEAL_GAS_IQR_MEV || return nothing
    return median_of(s)
end

"""
    CompositionTable{T}

A built composition: the kernel-side [`CompositionView`](@ref) plus what the
build decided. `munu_offset` is the constant `c` in `μ_ν = mu_e − muhat + c`
(MeV), and `light_fields` the datasets summed into `X_l`.
"""
struct CompositionTable{T<:AbstractFloat}
    view::CompositionView{T,Array{T,4}}
    munu_offset::Float64
    light_fields::Vector{String}
end

"""
    CompositionView(ct::CompositionTable)

The kernel-side view of a built composition. Use `Adapt.adapt` to move it to a
device.
"""
CompositionView(ct::CompositionTable) = ct.view

fingerprint(ct::CompositionTable) = fingerprint(ct.view)
q_np(ct::CompositionTable) = q_np(ct.view)
matches(cv::CompositionView, eos::EOSTable) = matches(cv, eos.view)
matches(ct::CompositionTable, eos) = matches(ct.view, eos)
composition(ct::CompositionTable, ρ★::T, T_MeV::T, yₑ::T) where {T<:Real} = composition(ct.view, ρ★, T_MeV, yₑ)

"""
    CompositionTable(raw::RawTable, eos::EOSTable; munu_offset = nothing) -> CompositionTable

The composition of the table `raw`, for evaluation with [`composition`](@ref)
in a kernel next to `eos`, which must have been built from the same file
([`fingerprint`](@ref)); otherwise this throws `ArgumentError`. Opt-in: a code
that does not need the composition carries none of its arrays (nine fields per
node, 180 MB in Float64 on DD2).

Reads `mu_e`, `muhat`, `Xn`, `Xp`, `Xa`, `Xh`, `Abar`, `Zbar`, and those of
`Xd`, `Xt`, `X3he`, `X4li` and `Xl` the table has; every one must be finite.

`munu_offset` is the constant `c` in `μ_ν = mu_e − muhat + c`: 0 if the table's
`muhat` includes `Q` ([`q_np`](@ref)), `−Q` if it does not. By default it is
read off the table's free nucleons in their ideal-gas limit, as
[`cold_beta_slice`](@ref) does; a table without that limit falls back to its
`munu`, read only then. Either must give 0 or `−Q`, or the build throws; pass
it explicitly for a table with another convention. Stored μ̂ is
`muhat − munu_offset`, so that it includes `Q` on every table. All four real
tables' `muhat` already does.

Mass fractions are clamped to `[0, 1]` at the nodes, and `Ā`, `Z̄` to `≥ 0`;
where `Xh = 0` they are stored as 1 and 0.
"""
function CompositionTable(raw::RawTable{T}, eos::EOSTable; munu_offset=nothing) where {T}
    v = eos.view
    fp = table_fingerprint(raw)
    fp == v.fingerprint ||
        throw(ArgumentError("CompositionTable: the table is not the one the EOS was built from " *
                            "(fingerprint $(repr(fp)) vs $(repr(v.fingerprint)))"))
    nx, nu, ny = size(raw)
    min(nx, nu, ny) >= 2 || throw(ArgumentError("CompositionTable: every axis needs at least two points"))
    tol = BuildOptions().uniform_tol
    check_uniform_axis(raw.logρ, "logrho", tol)
    check_uniform_axis(raw.logT, "logtemp", tol)
    check_uniform_axis(raw.Yₑ, "ye", tol)
    check_composition_box(raw, v)

    need = ("mu_e", "muhat", "Xn", "Xp", "Xa", "Xh", "Abar", "Zbar")
    for f in need
        check_composition_field(raw, f)
    end
    light = String[f for f in LIGHT_CLUSTER_FIELDS if has_field(raw, f)]
    foreach(f -> check_composition_field(raw, f), light)

    c₀ = muhat_offset(raw, "CompositionTable"; munu_offset)
    mue, muh = field(raw, "mu_e"), field(raw, "muhat")

    Xn, Xp, Xa, Xh = field(raw, "Xn"), field(raw, "Xp"), field(raw, "Xa"), field(raw, "Xh")
    Abar, Zbar = field(raw, "Abar"), field(raw, "Zbar")
    lights = [field(raw, f) for f in light]
    c = Array{T,4}(undef, COMP_NFIELDS, nx, nu, ny)
    unit(x) = clamp(x, zero(T), one(T))
    for k in 1:ny, j in 1:nu, i in 1:nx
        xh = unit(Xh[i, j, k])
        xl = zero(T)
        for L in lights
            xl += L[i, j, k]
        end
        c[COMP_μ_E, i, j, k] = mue[i, j, k]
        c[COMP_μ̂, i, j, k] = muh[i, j, k] - T(c₀)
        c[COMP_X_N, i, j, k] = unit(Xn[i, j, k])
        c[COMP_X_P, i, j, k] = unit(Xp[i, j, k])
        c[COMP_X_α, i, j, k] = unit(Xa[i, j, k])
        c[COMP_X_H, i, j, k] = xh
        c[COMP_X_L, i, j, k] = unit(xl)
        c[COMP_Ā, i, j, k] = xh > 0 ? max(Abar[i, j, k], zero(T)) : one(T)
        c[COMP_Z̄, i, j, k] = xh > 0 ? max(Zbar[i, j, k], zero(T)) : zero(T)
    end

    # The density origin is relabelled by log10 κ exactly as `build_eos` does
    # it, so that a node of the table is a node of the EOS's spline too.
    κ = T(v.κ)
    x0 = raw.logρ[1] + log10(κ)
    hx = (raw.logρ[end] - raw.logρ[1]) / (nx - 1)
    hu = (raw.logT[end] - raw.logT[1]) / (nu - 1)
    hy = (raw.Yₑ[end] - raw.Yₑ[1]) / (ny - 1)
    view = CompositionView(c, x0, hx, raw.logT[1], hu, raw.Yₑ[1], hy, κ, T(Q_NP_MEV), fp)
    return CompositionTable{T}(view, c₀, light)
end

function check_composition_field(t::RawTable, name::AbstractString)
    has_field(t, name) || throw(ArgumentError("CompositionTable: table has no '$name' field"))
    all(isfinite, field(t, name)) || throw(ArgumentError("CompositionTable: field '$name' has a non-finite value"))
    return nothing
end

# The fingerprint already ties the table to the EOS; this guards the box
# against a coincidence of hashes, and allows the EOS's box to end below the
# table's top T row, as `trim_t_hi_rows` makes it.
function check_composition_box(t::RawTable, v::EOSTableView)
    lκ = log10(Float64(v.κ))
    same(a, b) = isapprox(Float64(a), Float64(b); atol=1.0e-8 * max(1.0, abs(Float64(b))))
    ok = same(t.logρ[1] + lκ, v.x_lo) && same(t.logρ[end] + lκ, v.x_hi) && same(t.Yₑ[1], v.y_lo) &&
         same(t.Yₑ[end], v.y_hi) && same(t.logT[1], v.u_lo) && Float64(v.u_hi) <= Float64(t.logT[end]) + 1.0e-8
    ok || throw(ArgumentError("CompositionTable: the table's axes do not match the EOS's box"))
    return nothing
end

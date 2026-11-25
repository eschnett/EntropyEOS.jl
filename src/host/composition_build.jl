# Building the composition view from a raw table.
#
# The tables agree on what they tabulate but not on its conventions, and those
# are resolved here, once, so that a kernel sees one convention for every
# table:
#
#   * μ̂. O'Connor–Ott LS220's `muhat` lacks the neutron–proton mass difference
#     Q, the Hempel and SRO tables' includes it. The offset that turns
#     `mu_e − muhat` into the table's `munu` is measured on the coldest row, as
#     `cold_beta_slice` does (0, or −Q on O'Connor–Ott LS220), and taken out of
#     μ̂, so that μ̂ always includes Q and μ_ν = μ_e − μ̂. Measured on all four
#     real tables (2026-11-25), the offset is the same at every temperature.
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

Reads `mu_e`, `muhat`, `Xn`, `Xp`, `Xa`, `Xh`, `Abar`, `Zbar`, those of
`Xd`, `Xt`, `X3he`, `X4li` and `Xl` the table has, and `munu` unless
`munu_offset` is given. Every one must be finite.

`munu_offset` is the median of `munu − (mu_e − muhat)` on the coldest row, as in
[`cold_beta_slice`](@ref) but without its energy check. It must be 0 or
`−Q` ([`q_np`](@ref)) to within 10 keV, or the build throws; pass it explicitly
for a table with another convention. Stored μ̂ is `muhat − munu_offset`, so
that it includes `Q` on every table.

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
    munu_offset === nothing && check_composition_field(raw, "munu")
    light = String[f for f in LIGHT_CLUSTER_FIELDS if has_field(raw, f)]
    foreach(f -> check_composition_field(raw, f), light)

    mue, muh = field(raw, "mu_e"), field(raw, "muhat")
    c₀ = if munu_offset === nothing
        munu = field(raw, "munu")
        d = [Float64(munu[i, 1, k]) - (Float64(mue[i, 1, k]) - Float64(muh[i, 1, k])) for i in 1:nx, k in 1:ny]
        m = median_of(vec(d))
        abs(m) <= MUNU_OFFSET_TOL_MEV || abs(m + Q_NP_MEV) <= MUNU_OFFSET_TOL_MEV ||
            throw(ArgumentError("CompositionTable: munu − (mu_e − muhat) has median $m MeV on the coldest row, " *
                                "neither 0 nor −Q = $(-Q_NP_MEV) MeV; pass munu_offset (MeV) explicitly"))
        m
    else
        Float64(munu_offset)
    end

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

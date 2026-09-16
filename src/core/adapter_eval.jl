# The F(ρ,T,Yₑ) → U(ρ,s,Yₑ) adapter: evaluation and the designed domain
# extensions.
#
# Translated from `entropy_eos/core/adapter_eval.hpp`.
#
# The table supplies two fitted fields on a (x, u, y) grid, where
# x = log10(ρ* [g/cc]), u = log10(T [MeV]) and y = Yₑ:
#
#   σ̂  the entropy, in k_B per baryon
#   L̂  log10(ε_cgs + energy_shift_cgs)
#
# `evaluate` delivers U -- the specific internal energy as a function of
# *entropy* rather than temperature -- plus the ρ and s derivatives the
# con2prim Newton consumes. Internally it solves σ(x, u, y) = s for u, then
# applies the implicit-function-theorem chain rule.
#
# Units: ρ is ρ* in κ-rescaled g/cc, s is k_B per baryon, U = ε/c² is
# dimensionless, p is returned as p/c² in g/cc, T̂ = U_s is dimensionless, and
# T_MeV is in MeV.

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

"""
    EOSPoint

One evaluation of the EOS.

The first five fields are what the con2prim Newton consumes; the next four are
derived. Note the two distinct temperatures: `T_MeV` is the table temperature
the solve landed on, which is what downstream tabulated physics indexes by,
while `T̂ = U_s` is the thermodynamic conjugate the solver must use internally.
They agree exactly only if the underlying table is thermodynamically
consistent, and conflating them is a silent physics bug.

`u_solved` is `log10(T_MeV)`, threaded back out as the warm start for the next
call -- there is no hidden mutable state anywhere, which is what makes
evaluation re-entrant and safe to call in parallel across grid points.
"""
struct EOSPoint{T<:AbstractFloat}
    U::T
    U_ρ::T
    U_s::T
    U_ρρ::T
    U_ρs::T
    T̂::T
    p::T
    h::T
    cs²::T
    T_MeV::T
    μ̃::T
    u_solved::T
    iters::Int32
    flags::UInt32
end

"""
    SRange

The pointwise physical entropy window `[s(T_min), s(T_max)]` at fixed `(ρ*, Yₑ)`.

The physical domain in `(ρ, s)` is *not* rectangular: `s_max` at the highest
density is far below `s_max` at the lowest. Every entropy clamp must therefore
be taken at the already-clamped `(ρ, Yₑ)`.
"""
struct SRange{T<:AbstractFloat}
    s_min::T
    s_max::T
end

"""
    UHighTailInfo

Diagnostics for the high-temperature tail at one seam point: how the causal
cap and the monotonicity floor interacted. Audit hook only; nothing in the
run-time path reads it.
"""
struct UHighTailInfo{T<:AbstractFloat}
    α::T           # σ's log-tail growth rate dln(σ)/du; 0 if the log tail is inactive
    b_raw::T       # L's asymptotic b = dln(ε̂)/du after the floor, before the cap
    b_cap::T       # the causal cap on b, (1 + cs²_ext_cap)·α; 0 if inactive
    m_L_raw::T     # L's phase-2 slope after the floor, before the cap
    m_L_cap::T     # b_cap in L-slope units; 0 if inactive
    clamped::Bool  # the cap actually lowers the effective slope here
    floor_wins::Bool  # the cap fell below the floor, so the floor wins
end

# ---------------------------------------------------------------------------
# Tail machinery
#
# Every extension is a 1-D operator on one axis at a time, acting on a sample
# taken at the *physical* seam. The composition order is fixed and load-bearing:
# u-tail first, then x-tail applied to the possibly-already-u-tailed result.
# For an interior point the whole thing falls through to a single plain
# `bspline_eval3` call, bit for bit.
# ---------------------------------------------------------------------------

"""Value, slope and curvature of one 1-D track at a seam."""
struct Track1D{T<:AbstractFloat}
    f0::T
    f1::T
    f2::T
end

"""Which axis a tail is being applied along."""
@enumx TailAxis x u

"""
Blend width and slope guards for one tail.

`m_floor` enforces monotonicity and is applied *first*; `m_cap` is the causal
cap. When the cap falls below the floor the floor wins, lexicographically.
"""
struct TailSpec{T<:AbstractFloat}
    w::T
    m_floor::T
    m_cap::T
end

"""Everything `extended_sample` needs to know about one field's extensions."""
struct ExtSpec{T<:AbstractFloat}
    x_lo::T
    x_hi::T
    u_lo::T
    u_hi::T
    x_ext_lo::T
    u_m_floor::T
    x_low_log::Bool
    u_high_log::Bool
    u_high_b_cap::T
    shift_hat::T
    inv_c²::T
end

# ---------------------------------------------------------------------------
# The view
# ---------------------------------------------------------------------------

"""
    EOSTableView

Kernel-side view of a built EOS table: the two fitted splines plus the scalars
the evaluation needs. Everything except the two coefficient arrays is stored by
value, so this is a small object -- pass it by value into kernels.

Unlike the C++ original, which holds raw pointers and needs vendor-specific
mirror objects to keep the device allocations alive, the coefficient arrays are
held as ordinary array fields. `Adapt.adapt_structure` moves them to a device
and the resulting view keeps them alive by itself.
"""
struct EOSTableView{T<:AbstractFloat,A<:AbstractArray{T,3}}
    σ::BsplineView3{T,A}   # entropy, k_B per baryon
    L::BsplineView3{T,A}   # log10(ε_cgs + energy_shift_cgs)

    κ::T           # <= 1; the rescaled baryon mass is κ · m_B_table
    shift_hat::T   # energy_shift_cgs / c²
    conv_t::T      # MeV_to_erg / (m_B_table_g · c²)
    inv_c²::T      # 1 / c²

    x_lo::T        # physical box in x = log10(ρ* [g/cc])
    x_hi::T
    u_lo::T        # physical box in u = log10(T [MeV])
    u_hi::T
    y_lo::T        # physical box in y = Yₑ
    y_hi::T

    x_ext_lo::T    # extended box; Yₑ has no extension and is hard-clamped
    x_ext_hi::T
    u_ext_lo::T
    u_ext_hi::T

    ext_slope_floor_σ::T
    ext_slope_floor_L::T
    cs²_ext_cap::T
    max_iter::Int32
end

Base.eltype(::EOSTableView{T}) where {T} = T

# Implementations land in milestone M6 (tails, then evaluate/srange).

# Conserved to primitive variables, in the entropy-rapidity formulation.
#
# Translated from `entropy_eos/core/con2prim.hpp`.
#
# The unknowns are the entropy per baryon s and the rapidity w. Both choices
# remove constraints from the iterate space: any (s, w) inside the table domain
# is a physical state, w is well conditioned at both v → 0 and v → 1, and the
# EOS adapter is natively U(ρ, s, Yₑ) with U_s = T̂ > 0 everywhere, which makes
# the inner inversion monotone. Yₑ = D_Y/D is exact and sits outside the
# iteration; ρ = D/cosh(w) is exact given w.
#
# Production path is a damped 2×2 Newton with an analytic Jacobian. The nested
# 1-D scheme is the guaranteed fallback: the inner solve for w at fixed s has a
# unique root, and the outer solve in s is bracketed by a multi-point scan.

"""
    Con2PrimIn

Conserved state handed to the solver, all components in κ-rescaled g/cm³.
"""
struct Con2PrimIn{T<:Real}
    D::T
    τ::T
    D_Y::T
    S_par::T
    S_perp::T
    B²::T
end

"""
    Con2PrimOptions

Solver knobs. Every default was measured at `Float64`; see `defs.jl` for how
the tolerances behave at other scalar types, and note that the physics is not
validated below `Float64`.
"""
struct Con2PrimOptions{T<:Real}
    tol::T
    max_iter_newton::Int32
    max_iter_1d::Int32
    w_max::T
    τ_floor_rel::T
    bracket_scan::Int32
    seed_passes::Int32
    seed_s_iters::Int32
end

"""
    Con2PrimOptions{T}(; kwargs...)
    Con2PrimOptions(; kwargs...)

The unparameterized form defaults to `Float64`. Generic code should write
`Con2PrimOptions{eltype(eos)}()` instead, so the tolerances scale with the
scalar type actually in use.
"""
function Con2PrimOptions{T}(;
    tol=con2prim_tol(T),
    max_iter_newton=30,
    max_iter_1d=60,
    w_max=12,
    τ_floor_rel=tau_floor_rel(T),
    bracket_scan=17,
    seed_passes=3,
    seed_s_iters=16,
) where {T<:AbstractFloat}
    return Con2PrimOptions{T}(
        T(tol), Int32(max_iter_newton), Int32(max_iter_1d), T(w_max),
        T(τ_floor_rel), Int32(bracket_scan), Int32(seed_passes), Int32(seed_s_iters),
    )
end

Con2PrimOptions(; kwargs...) = Con2PrimOptions{Float64}(; kwargs...)

"""
    Con2PrimOut

Recovered primitive state plus the EOS point it corresponds to.

A failed solve still returns a fully populated best-effort state -- the best
iterate found, judged by the scaled residual norm. `result` reports what the
solver did; it is not a validity signal.
"""
struct Con2PrimOut{T<:Real}
    ρ::T
    s::T
    ye::T
    w::T
    W::T
    v_par::T
    v_perp::T
    eos::EOSPoint{T}
    result::C2PResult.T
    iters_newton::Int32
    iters_fallback::Int32
    flags::UInt32
end

"""
    Residuals

One evaluation of the residual pair and its analytic Jacobian, plus the
intermediate quantities the caller reuses.

The two residuals are the momentum residual `f₁ = sinh(w) - cosh(w)·V`,
deliberately not squared so the S → 0 root stays simple rather than double,
and the normalized energy residual `f₂ = (τ_model - τ)/max(τ, τ_floor_rel·D)`.
"""
struct Residuals{T<:Real}
    s::T
    w::T
    ρ::T
    coshw::T
    pt::EOSPoint{T}
    z::T
    v_par::T
    v_perp::T
    V::T
    f1::T
    f2::T
    df1_ds::T
    df1_dw::T
    df2_ds::T
    df2_dw::T
end

"""Outcome of the fallback's multi-point bracket search over `s`."""
struct BracketScanResult{T<:Real}
    bracketed::Bool
    s_lo::T
    s_hi::T
    s_best::T
end

"""A manufactured cold start: an exact seed derived without any prior iterate."""
struct ColdSeed{T<:Real}
    s::T
    w::T
    u::T
    ρ::T
    z::T
end

# Implementation lands in milestone M9.

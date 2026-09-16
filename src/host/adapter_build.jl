# Building the F → U adapter from a raw table.
#
# Translated from `entropy_eos/host/adapter_build.{hpp,cpp}`.
#
# This runs once, at startup. It validates the axes, fits the two splines,
# derives the baryon-mass rescaling κ, and records a monotonicity audit.
#
# It does *not* repair the table. The C++ tool `eos_repair` does that offline,
# and this package consumes the repaired result; `check_table` is what detects
# an unrepaired one.
#
# The κ rescaling is exact, not an approximation: with m_B* = κ·m_B and
# ρ* = κ·ρ, the identity m_B*(1 + U) = m_B(1 + ε) holds per baryon, and U ≥ 0
# by construction -- which is what lets prim2con form τ without cancellation.
# It is applied by relabelling the log-density grid origin by log10(κ), with no
# refit. Note that κ is part of the EOS identity, not an internal detail: a
# table swap that changes κ changes D, so checkpoints are not interchangeable
# across it.

"""
    BuildOptions

Adapter build knobs. These are host-side and stay `Float64`; the fit is always
done in double precision and narrowed afterwards if a narrower view is wanted.
"""
Base.@kwdef struct BuildOptions
    m_B_table_g::Float64 = M_B_DEFAULT_G
    refine::Int = 4
    uniform_tol::Float64 = 1.0e-8
    κ_margin_rel::Float64 = 1.0e-6
    κ_margin_abs::Float64 = 1.0e-12
    ext_cells::Int = 8
    ext_slope_floor_σ::Float64 = 1.0e-6
    ext_slope_floor_L::Float64 = 1.0e-8
    cs²_ext_cap::Float64 = 0.99
end

"""A location in the spline's own coordinates, with the value found there."""
struct AuditLoc{T<:AbstractFloat}
    x::T
    u::T
    y::T
    value::T
end

"""Worst-case monotonicity in u for one field, sampled on a refined grid."""
struct MonotonicityAudit{T<:AbstractFloat}
    min_value::T
    violation_count::Int
    worst::Vector{AuditLoc{T}}
end

"""Monotonicity audits for both fitted fields."""
struct AdapterAudit{T<:AbstractFloat}
    σ_u::MonotonicityAudit{T}
    L_u::MonotonicityAudit{T}
end

"""
    EOSTable{T}

A built adapter: the kernel-side view plus the host-only provenance a view does
not need.

Named `EOSTable` rather than the C++'s `EntropyEOS`, which would collide with
the module name.
"""
struct EOSTable{T<:AbstractFloat}
    view::EOSTableView{T,Array{T,3}}
    m_B_star_g::Float64
    m_B_table_g::Float64
    audit::AdapterAudit{T}
end

"""
    EOSTableView(t::EOSTable)

The kernel-side view of a built table. Use `Adapt.adapt` to move it to a device.
"""
EOSTableView(t::EOSTable) = t.view

"""κ, the baryon-mass rescaling factor (≤ 1)."""
κ(t::EOSTable) = t.view.κ

# Implementation lands in milestone M7:
#   build_eos(table::RawTable, opts::BuildOptions = BuildOptions()) -> EOSTable

# Table diagnostics.
#
# Translated from `entropy_eos/host/check.{hpp,cpp}`.
#
# `check_table` is pure and never throws: structural problems that would make
# the later checks unsafe become fatal messages instead, and the remaining
# classes are then skipped. That makes it cheap enough to run in-process at
# startup, right after loading a table, which is the intended use.
#
# What it validates: axes finite and strictly increasing; the required fields
# and the `energy_shift` attribute present and finite; entropy nonnegative;
# entropy and logenergy strictly increasing in T; Maxwell consistency against
# the stored pressure by finite differences; and the stored sound speed both
# for range and against a finite-difference estimate.
#
# Non-finite values in a field the pipeline never interprets are reported but
# are *not* fatal -- which is exactly what lets LS220 be used despite carrying
# a few genuine Inf points in `cs2` and `gamma`.

"""
    CheckOptions

`m_B_g` is the table family's baryon-mass convention; see [`M_B_DEFAULT_G`](@ref).
"""
Base.@kwdef struct CheckOptions
    m_B_g::Float64 = M_B_DEFAULT_G
    tol_consistency::Float64 = 0.05
    worst_n::Int = 10
end

"""A grid location and the metric value found there, with physical coordinates."""
struct CheckLoc{T<:AbstractFloat}
    iρ::Int
    jT::Int
    kYₑ::Int
    value::T
    ρ::T
    temp::T
    ye::T
end

"""
    CheckClassResult

One named class of finding.

For a *violation* class, `count`, `max` and `rms` are taken over violating
points only. For a *diagnostic* class, `max` and `rms` are over every evaluated
point and `count` is how many exceeded the threshold. A skipped class carries
NaN rather than a misleading zero.
"""
struct CheckClassResult{T<:AbstractFloat}
    name::String
    count::Int
    max::T
    rms::T
    worst::Vector{CheckLoc{T}}
end

"""
    CheckReport

`status` is `fatal` when the table is structurally broken -- a missing or
non-finite interpreted field, a bad axis -- as opposed to merely carrying
physics noise.
"""
struct CheckReport{T<:AbstractFloat}
    status::Status.T
    fatal_messages::Vector{String}
    classes::Vector{CheckClassResult{T}}
end

# Implementation lands in milestone M5:
#   check_table(t::RawTable, opts::CheckOptions = CheckOptions()) -> CheckReport
#   Base.show(io, ::MIME"text/plain", r::CheckReport)

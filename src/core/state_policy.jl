# The invalid-state policy layer: a never-fails path on top of con2prim.
#
# Translated from `entropy_eos/core/state_policy.hpp`.
#
# The contract is that `con2prim_safe` never fails, for any input whatsoever --
# non-finite garbage, a collapse state, a superluminal momentum demand, vacuum,
# a τ below the coldest state the table can express. It returns a valid,
# exactly solvable state plus flags saying what it did.
#
# Every repair goes through primitives: clamp (ρ, s, Yₑ, w) per policy, evaluate
# the EOS, then run prim2con to regenerate the conservatives. τ and Sᵢ are never
# hand-edited, so `con2prim(returned cons) == returned prims` holds by
# construction rather than by hope.
#
# Physics fidelity is explicitly not a goal in the excision regime. The point is
# to keep the evolution going with a state that is valid and exactly solvable,
# however bad the physics has become.

"""
    PolicyOptions

Validity thresholds. `ρ_atm` and `ρ_ceiling` are in κ-rescaled g/cm³, not raw
table densities -- a common source of error.

`s_atm` and `ye_atm` use NaN as a sentinel: NaN for `s_atm` means the midpoint
of the physical entropy range at the atmosphere density, and NaN for `ye_atm`
means preserve the incoming Yₑ. This must stay NaN rather than becoming
`Union{Nothing,T}`, which would destroy isbits-ness and so the GPU path.
"""
struct PolicyOptions{T<:Real}
    ρ_atm::T
    s_atm::T
    ye_atm::T
    atm_trigger::T
    ρ_ceiling::T
    w_cap::T
    D_max::T
    τ_max::T
    collapse_to_atmosphere::Bool
end

"""
    PolicyOptions{T}(; kwargs...)
    PolicyOptions(; kwargs...)

The unparameterized form defaults to `Float64`. Most fields are normally
derived from a built table by `default_policy` rather than set by hand.
"""
function PolicyOptions{T}(;
    ρ_atm=zero(T),
    s_atm=T(NaN),
    ye_atm=T(NaN),
    atm_trigger=1.01,
    ρ_ceiling=zero(T),
    w_cap=zero(T),
    D_max=zero(T),
    τ_max=zero(T),
    collapse_to_atmosphere=false,
) where {T<:AbstractFloat}
    return PolicyOptions{T}(
        T(ρ_atm), T(s_atm), T(ye_atm), T(atm_trigger), T(ρ_ceiling),
        T(w_cap), T(D_max), T(τ_max), collapse_to_atmosphere,
    )
end

PolicyOptions(; kwargs...) = PolicyOptions{Float64}(; kwargs...)

"""A primitive state in the variables the policy layer clamps."""
struct PrimState{T<:Real}
    ρ::T
    s::T
    ye::T
    w::T
end

"""
    Con2PrimSafeOut

The never-fails result: the recovered state, *and* the conserved state that
`prim2con` generates from it. The caller adopts `cons` as its new conserved
state.

`policy_flags == 0` means the input was already policy-valid, and then `cons`
is the input conservatives bit-identically. `policy_flags`, not
`base.result`, is the validity signal.
"""
struct Con2PrimSafeOut{T<:Real}
    base::Con2PrimOut{T}
    cons::Prim2ConOut{T}
    policy_flags::UInt32
    solved::Bool
end

# Implementation lands in milestone M10.

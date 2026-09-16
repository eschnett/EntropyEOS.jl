# Primitive to conserved variables, in the rapidity formulation.
#
# Translated from `entropy_eos/core/prim2con.hpp`.
#
# This direction is closed form and cannot fail. The only subtlety is
# arithmetic: τ is formed in a cancellation-free way, so that every term is
# individually small and (for ε, p >= 0) manifestly nonnegative, and no digits
# are lost as w → 0 in a cold flow. Naively forming E - D would cancel two
# O(D) terms down to an O(τ) result.
#
# The same algebraic form is reused verbatim as the energy residual's model in
# con2prim; the two must stay identical.

"""
    Prim2ConOut

Conserved state, all components in κ-rescaled g/cm³.
"""
struct Prim2ConOut{T<:Real}
    D::T
    τ::T
    D_Y::T
    S_par::T
    S_perp::T
    B²::T
end

# Implementation lands in milestone M8.

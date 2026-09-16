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

"""
    prim2con(eos, ρ, s, yₑ, w, B², cos_vB, u_guess = NaN)

Conserved variables from primitives, in the projected scalar form.

`cos_vB` is the cosine of the angle between velocity and magnetic field. Its
value never affects the result when `B² == 0` or `w == 0`, since the parallel
and perpendicular velocities then reduce to direction-independent values.

This direction is closed form and cannot fail.
"""
function prim2con(eos::EOSTableView{S}, ρ::T, s::T, yₑ::T, w::T, B²::T, cos_vB::T,
                  u_guess::T=T(NaN)) where {S,T}
    pt = evaluate(eos, ρ, s, yₑ, u_guess)

    W = cosh(w)
    v = tanh(w)
    sin_vB² = one(T) - cos_vB * cos_vB
    sin_vB = safe_sqrt(sin_vB² > zero(T) ? sin_vB² : zero(T))
    v_par = v * cos_vB
    v_perp = v * sin_vB

    z = ρ * pt.h * W * W          # the total enthalpy density

    D = ρ * W
    D_Y = D * yₑ
    # The magnetic inertia cancels exactly along the field and survives only
    # across it -- this is the design's projection verbatim, not a special case
    # of a more general form.
    S_par = z * v_par
    S_perp = (z + B²) * v_perp

    # τ formed so that every term is individually small and, for ε and p ≥ 0,
    # manifestly nonnegative. Building it as E - D instead would cancel two
    # terms of order D down to a result of order τ, losing most of the digits
    # in a cold flow. This exact expression is reused as the energy residual's
    # model in con2prim; the two must stay identical.
    half_sinh = sinh(T(0.5) * w)
    sinh_w = sinh(w)
    τ = T(2) * D * half_sinh * half_sinh + ρ * pt.U * W * W + pt.p * sinh_w * sinh_w +
        T(0.5) * B² * (one(T) + v * v) - T(0.5) * B² * v_par * v_par

    return Prim2ConOut{T}(D, τ, D_Y, S_par, S_perp, B²)
end

prim2con(eos::EOSTableView{S}, ρ, s, yₑ, w, B², cos_vB, u_guess=NaN) where {S} =
    prim2con(eos, promote(float(ρ), float(s), float(yₑ), float(w), float(B²), float(cos_vB),
                          float(u_guess))...)

"""
    prim2con(eos, ρ, s, yₑ, w, v_dir::SVector{3}, B::SVector{3}, u_guess = NaN)

Flat-metric Cartesian convenience form, returning `(out, S)` where `S` is the
momentum 3-vector.

`v_dir` must be a *unit* vector giving the velocity's direction; its magnitude
is carried by the rapidity `w`, not by `v_dir`. Reducing a general metric to the
parallel and perpendicular projections is the caller's job.
"""
function prim2con(eos::EOSTableView{S}, ρ::T, s::T, yₑ::T, w::T, v_dir::SVector{3,T},
                  B::SVector{3,T}, u_guess::T=T(NaN)) where {S,T}
    B² = B[1] * B[1] + B[2] * B[2] + B[3] * B[3]
    Bmag = safe_sqrt(B²)

    b, cos_vB = if Bmag > zero(T)
        bb = B / Bmag
        bb, v_dir[1] * bb[1] + v_dir[2] * bb[2] + v_dir[3] * bb[3]
    else
        # No field to project against, so all of v counts as perpendicular:
        # b = 0 makes the parallel contribution vanish below and the
        # perpendicular direction carries everything.
        zero(SVector{3,T}), zero(T)
    end

    out = prim2con(eos, ρ, s, yₑ, w, B², cos_vB, u_guess)

    # Perpendicular unit direction, by Gram-Schmidt against b. Left as v_dir
    # when the projection is degenerate: with no field there is nothing to
    # remove, and when v is (anti)parallel to B the perpendicular momentum is
    # itself zero, so the direction contributes nothing either way.
    sin_vB²_raw = one(T) - cos_vB * cos_vB
    sin_vB² = sin_vB²_raw > zero(T) ? sin_vB²_raw : zero(T)
    t = if Bmag > zero(T) && sin_vB² > perp_degenerate(T)
        (v_dir - cos_vB * b) / safe_sqrt(sin_vB²)
    else
        v_dir
    end

    return out, out.S_par * b + out.S_perp * t
end

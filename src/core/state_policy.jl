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

# ---------------------------------------------------------------------------
# Resolving unset options
#
# A zero or non-finite field means "not set", and the table supplies the
# default. Written as positive tests so a NaN falls through to the default.
# ---------------------------------------------------------------------------

@inline function pol_ρ_ceiling(eos::AbstractEOS{S}, pol::PolicyOptions{T}) where {S,T}
    return (isfinite(pol.ρ_ceiling) && pol.ρ_ceiling > zero(T)) ? pol.ρ_ceiling : exp10(T(last(logρ_bounds(eos))))
end

@inline function pol_ρ_atm(eos::AbstractEOS{S}, pol::PolicyOptions{T}) where {S,T}
    lo = (isfinite(pol.ρ_atm) && pol.ρ_atm > zero(T)) ? pol.ρ_atm : exp10(T(first(logρ_bounds(eos))))
    hi = pol_ρ_ceiling(eos, pol)
    return lo > hi ? hi : lo
end

@inline function pol_w_cap(pol::PolicyOptions{T}) where {T}
    return (isfinite(pol.w_cap) && pol.w_cap > zero(T)) ? pol.w_cap : acosh(T(100))
end

@inline function pol_atm_trigger(pol::PolicyOptions{T}) where {T}
    return (isfinite(pol.atm_trigger) && pol.atm_trigger > zero(T)) ? pol.atm_trigger : one(T)
end

@inline function pol_ye(eos::AbstractEOS{S}, pol::PolicyOptions{T}, ye_in) where {S,T}
    y_lo, y_hi = yₑ_bounds(eos)
    y = pol.ye_atm
    isfinite(y) || (y = T(ye_in))
    isfinite(y) || (y = T(0.5) * (T(y_lo) + T(y_hi)))
    return clamp(y, T(y_lo), T(y_hi))
end

# ---------------------------------------------------------------------------
# Deriving the bounds
# ---------------------------------------------------------------------------

"""
    policy_derive_bounds(eos, pol)

Return a copy of `pol` with the collapse ceilings filled in.

`D_max` follows exactly from the density ceiling and the rapidity cap. `τ_max`
is found by scanning a small grid in density and electron fraction at the
hottest state, because `ρh` is not monotone on a real table -- taking the
corner alone would underestimate it.
"""
function policy_derive_bounds(eos::AbstractEOS{S}, pol::PolicyOptions{T}) where {S,T}
    w_cap = pol_w_cap(pol)
    W_cap = cosh(w_cap)
    ρ_ceiling = pol_ρ_ceiling(eos, pol)
    D_max = ρ_ceiling * W_cap

    x_lo, x_hi = logρ_bounds(eos)
    y_lo, y_hi = yₑ_bounds(eos)
    nx, ny = 5, 3
    max_ρh = zero(T)
    for i in 0:(nx - 1)
        fx = T(i) / T(nx - 1)
        ρ = exp10(T(x_lo) + fx * (T(x_hi) - T(x_lo)))
        for j in 0:(ny - 1)
            fy = T(j) / T(ny - 1)
            yₑ = T(y_lo) + fy * (T(y_hi) - T(y_lo))
            sr = srange(eos, ρ, yₑ)
            pt = evaluate(eos, ρ, sr.s_max, yₑ, T(NaN))
            ρh = ρ * pt.h
            (isfinite(ρh) && ρh > max_ρh) && (max_ρh = ρh)
        end
    end

    return PolicyOptions{T}(; ρ_atm=pol.ρ_atm, s_atm=pol.s_atm, ye_atm=pol.ye_atm,
                            atm_trigger=pol.atm_trigger, ρ_ceiling=pol.ρ_ceiling, w_cap=pol.w_cap,
                            D_max=D_max, τ_max=max_ρh * W_cap * W_cap,
                            collapse_to_atmosphere=pol.collapse_to_atmosphere)
end

"""
    default_policy(eos, ρ_atm)

Everything but the atmosphere density derived from the table itself.

The rapidity cap corresponds to a Lorentz factor of 100, which must stay well
below the solver's own `w_max` or the cap is silently inoperative.
"""
function default_policy(eos::AbstractEOS{S}, ρ_atm::T) where {S,T}
    pol = PolicyOptions{T}(; ρ_atm=ρ_atm, s_atm=T(NaN), ye_atm=T(NaN), atm_trigger=T(1.01),
                           ρ_ceiling=exp10(T(last(logρ_bounds(eos)))), w_cap=acosh(T(100)),
                           collapse_to_atmosphere=false)
    return policy_derive_bounds(eos, pol)
end

"""
    policy_atmosphere(eos, pol, ye_in)

The atmosphere state: at rest, at the floor density, at the midpoint of the
physical entropy window unless the policy pins it.
"""
function policy_atmosphere(eos::AbstractEOS{S}, pol::PolicyOptions{T}, ye_in) where {S,T}
    ρ = pol_ρ_atm(eos, pol)
    yₑ = pol_ye(eos, pol, ye_in)
    sr = srange(eos, ρ, yₑ)
    s = isfinite(pol.s_atm) ? clamp(pol.s_atm, sr.s_min, sr.s_max) : T(0.5) * (sr.s_min + sr.s_max)
    return PrimState{T}(ρ, s, yₑ, zero(T))
end

# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

"""
    pol_project_core(eos, pol, in) -> (state, flags)

Clamp a primitive state into validity. The order is fixed and matters:
finiteness, then density, then electron fraction, then entropy at the *already
clamped* density and fraction, then rapidity.

The entropy is clamped to the **physical** window, not the extended one: a
converged state out in the extension zone is outside the table's validity, even
though the solver can represent it.

Exactly one entropy-window lookup on every path, and no solves.
"""
function pol_project_core(eos::AbstractEOS{S}, pol::PolicyOptions{T}, in::PrimState{T}) where {S,T}
    if !(isfinite(in.ρ) && isfinite(in.s) && isfinite(in.ye) && isfinite(in.w))
        return policy_atmosphere(eos, pol, in.ye), FLAG_POL_NONFINITE | FLAG_POL_ATMOSPHERE
    end

    ρ_hi = pol_ρ_ceiling(eos, pol)
    ρ_lo = pol_ρ_atm(eos, pol)
    # Deliberately the unscaled floor rather than the trigger, so that this is
    # idempotent and an atmosphere point checks clean.
    in.ρ < ρ_lo && return policy_atmosphere(eos, pol, in.ye), FLAG_POL_ATMOSPHERE

    flags = UInt32(0)
    ρ, s, yₑ, w = in.ρ, in.s, in.ye, in.w

    if ρ > ρ_hi
        ρ = ρ_hi
        flags |= FLAG_POL_ρ_CLAMPED
    end
    y_lo, y_hi = yₑ_bounds(eos)
    if yₑ < y_lo
        yₑ = T(y_lo)
        flags |= FLAG_POL_YE_CLAMPED
    elseif yₑ > y_hi
        yₑ = T(y_hi)
        flags |= FLAG_POL_YE_CLAMPED
    end

    sr = srange(eos, ρ, yₑ)
    if s < sr.s_min
        s = sr.s_min
        flags |= FLAG_POL_S_FLOORED
    elseif s > sr.s_max
        s = sr.s_max
        flags |= FLAG_POL_S_CEILED
    end

    w_cap = pol_w_cap(pol)
    if w < zero(T)
        w = zero(T)
        flags |= FLAG_POL_W_CAPPED
    elseif w > w_cap
        w = w_cap
        flags |= FLAG_POL_W_CAPPED
    end

    return PrimState{T}(ρ, s, yₑ, w), flags
end

"""
    check_prim_state(eos, ps, pol) -> flags

What a projection *would* change, without changing anything. No solves, so this
is cheap enough to run at every point of every timestep.
"""
@inline function check_prim_state(eos::AbstractEOS{S}, ps::PrimState{T}, pol::PolicyOptions{T}) where {S,T}
    return pol_project_core(eos, pol, ps)[2]
end

"""
    project_prim_state(eos, ps, pol) -> (state, flags)

Clamp into validity, reporting what changed. Idempotent.
"""
@inline function project_prim_state(eos::AbstractEOS{S}, ps::PrimState{T},
                                    pol::PolicyOptions{T}) where {S,T}
    return pol_project_core(eos, pol, ps)
end

# ---------------------------------------------------------------------------
# Packaging a repaired state
# ---------------------------------------------------------------------------

"""
    pol_package(eos, ps, S_par, S_perp, B², u_guess) -> (base, cons)

Regenerate conserved variables from a repaired primitive state.

This is the whole mechanism: every repair goes through primitives and the
conservatives come back out of `prim2con`, never edited by hand. That is what
makes `con2prim(returned cons) == returned prims` true by construction rather
than by hope.

The field direction is recovered from the incoming momentum, so the repaired
state keeps pointing the way the original did.
"""
function pol_package(eos::AbstractEOS{S}, ps::PrimState{T}, S_par::T, S_perp::T, B²::T,
                     u_guess::T) where {S,T}
    pt = evaluate(eos, ps.ρ, ps.s, ps.ye, u_guess)
    W = cosh(ps.w)
    v = tanh(ps.w)
    z = ps.ρ * pt.h * W * W
    sp = isfinite(S_par) ? S_par : zero(T)
    spp = (isfinite(S_perp) && S_perp > zero(T)) ? S_perp : zero(T)

    cos_vB = zero(T)
    if z > zero(T) && isfinite(z)
        vp = sp / z
        vq = spp / (z + B²)
        Vr = safe_sqrt(vp * vp + vq * vq)
        (isfinite(Vr) && Vr > zero(T)) && (cos_vB = clamp(vp / Vr, -one(T), one(T)))
    end

    cons = prim2con(eos, ps.ρ, ps.s, ps.ye, ps.w, B², cos_vB, pt.u_solved)
    sin_vB² = one(T) - cos_vB * cos_vB
    sin_vB = safe_sqrt(sin_vB² > zero(T) ? sin_vB² : zero(T))
    base = Con2PrimOut{T}(ps.ρ, ps.s, ps.ye, ps.w, W, v * cos_vB, v * sin_vB, pt,
                          C2PResult.converged_newton, Int32(0), Int32(0), pt.flags)
    return base, cons
end

"""
    pol_collapse(eos, pol, in, ye_in, B²) -> (base, cons, flags)

Handle a state past the collapse ceiling, by excising to atmosphere or by
projecting onto the ceiling primitives.
"""
function pol_collapse(eos::AbstractEOS{S}, pol::PolicyOptions{T}, in::Con2PrimIn{T}, ye_in::T,
                      B²::T) where {S,T}
    if pol.collapse_to_atmosphere
        ps = policy_atmosphere(eos, pol, ye_in)
        base, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², T(NaN))
        return base, cons, FLAG_POL_CEILING | FLAG_POL_ATMOSPHERE
    end

    w_cap = pol_w_cap(pol)
    ρ = pol_ρ_ceiling(eos, pol)
    yₑ = pol_ye(eos, pol, ye_in)
    sr = srange(eos, ρ, yₑ)
    s = sr.s_max
    pt = evaluate(eos, ρ, s, yₑ, T(NaN))

    Wc = cosh(w_cap)
    z = ρ * pt.h * Wc * Wc
    flags = FLAG_POL_CEILING
    V = zero(T)
    if z > zero(T) && isfinite(z)
        sp = isfinite(in.S_par) ? in.S_par : zero(T)
        spp = (isfinite(in.S_perp) && in.S_perp > zero(T)) ? in.S_perp : zero(T)
        V = safe_sqrt((sp / z)^2 + (spp / (z + B²))^2)
    end
    isfinite(V) || (V = zero(T))

    w = if V < tanh(w_cap)
        clamp(atanh(V), zero(T), w_cap)
    else
        flags |= FLAG_POL_W_CAPPED
        w_cap
    end

    ps = PrimState{T}(ρ, s, yₑ, w)
    base, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², pt.u_solved)
    return base, cons, flags
end

# ---------------------------------------------------------------------------
# Checking a conserved state
# ---------------------------------------------------------------------------

"""
    check_con_state(eos, in, pol; check_endpoints = false, solver = Con2PrimOptions())

Diagnose a conserved state without repairing it.

By default this is pure arithmetic -- no EOS evaluation at all -- so it is cheap
enough to run at every point of every timestep. With `check_endpoints` it also
solves at the two ends of the entropy window, which detects a state whose energy
lies outside anything the table can express.
"""
function check_con_state(eos::AbstractEOS{S}, in::Con2PrimIn{T}, pol::PolicyOptions{T};
                         check_endpoints::Bool=false, solver::Con2PrimOptions{T}=Con2PrimOptions{T}()) where {S,T}
    fin = isfinite(in.D) && isfinite(in.τ) && isfinite(in.D_Y) && isfinite(in.S_par) &&
          isfinite(in.S_perp) && isfinite(in.B²)
    (fin && in.D > zero(T) && in.B² >= zero(T)) ||
        return FLAG_POL_NONFINITE | FLAG_POL_ATMOSPHERE

    in.D < pol_atm_trigger(pol) * pol_ρ_atm(eos, pol) && return FLAG_POL_ATMOSPHERE

    flags = UInt32(0)
    B² = in.B² > zero(T) ? in.B² : zero(T)
    D_over = isfinite(pol.D_max) && pol.D_max > zero(T) && in.D > pol.D_max
    # The magnetic part of τ is excluded from the comparison, which is the
    # conservative choice.
    τ_over = isfinite(pol.τ_max) && pol.τ_max > zero(T) && (in.τ - B²) > pol.τ_max
    (D_over || τ_over) && (flags |= FLAG_POL_CEILING)

    (flags != 0 || !check_endpoints) && return flags

    y_lo, y_hi = yₑ_bounds(eos)
    yₑ = clamp(in.D_Y / in.D, T(y_lo), T(y_hi))
    sr_a = srange(eos, in.D, yₑ)
    sr_b = srange(eos, in.D / cosh(solver.w_max), yₑ)
    s_lo = min(sr_a.s_min, sr_b.s_min)
    s_hi = max(sr_a.s_max, sr_b.s_max)

    r_lo, _ = inner_solve_w(eos, in.D, in.τ, yₑ, in.S_par, in.S_perp, B², s_lo, solver.w_max,
                            solver.τ_floor_rel, zero(T), T(NaN), solver.max_iter_1d, solver.tol)
    isfinite(r_lo.f2) && r_lo.f2 > zero(T) && return flags | FLAG_POL_S_FLOORED

    r_hi, _ = inner_solve_w(eos, in.D, in.τ, yₑ, in.S_par, in.S_perp, B², s_hi, solver.w_max,
                            solver.τ_floor_rel, r_lo.w, r_lo.pt.u_solved, solver.max_iter_1d, solver.tol)
    isfinite(r_hi.f2) && r_hi.f2 < zero(T) && return flags | FLAG_POL_CEILING

    return flags
end

# ---------------------------------------------------------------------------
# The never-fails path
# ---------------------------------------------------------------------------

"""
    con2prim_safe(eos, in, opts, pol; s_guess = NaN, w_guess = NaN, u_guess = NaN)

Recover primitives from conserved variables, for *any* input whatsoever --
non-finite garbage, a collapse state, a superluminal momentum demand, vacuum, or
an energy below the coldest state the table can express.

Returns the recovered state together with the conserved variables that
correspond to it. The caller adopts `cons` as its new conserved state; because
every repair goes through primitives, that state is exactly solvable rather than
merely plausible.

`policy_flags == 0` means the input was already valid, and then `cons` is the
input bit-identically. That flag word, not `base.result`, is the validity
signal: the returned state is usable even when the solver reports a failure.

Physics fidelity is explicitly not the goal in the excision regime. The point is
to keep the evolution running with a state that is valid, however bad the
physics has become.
"""
function con2prim_safe(eos::AbstractEOS{S}, in::Con2PrimIn{T}, opts::Con2PrimOptions{T},
                       pol::PolicyOptions{T}, s_guess::T=T(NaN), w_guess::T=T(NaN),
                       u_guess::T=T(NaN)) where {S,T}
    B² = (isfinite(in.B²) && in.B² > zero(T)) ? in.B² : zero(T)
    fin = isfinite(in.D) && isfinite(in.τ) && isfinite(in.D_Y) && isfinite(in.S_par) &&
          isfinite(in.S_perp) && isfinite(in.B²)

    # 0. Garbage in: excise to atmosphere rather than propagate it.
    if !(fin && in.D > zero(T) && in.B² >= zero(T))
        ye_in = (isfinite(in.D) && in.D > zero(T) && isfinite(in.D_Y)) ? in.D_Y / in.D : T(NaN)
        ps = policy_atmosphere(eos, pol, ye_in)
        base, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², T(NaN))
        return Con2PrimSafeOut{T}(base, cons, FLAG_POL_NONFINITE | FLAG_POL_ATMOSPHERE, false)
    end

    ye_in = in.D_Y / in.D

    # 1. Vacuum and collapse, both pure arithmetic, before any solve is attempted.
    if in.D < pol_atm_trigger(pol) * pol_ρ_atm(eos, pol)
        ps = policy_atmosphere(eos, pol, ye_in)
        base, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², T(NaN))
        return Con2PrimSafeOut{T}(base, cons, FLAG_POL_ATMOSPHERE, false)
    end

    D_over = isfinite(pol.D_max) && pol.D_max > zero(T) && in.D > pol.D_max
    τ_over = isfinite(pol.τ_max) && pol.τ_max > zero(T) && (in.τ - B²) > pol.τ_max
    if D_over || τ_over
        base, cons, flags = pol_collapse(eos, pol, in, ye_in, B²)
        return Con2PrimSafeOut{T}(base, cons, flags, false)
    end

    # 2. The ordinary path.
    base = con2prim(eos, in, opts, s_guess, w_guess, u_guess)
    res = base.result
    it_n, it_f = base.iters_newton, base.iters_fallback
    u_warm = base.eos.u_solved

    if res === C2PResult.converged_newton || res === C2PResult.converged_fallback
        # 3. Cap the rapidity first, preserving D exactly, then project.
        w_cap = pol_w_cap(pol)
        ps = PrimState{T}(base.ρ, base.s, base.ye, base.w)
        pre = UInt32(0)
        if isfinite(base.w) && base.w > w_cap
            pre = FLAG_POL_W_CAPPED
            ps = PrimState{T}(in.D / cosh(w_cap), base.s, base.ye, w_cap)
        end
        ps, pf = project_prim_state(eos, ps, pol)
        pf |= pre

        if pf == 0
            # Already valid: hand back the input conservatives untouched, rather
            # than a round trip through prim2con that would perturb them.
            return Con2PrimSafeOut{T}(base, Prim2ConOut{T}(in.D, in.τ, in.D_Y, in.S_par, in.S_perp, in.B²),
                                      UInt32(0), true)
        end
        nb, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², u_warm)
        return Con2PrimSafeOut{T}(Con2PrimOut{T}(nb.ρ, nb.s, nb.ye, nb.w, nb.W, nb.v_par, nb.v_perp,
                                                 nb.eos, res, it_n, it_f, nb.flags), cons, pf, true)
    end

    # 4. The solve failed: diagnose with the outer function's own endpoints.
    w_ref = isfinite(base.w) ? clamp(base.w, zero(T), opts.w_max) : zero(T)
    y_lo, y_hi = yₑ_bounds(eos)
    ye_c = clamp(ye_in, T(y_lo), T(y_hi))
    ρ_ref = in.D / cosh(w_ref)
    sr_ref = srange(eos, ρ_ref, ye_c)

    r_lo, _ = inner_solve_w(eos, in.D, in.τ, ye_in, in.S_par, in.S_perp, B², sr_ref.s_min, opts.w_max,
                            opts.τ_floor_rel, w_ref, u_warm, opts.max_iter_1d, opts.tol)
    if isfinite(r_lo.f2) && r_lo.f2 > zero(T)
        # τ sits below the coldest state compatible with this momentum: floor
        # the entropy onto the physical minimum.
        w_cap = pol_w_cap(pol)
        w1 = isfinite(r_lo.w) ? clamp(r_lo.w, zero(T), w_cap) : zero(T)
        ρ1 = clamp(in.D / cosh(w1), pol_ρ_atm(eos, pol), pol_ρ_ceiling(eos, pol))
        sr1 = srange(eos, ρ1, ye_c)
        ps, pf = project_prim_state(eos, PrimState{T}(ρ1, sr1.s_min, ye_in, w1), pol)
        pf |= FLAG_POL_S_FLOORED
        nb, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², r_lo.pt.u_solved)
        return Con2PrimSafeOut{T}(Con2PrimOut{T}(nb.ρ, nb.s, nb.ye, nb.w, nb.W, nb.v_par, nb.v_perp,
                                                 nb.eos, res, it_n, it_f, nb.flags), cons, pf, true)
    end

    r_hi, _ = inner_solve_w(eos, in.D, in.τ, ye_in, in.S_par, in.S_perp, B², sr_ref.s_max, opts.w_max,
                            opts.τ_floor_rel, r_lo.w, r_lo.pt.u_solved, opts.max_iter_1d, opts.tol)
    if isfinite(r_hi.f2) && r_hi.f2 < zero(T)
        base2, cons, flags = pol_collapse(eos, pol, in, ye_in, B²)
        return Con2PrimSafeOut{T}(Con2PrimOut{T}(base2.ρ, base2.s, base2.ye, base2.w, base2.W,
                                                 base2.v_par, base2.v_perp, base2.eos, res, it_n, it_f,
                                                 base2.flags), cons, flags, true)
    end

    # Neither diagnosis fits: excise.
    ps = policy_atmosphere(eos, pol, ye_in)
    nb, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², r_hi.pt.u_solved)
    return Con2PrimSafeOut{T}(Con2PrimOut{T}(nb.ρ, nb.s, nb.ye, nb.w, nb.W, nb.v_par, nb.v_perp, nb.eos,
                                             res, it_n, it_f, nb.flags), cons, FLAG_POL_ATMOSPHERE, true)
end

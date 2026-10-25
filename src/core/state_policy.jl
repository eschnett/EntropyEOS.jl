# The state policy, in two levels: a projection onto the valid range, and the
# application's physics on top of it.
#
# This replaces the translation of `entropy_eos/core/state_policy.hpp` that
# 1.x carried; the C++ follows this design (CODE.md, "State policy").
#
# Level 1, `con2prim_safe`, never fails, for any input whatsoever -- non-finite
# garbage, vacuum, a collapse state, a superluminal momentum demand, an energy
# outside anything the table can express. It returns the valid state closest
# to the input, by one rule per face of the valid range, and flags each face
# that was active. It has no notion of an atmosphere or of collapse.
#
# Level 2, `apply_policy`, decides what a state should become. The reference
# implementation replaces it by the atmosphere where 1.x did. An application
# with its own rules replaces that function rather than configuring it.
#
# Every output goes through primitives: the conservatives come back out of
# `prim2con`, never edited by hand, so `con2prim(returned cons)` reproduces the
# returned primitives by construction.

"""
    ProjectionOptions

The valid range that `con2prim_safe` projects onto: `ρ ∈ [ρ_min, ρ_max]` and
`w ≤ w_cap`, with `Yₑ` in the table's range and `s` in the physical window at
`(ρ, Yₑ)`. Densities are κ-rescaled g/cm³, not raw table densities -- a common
source of error.

A zero or non-finite field means "not set": `ρ_min` and `ρ_max` then default to
the table's density box, and `w_cap` to a Lorentz factor of 100.
"""
struct ProjectionOptions{T<:Real}
    ρ_min::T
    ρ_max::T
    w_cap::T
end

"""
    ProjectionOptions{T}(; ρ_min = 0, ρ_max = 0, w_cap = 0)
    ProjectionOptions(; kwargs...)

The unparameterized form defaults to `Float64`. Usually built by
[`default_projection`](@ref).
"""
function ProjectionOptions{T}(; ρ_min=zero(T), ρ_max=zero(T), w_cap=zero(T)) where {T<:AbstractFloat}
    return ProjectionOptions{T}(T(ρ_min), T(ρ_max), T(w_cap))
end

ProjectionOptions(; kwargs...) = ProjectionOptions{Float64}(; kwargs...)

"""
    PolicyOptions

The reference physics of [`apply_policy`](@ref): when a state goes to the
atmosphere, and what the atmosphere is.

  * `s_atm`: the atmosphere's entropy, clamped into the window at `ρ_min`. NaN
    means the window's minimum, the coldest atmosphere.
  * `ye_atm`: the atmosphere's `Yₑ`. NaN means the incoming state's, clamped.
  * `atm_trigger`: a state with `D < atm_trigger·ρ_min` goes to the atmosphere.
    A non-positive or non-finite value means 1.
  * `collapse_to_atmosphere`: a state on the ρ_max face goes to the atmosphere.

The sentinels must stay NaN rather than become `Union{Nothing,T}`, which would
destroy isbits-ness and so the GPU path.
"""
struct PolicyOptions{T<:Real}
    s_atm::T
    ye_atm::T
    atm_trigger::T
    collapse_to_atmosphere::Bool
end

"""
    PolicyOptions{T}(; s_atm = NaN, ye_atm = NaN, atm_trigger = 1.01, collapse_to_atmosphere = false)
    PolicyOptions(; kwargs...)

The unparameterized form defaults to `Float64`. Needs no table.
"""
function PolicyOptions{T}(; s_atm=T(NaN), ye_atm=T(NaN), atm_trigger=1.01,
                          collapse_to_atmosphere=false) where {T<:AbstractFloat}
    return PolicyOptions{T}(T(s_atm), T(ye_atm), T(atm_trigger), collapse_to_atmosphere)
end

PolicyOptions(; kwargs...) = PolicyOptions{Float64}(; kwargs...)

"""A primitive state in the variables the projection bounds."""
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

`policy_flags == 0` means the input was already valid, and then `cons` is the
input conservatives bit-identically. `policy_flags`, not `base.result`, is the
validity signal. `solved` says whether `con2prim` ran, and `base.result` is
then what it returned.
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

@inline function proj_ρ_max(eos::AbstractEOS{S}, proj::ProjectionOptions{T}) where {S,T}
    return (isfinite(proj.ρ_max) && proj.ρ_max > zero(T)) ? proj.ρ_max : exp10(T(last(logρ_bounds(eos))))
end

@inline function proj_ρ_min(eos::AbstractEOS{S}, proj::ProjectionOptions{T}) where {S,T}
    lo = (isfinite(proj.ρ_min) && proj.ρ_min > zero(T)) ? proj.ρ_min : exp10(T(first(logρ_bounds(eos))))
    hi = proj_ρ_max(eos, proj)
    return lo > hi ? hi : lo
end

@inline function proj_w_cap(proj::ProjectionOptions{T}) where {T}
    return (isfinite(proj.w_cap) && proj.w_cap > zero(T)) ? proj.w_cap : acosh(T(100))
end

@inline function pol_atm_trigger(pol::PolicyOptions{T}) where {T}
    return (isfinite(pol.atm_trigger) && pol.atm_trigger > zero(T)) ? pol.atm_trigger : one(T)
end

"""
    default_projection(eos, ρ_min)

The valid range with everything but its floor density taken from the table:
`ρ_max` at the top of the density box, and `w_cap` at a Lorentz factor of 100,
which must stay well below the solver's own `w_max` or the cap is silently
inoperative.
"""
function default_projection(eos::AbstractEOS{S}, ρ_min::T) where {S,T}
    return ProjectionOptions{T}(; ρ_min=ρ_min, ρ_max=exp10(T(last(logρ_bounds(eos)))), w_cap=acosh(T(100)))
end

# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------

"""
    clamp_yₑ(eos, yₑ) -> (yₑ, flags)

The Yₑ faces. A value within one ulp of the table's range is valid, since
`Yₑ·D/D` can come back one ulp off.
"""
@inline function clamp_yₑ(eos::AbstractEOS{S}, yₑ::T) where {S,T}
    y_lo, y_hi = yₑ_bounds(eos)
    lo, hi = T(y_lo), T(y_hi)
    yₑ < prevfloat(lo) && return lo, FLAG_POL_YE_FLOORED
    yₑ > nextfloat(hi) && return hi, FLAG_POL_YE_CEILED
    return yₑ, UInt32(0)
end

"""
    s_face(sr, s) -> flags

The s face that `s` lies beyond, to the slack, or 0 inside the window `sr`.
[`inside_window`](@ref) and [`project_prim_state`](@ref) share it, so that what
counts as valid and what gets projected cannot drift apart.
"""
@inline function s_face(sr::SRange{T}, s::T) where {T}
    slack = face_slack(T) * (sr.s_max - sr.s_min)
    s < sr.s_min - slack && return FLAG_POL_S_FLOORED
    s > sr.s_max + slack && return FLAG_POL_S_CEILED
    return UInt32(0)
end

"""
    inside_window(eos, ρ, s, yₑ) -> Bool

Whether `s` and `Yₑ` lie inside their faces of the valid range at density `ρ`,
to the slack with which [`con2prim_safe`](@ref) tests them: `Yₑ` within one ulp
of the table's range, and `s` within `srange(eos, ρ, yₑ)` widened on each side
by `face_slack` of its width (10⁻⁷ at `Float64`). `ρ` is not tested against a
range. A non-finite argument or `ρ ≤ 0` gives `false`.

This is the validity test for states an application makes itself, for example
by reconstruction. Every state that `con2prim_safe` or
[`project_prim_state`](@ref) returns passes it, including one within the slack
of an s face, which `con2prim_safe` returns unflagged; testing the window
exactly would reject that state. The EOS can be evaluated throughout the band,
which lies far inside [`srange_extended`](@ref).
"""
@inline function inside_window(eos::AbstractEOS{S}, ρ::T, s::T, yₑ::T) where {S,T}
    (isfinite(ρ) && ρ > zero(T) && isfinite(s) && isfinite(yₑ)) || return false
    clamp_yₑ(eos, yₑ)[2] == 0 || return false
    return s_face(srange(eos, ρ, yₑ), s) == 0
end

"""
    fixed_state(eos, proj, yₑ = NaN)

What `con2prim_safe` returns where there is nothing to project: at `ρ_min`, at
rest, at the entropy window's minimum, with the input's (already clamped) `yₑ`,
or the middle of the Yₑ range where there is none. It exists only so that a
kernel never propagates NaN; level 2 is expected to replace it.
"""
@inline function fixed_state(eos::AbstractEOS{S}, proj::ProjectionOptions{T}, yₑ::T=T(NaN)) where {S,T}
    ρ = proj_ρ_min(eos, proj)
    if !isfinite(yₑ)
        y_lo, y_hi = yₑ_bounds(eos)
        yₑ = T(0.5) * (T(y_lo) + T(y_hi))
    end
    return PrimState{T}(ρ, srange(eos, ρ, yₑ).s_min, yₑ, zero(T))
end

"""
    pol_package(eos, ps, S_par, S_perp, B², u_guess) -> (base, cons)

Regenerate conserved variables from a projected primitive state.

The velocity's direction relative to `B` is taken from the incoming momentum at
the projected state's own enthalpy density, which makes the returned momentum
exactly parallel to the input: a face that lowers `S` scales it, never turns it.
"""
function pol_package(eos::AbstractEOS{S}, ps::PrimState{T}, S_par::T, S_perp::T, B²::T,
                     u_guess::T) where {S,T}
    pt = evaluate(eos, ps.ρ, ps.s, ps.ye, u_guess)
    W = cosh(ps.w)
    v = tanh(ps.w)
    cos_vB, _ = velocity_direction(S_par, S_perp, B², ps.ρ * pt.h * W * W)
    cons = prim2con_at(pt, ps.ρ, ps.ye, ps.w, B², cos_vB)
    sin_vB² = one(T) - cos_vB * cos_vB
    sin_vB = safe_sqrt(sin_vB² > zero(T) ? sin_vB² : zero(T))
    base = Con2PrimOut{T}(ps.ρ, ps.s, ps.ye, ps.w, W, v * cos_vB, v * sin_vB, pt,
                          C2PResult.converged_newton, Int32(0), Int32(0), pt.flags)
    return base, cons
end

"""Package `ps`, keeping the outcome of the `con2prim` call (if any) in `base`."""
@inline function package_out(eos::AbstractEOS{S}, ps::PrimState{T}, in::Con2PrimIn{T}, B²::T, u::T,
                             res::C2PResult.T, it_n::Int32, it_f::Int32, flags::UInt32,
                             solved::Bool) where {S,T}
    nb, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², u)
    base = Con2PrimOut{T}(nb.ρ, nb.s, nb.ye, nb.w, nb.W, nb.v_par, nb.v_perp, nb.eos, res, it_n, it_f,
                          nb.flags)
    return Con2PrimSafeOut{T}(base, cons, flags, solved)
end

"""
    valid_w_interval(D, ρ_min, ρ_max, w_cap) -> (w_lo, w_hi, hi_is_cap)

The rapidities at which a state keeping `D` has `ρ ∈ [ρ_min, ρ_max]` and
`w ≤ w_cap`. `hi_is_cap` says whether `w_cap` or `ρ_min` bounds it above.
"""
@inline function valid_w_interval(D::T, ρ_min::T, ρ_max::T, w_cap::T) where {T}
    q = D / ρ_min
    w_ρmin = acosh(q > one(T) ? q : one(T))
    hi_is_cap = w_cap <= w_ρmin
    w_hi = hi_is_cap ? w_cap : w_ρmin
    q = D / ρ_max
    w_lo = acosh(q > one(T) ? q : one(T))
    w_lo > w_hi && (w_lo = w_hi)
    return w_lo, w_hi, hi_is_cap
end

"""Is a converged state inside every face, to the slack?"""
@inline function inside_faces(eos::AbstractEOS{S}, ρ::T, s::T, yₑ::T, w::T, ρ_min::T, ρ_max::T,
                              w_cap::T) where {S,T}
    δ = face_slack(T)
    (isfinite(ρ) && isfinite(s) && isfinite(w)) || return false
    (ρ >= ρ_min * (one(T) - δ) && ρ <= ρ_max * (one(T) + δ) && w <= w_cap * (one(T) + δ)) || return false
    return inside_window(eos, ρ, s, yₑ)
end

@inline con_finite(in::Con2PrimIn) = isfinite(in.D) && isfinite(in.τ) && isfinite(in.D_Y) &&
                                     isfinite(in.S_par) && isfinite(in.S_perp) && isfinite(in.B²)

# ---------------------------------------------------------------------------
# Primitive states
# ---------------------------------------------------------------------------

"""
    project_prim_state(eos, ps, proj) -> (state, flags)

The projection on primitives, where every face is a plain clamp: ρ, then Yₑ,
then s in the window at the clamped `(ρ, Yₑ)`, then w. Each face is tested with
the same slack as in `con2prim_safe`, so an output of `con2prim_safe` checks
clean; the Yₑ and s faces are [`inside_window`](@ref)'s. Idempotent, bit for bit.

A non-finite state, `ρ ≤ 0` or `w < 0` has no closest valid state, and gives
`FLAG_POL_NONFINITE` with the fixed state. `w` is a magnitude, so `w ≥ 0` is not
a face.
"""
function project_prim_state(eos::AbstractEOS{S}, ps::PrimState{T}, proj::ProjectionOptions{T}) where {S,T}
    if !(isfinite(ps.ρ) && isfinite(ps.s) && isfinite(ps.ye) && isfinite(ps.w) && ps.ρ > zero(T) &&
         ps.w >= zero(T))
        return fixed_state(eos, proj), FLAG_POL_NONFINITE
    end
    δ = face_slack(T)
    ρ_min = proj_ρ_min(eos, proj)
    ρ_max = proj_ρ_max(eos, proj)
    w_cap = proj_w_cap(proj)

    ρ, s, w = ps.ρ, ps.s, ps.w
    flags = UInt32(0)
    if ρ < ρ_min * (one(T) - δ)
        ρ = ρ_min
        flags |= FLAG_POL_ρ_FLOORED
    elseif ρ > ρ_max * (one(T) + δ)
        ρ = ρ_max
        flags |= FLAG_POL_ρ_CEILED
    end
    yₑ, fy = clamp_yₑ(eos, ps.ye)
    flags |= fy

    sr = srange(eos, ρ, yₑ)
    fs = s_face(sr, s)
    if fs == FLAG_POL_S_FLOORED
        s = sr.s_min
    elseif fs == FLAG_POL_S_CEILED
        s = sr.s_max
    end
    flags |= fs

    if w > w_cap * (one(T) + δ)
        w = w_cap
        flags |= FLAG_POL_W_CEILED
    end
    return PrimState{T}(ρ, s, yₑ, w), flags
end

"""
    check_prim_state(eos, ps, proj) -> flags

What [`project_prim_state`](@ref) *would* change, without changing anything. No
solves, so this is cheap enough to run at every point of every timestep.
"""
@inline function check_prim_state(eos::AbstractEOS{S}, ps::PrimState{T}, proj::ProjectionOptions{T}) where {S,T}
    return project_prim_state(eos, ps, proj)[2]
end

# ---------------------------------------------------------------------------
# Checking a conserved state
# ---------------------------------------------------------------------------

"""
    check_con_state(eos, in, proj; check_endpoints = false, solver = Con2PrimOptions())

Diagnose a conserved state without repairing it. Every flag it reports is one
that `con2prim_safe` sets; it does not promise to report every face.

By default this is pure arithmetic -- no EOS evaluation at all -- so it is cheap
enough to run at every point of every timestep. It reports a non-finite input,
`D < ρ_min` (`ρ_FLOORED`), `D > ρ_max·cosh(w_cap)` (`ρ_CEILED`), and `Yₑ`
outside the table. With `check_endpoints` it also runs the entropy-edge part of
`con2prim_safe`'s diagnosis, and reports `S_FLOORED` or `S_CEILED` where that
alone decides the projection, and only when no ρ face is reported. It tests the
edges moved outward by the face slack, and the energy residual against
`check_f2_tol` (the solver's tolerance at Float64, 128 times it at Float32), so
that a state `con2prim_safe` accepts as lying on a face is not reported.
"""
function check_con_state(eos::AbstractEOS{S}, in::Con2PrimIn{T}, proj::ProjectionOptions{T};
                         check_endpoints::Bool=false,
                         solver::Con2PrimOptions{T}=Con2PrimOptions{T}()) where {S,T}
    (con_finite(in) && in.D > zero(T) && in.B² >= zero(T)) || return FLAG_POL_NONFINITE

    δ = face_slack(T)
    ρ_min = proj_ρ_min(eos, proj)
    ρ_max = proj_ρ_max(eos, proj)
    w_cap = proj_w_cap(proj)
    D = in.D
    yₑ, flags = clamp_yₑ(eos, in.D_Y / D)

    D < ρ_min * (one(T) - δ) && return flags | FLAG_POL_ρ_FLOORED
    D > ρ_max * cosh(w_cap) * (one(T) + δ) && return flags | FLAG_POL_ρ_CEILED
    check_endpoints || return flags

    # First the branch con2prim_safe's diagnosis takes, on the edges
    # themselves; where it would run a face solve, nothing can be promised.
    # Then the cold or hot test, on the edges moved outward by the slack: a
    # root that fails it there lies beyond the band con2prim_safe accepts as
    # valid, and so cannot be returned unflagged. One call site each, as in
    # con2prim_safe.
    tol = solver.tol
    tol_f2 = max(tol, check_f2_tol(T))
    w_lo, w_hi, _ = valid_w_interval(D, ρ_min, ρ_max, w_cap)
    e = es = nan_edge_ends(T)
    for pass in 1:2
        ee = edge_ends(eos, D, in.τ, yₑ, in.S_par, in.S_perp, in.B², pass == 1 ? zero(T) : δ, w_lo, w_hi, T(NaN),
                       solver)
        if pass == 1
            e = ee
            (edge_φ(e.min_lo) > zero(T) || edge_φ(e.max_hi) < zero(T) || edge_φ(e.min_hi) < zero(T)) &&
                return flags
        else
            es = ee
        end
    end
    for k in 1:2
        upper = k == 2
        # con2prim_safe's hot test runs on the edge only where its root is in range.
        upper && !(edge_φ(e.max_lo) <= zero(T)) && break
        ra = upper ? es.max_lo : es.min_lo
        rb = upper ? es.max_hi : es.min_hi
        (edge_φ(ra) <= zero(T) && edge_φ(rb) >= zero(T)) || break
        r = edge_root(eos, D, in.τ, yₑ, in.S_par, in.S_perp, in.B², upper, δ, ra, rb, solver)
        f1_converged(r, tol) || break
        !upper && r.f2 > tol_f2 && return flags | FLAG_POL_S_FLOORED
        upper && r.f2 < -tol_f2 && return flags | FLAG_POL_S_CEILED
    end
    return flags
end

# ---------------------------------------------------------------------------
# Level 1: the projection
# ---------------------------------------------------------------------------

# The conditions under which a try in `con2prim_safe` is the answer.
const TRY_ALWAYS = Int8(0)      # no condition: no other face can be active
const TRY_COLD = Int8(1)        # on an edge, f₂ > 0: τ raised
const TRY_HOT = Int8(2)         # on an edge, f₂ < 0: τ lowered
const TRY_S_LOWERED = Int8(3)   # at w_hi, φ < 0: S lowered
const TRY_D_LOWERED = Int8(4)   # at ρ_max, ρ_max·cosh w < D: D lowered

"""One candidate solve in `con2prim_safe`: an edge root, or a face solve at fixed ρ."""
struct FaceTry{T<:Real}
    edge::Bool
    upper::Bool
    ρ::T
    w::T
    free_w::Bool
    flags::UInt32
    cond::Int8
end

# Which branch of `con2prim_safe` the first stage took.
const SAFE_NONFINITE = UInt8(0)   # no solve: the input is not a state
const SAFE_ρ_FLOOR = UInt8(1)     # D below ρ_min: put at rest at ρ_min
const SAFE_ρ_CEIL = UInt8(2)      # D above ρ_max·cosh(w_cap): on ρ_max
const SAFE_ORDINARY = UInt8(3)    # solved, then diagnosed if not inside

"""
    Con2PrimSafeStage

What [`con2prim_safe_start`](@ref) hands to [`con2prim_safe_finish`](@ref): the
solver's stage, the clamped Yₑ and its flags, and the branch taken.
"""
struct Con2PrimSafeStage{T<:Real}
    c2p::Con2PrimStage{T}
    yₑ::T
    flags::UInt32
    branch::UInt8
    done::Bool
end

@inline is_done(st::Con2PrimSafeStage) = st.done

"""The placeholder output of a first stage that is not done: no EOS call, all NaN."""
@inline function nan_safe_out(::Type{T}) where {T}
    n = T(NaN)
    return Con2PrimSafeOut{T}(nan_c2p_out(T), Prim2ConOut{T}(n, n, n, n, n, n), UInt32(0), false)
end

"""
    con2prim_safe(eos, in, opts, proj, s_guess = NaN, w_guess = NaN, u_guess = NaN)

Recover primitives from conserved variables, for *any* input whatsoever, and
project them onto the valid range `proj` describes.

Returns the recovered state together with the conserved variables that
correspond to it; the caller adopts `cons` as its new conserved state. Each
face of the valid range that was active sets its own bit in `policy_flags`, and
the returned state lies on it. Each face fixes one primitive and frees one
conservative; everything else is kept, to the solver's tolerance:

| face | flag | what gives way |
| --- | --- | --- |
| `s_max` | `S_CEILED` | τ lowered |
| `s_min` | `S_FLOORED` | τ raised |
| `w_cap` | `W_CEILED` | S lowered |
| `ρ_max` | `ρ_CEILED` | D lowered |
| `ρ_min` | `ρ_FLOORED` | S lowered to `w = acosh(D′/ρ_min)`, D raised to `D′ = max(D, ρ_min)` |
| Yₑ bounds | `YE_FLOORED`, `YE_CEILED` | `D_Y` set to `Yₑ·D` with Yₑ clamped |

`policy_flags == 0` means the input was already valid, and then `cons` is the
input bit-identically. Two outcomes are not projections: `NONFINITE` (a
non-finite input, `D ≤ 0` or `B² < 0`) and `NOT_CONVERGED` (the solve failed,
but the diagnosis finds a valid state inside the range). Both return a fixed
valid state only so that a kernel never propagates NaN; [`apply_policy`](@ref)
is expected to replace it.

This level has no notion of an atmosphere or of collapse; that is
[`apply_policy`](@ref)'s.

This is [`con2prim_safe_start`](@ref) followed, where needed, by
[`con2prim_safe_finish`](@ref), and gives the same result bit for bit. On a
GPU the two stages can run as separate kernels.
"""
function con2prim_safe(eos::AbstractEOS{S}, in::Con2PrimIn{T}, opts::Con2PrimOptions{T},
                       proj::ProjectionOptions{T}, s_guess::T=T(NaN), w_guess::T=T(NaN),
                       u_guess::T=T(NaN)) where {S,T}
    out, st = con2prim_safe_start(eos, in, opts, proj, s_guess, w_guess, u_guess)
    is_done(st) && return out
    return con2prim_safe_finish(eos, in, opts, proj, st)
end

"""
    con2prim_safe_start(eos, in, opts, proj, s_guess = NaN, w_guess = NaN, u_guess = NaN) -> (out, stage)

The first stage of [`con2prim_safe`](@ref): [`con2prim_start`](@ref) on the
input with Yₑ clamped. It is done when the Newton converged on a state inside
every face, the common case, and then `out` is `con2prim_safe`'s result.
Otherwise `out` is a placeholder of NaNs, and [`con2prim_safe_finish`](@ref)
on `stage` gives the result: the fallback, the diagnosis and the face solves,
and the inputs no solve applies to. Run as two kernels the same way as
[`con2prim_start`](@ref).
"""
function con2prim_safe_start(eos::AbstractEOS{S}, in::Con2PrimIn{T}, opts::Con2PrimOptions{T},
                             proj::ProjectionOptions{T}, s_guess::T=T(NaN), w_guess::T=T(NaN),
                             u_guess::T=T(NaN)) where {S,T}
    ρ_min = proj_ρ_min(eos, proj)
    ρ_max = proj_ρ_max(eos, proj)
    w_cap = proj_w_cap(proj)
    δ = face_slack(T)
    D, τ, S_par, S_perp = in.D, in.τ, in.S_par, in.S_perp
    B² = (isfinite(in.B²) && in.B² > zero(T)) ? in.B² : zero(T)
    n = T(NaN)
    none = Con2PrimStage{T}(n, n, n, Int32(0), false)

    if !(con_finite(in) && D > zero(T) && in.B² >= zero(T))
        return nan_safe_out(T), Con2PrimSafeStage{T}(none, n, FLAG_POL_NONFINITE, SAFE_NONFINITE, false)
    end

    # 1. Yₑ decouples from the rest, and is clamped first.
    yₑ, flags = clamp_yₑ(eos, in.D_Y / D)

    # 2. Faces decided by D alone, before any solve. Below ρ_min every rapidity
    #    gives ρ < ρ_min, so the state is put at rest at ρ_min, the least mass
    #    added; above ρ_max·cosh(w_cap) none gives ρ ≤ ρ_max.
    if D < ρ_min * (one(T) - δ)
        return nan_safe_out(T), Con2PrimSafeStage{T}(none, yₑ, flags, SAFE_ρ_FLOOR, false)
    elseif D > ρ_max * cosh(w_cap) * (one(T) + δ)
        return nan_safe_out(T), Con2PrimSafeStage{T}(none, yₑ, flags, SAFE_ρ_CEIL, false)
    end

    # 3. The ordinary path. Only a state inside every face is taken from the
    #    solve; anything else is decided by the diagnosis in the second stage,
    #    so that a converged and a failed solve of neighbouring states project
    #    alike.
    D_Y = flags == 0 ? in.D_Y : yₑ * D
    base, c = con2prim_start(eos, Con2PrimIn{T}(D, τ, D_Y, S_par, S_perp, B²), opts, s_guess, w_guess, u_guess)
    if is_done(c) && inside_faces(eos, base.ρ, base.s, yₑ, base.w, ρ_min, ρ_max, w_cap)
        # The input's own B², so that a -0.0 also comes back bit for bit.
        return Con2PrimSafeOut{T}(base, Prim2ConOut{T}(D, τ, D_Y, S_par, S_perp, in.B²), flags, true),
               Con2PrimSafeStage{T}(c, yₑ, flags, SAFE_ORDINARY, true)
    end
    return nan_safe_out(T), Con2PrimSafeStage{T}(c, yₑ, flags, SAFE_ORDINARY, false)
end

"""
    con2prim_safe_finish(eos, in, opts, proj, stage) -> Con2PrimSafeOut

The second stage of [`con2prim_safe`](@ref), for a `stage` from
[`con2prim_safe_start`](@ref).
"""
function con2prim_safe_finish(eos::AbstractEOS{S}, in::Con2PrimIn{T}, opts::Con2PrimOptions{T},
                              proj::ProjectionOptions{T}, st::Con2PrimSafeStage{T}) where {S,T}
    # Every solver below has exactly one call site, and the result is packaged
    # at one: a GPU compiler inlines every call, and each copy carries whole EOS
    # evaluations. The decision is therefore made first, as at most two "tries"
    # -- an edge root or a face solve, each with the condition under which it is
    # the answer -- which one loop then runs.
    res, it_n, it_f, solved = C2PResult.converged_newton, Int32(0), Int32(0), false
    ρ_min = proj_ρ_min(eos, proj)
    ρ_max = proj_ρ_max(eos, proj)
    w_cap = proj_w_cap(proj)
    tol = opts.tol
    D, τ, S_par, S_perp = in.D, in.τ, in.S_par, in.S_perp
    B² = (isfinite(in.B²) && in.B² > zero(T)) ? in.B² : zero(T)

    yₑ = st.yₑ
    flags = st.flags
    u = T(NaN)
    e = nan_edge_ends(T)
    ntry = 0
    try1 = try2 = FaceTry{T}(false, false, zero(T), zero(T), false, UInt32(0), TRY_ALWAYS)

    if st.branch == SAFE_ρ_FLOOR
        ntry = 1
        try1 = FaceTry{T}(false, false, ρ_min, zero(T), false, FLAG_POL_ρ_FLOORED, TRY_ALWAYS)
    elseif st.branch == SAFE_ρ_CEIL
        ntry = 1
        try1 = FaceTry{T}(false, false, ρ_max, zero(T), true, FLAG_POL_ρ_CEILED, TRY_ALWAYS)
    elseif st.branch == SAFE_ORDINARY
        D_Y = flags == 0 ? in.D_Y : yₑ * D
        base = con2prim_finish(eos, Con2PrimIn{T}(D, τ, D_Y, S_par, S_perp, B²), opts, st.c2p)
        res, it_n, it_f, solved = base.result, base.iters_newton, base.iters_fallback, true
        converged = res === C2PResult.converged_newton || res === C2PResult.converged_fallback
        # The fallback may have converged on a state inside every face.
        if converged && inside_faces(eos, base.ρ, base.s, yₑ, base.w, ρ_min, ρ_max, w_cap)
            # The input's own B², so that a -0.0 also comes back bit for bit.
            return Con2PrimSafeOut{T}(base, Prim2ConOut{T}(D, τ, D_Y, S_par, S_perp, in.B²), flags, true)
        end

        # 4. The diagnosis, on the rapidities at which keeping D leaves ρ
        #    and w in range. Along an entropy edge the state that keeps D and
        #    S is a root of φ; G = f₂ along the solve's curve rises with s,
        #    so the cold end of the valid part of that curve decides "too
        #    cold", the hot end "too hot".
        u = base.eos.u_solved
        w_lo, w_hi, hi_is_cap = valid_w_interval(D, ρ_min, ρ_max, w_cap)
        hi_flag = hi_is_cap ? FLAG_POL_W_CEILED : FLAG_POL_ρ_FLOORED
        ρ_hi = hi_is_cap ? D / cosh(w_cap) : ρ_min
        on_ρmax = FaceTry{T}(false, false, ρ_max, zero(T), true, FLAG_POL_ρ_CEILED, TRY_ALWAYS)
        on_w_hi = FaceTry{T}(false, false, ρ_hi, w_hi, false, hi_flag, TRY_ALWAYS)
        e = edge_ends(eos, D, τ, yₑ, S_par, S_perp, B², zero(T), w_lo, w_hi, u, opts)
        ntry = 1
        if edge_φ(e.min_lo) > zero(T)
            # 4.1 The s_min edge's root lies above ρ_max: every state
            #     keeping D and S is too cold or too dense, and D has to give.
            try1 = on_ρmax
        elseif edge_φ(e.max_hi) < zero(T)
            # 4.2 The s_max edge's root lies beyond w_hi: every state
            #     keeping D and S is too hot or too fast, and S has to give.
            try1 = on_w_hi
        else
            # 4.3 Too cold at the cold end: on the s_min edge if its root is
            #     in range, else at w_hi if the state there carries less
            #     momentum than the input. 4.4 Too hot at the hot end: on the
            #     s_max edge if its root is in range, else at ρ_max if that
            #     removes mass.
            ntry = 2
            try1 = edge_φ(e.min_hi) >= zero(T) ?
                   FaceTry{T}(true, false, zero(T), zero(T), false, FLAG_POL_S_FLOORED, TRY_COLD) :
                   FaceTry{T}(false, false, ρ_hi, w_hi, false, hi_flag, TRY_S_LOWERED)
            try2 = edge_φ(e.max_lo) <= zero(T) ?
                   FaceTry{T}(true, true, zero(T), zero(T), false, FLAG_POL_S_CEILED, TRY_HOT) :
                   FaceTry{T}(false, false, ρ_max, zero(T), true, FLAG_POL_ρ_CEILED, TRY_D_LOWERED)
        end
    end

    # Run the tries. A face state that fails its condition by no more than the
    # tolerance is itself a solution, and is kept in `near`.
    F = FaceState{T}(T(NaN), T(NaN), T(NaN), T(NaN), T(NaN), UInt32(0))
    found = false
    near = F
    near_res = T(Inf)
    for k in 1:ntry
        t = k == 1 ? try1 : try2
        resid = T(NaN)
        if t.edge
            r = edge_root(eos, D, τ, yₑ, S_par, S_perp, B², t.upper, zero(T), t.upper ? e.max_lo : e.min_lo,
                          t.upper ? e.max_hi : e.min_hi, opts)
            Ft = FaceState{T}(r.ρ, r.s, r.w, r.pt.u_solved, zero(T), t.flags)
            resid = f1_converged(r, tol) ? r.f2 : T(NaN)
            accept = t.cond == TRY_COLD ? r.f2 > zero(T) : r.f2 < zero(T)
        else
            G = face_solve(eos, t.ρ, t.w, t.free_w, w_cap, yₑ, τ, S_par, S_perp, B², u, opts)
            Ft = FaceState{T}(G.ρ, G.s, G.w, G.u, G.φ, G.flags | t.flags)
            t.cond == TRY_S_LOWERED && (resid = G.φ)
            t.cond == TRY_D_LOWERED && (resid = (t.ρ * cosh(G.w) - D) / D)
            accept = t.cond == TRY_ALWAYS || (t.cond == TRY_S_LOWERED && G.φ < zero(T)) ||
                     (t.cond == TRY_D_LOWERED && t.ρ * cosh(G.w) < D)
        end
        if accept
            F, found = Ft, true
            break
        end
        if abs(resid) <= tol && abs(resid) < near_res
            near, near_res = Ft, abs(resid)
        end
    end

    # 4.5 Neither cold nor hot. A face state that solves the input to the
    #     tolerance is a solution on the face: where the energy barely depends
    #     on s, as for a cold state in Float32, G is flat to within the
    #     tolerance across the window and the solve may have converged
    #     anywhere. Otherwise a valid state lies inside the range, but the solve
    #     did not return it; this is only seen where the inner root is not
    #     unique.
    if !found && ntry > 0 && near_res <= tol
        F, found = near, true
    end
    found && !(isfinite(F.ρ) && isfinite(F.s) && isfinite(F.w)) && (found = false)
    if found
        flags |= F.flags
    elseif flags & FLAG_POL_NONFINITE == 0
        flags |= FLAG_POL_NOT_CONVERGED
    end
    ps = found ? PrimState{T}(F.ρ, F.s, yₑ, F.w) : fixed_state(eos, proj, yₑ)
    return package_out(eos, ps, in, B², found ? F.u : T(NaN), res, it_n, it_f, flags, solved)
end

# ---------------------------------------------------------------------------
# Level 2: the application's physics
# ---------------------------------------------------------------------------

"""
    apply_policy(eos, in, out, proj, pol) -> Con2PrimSafeOut

The reference physics on top of [`con2prim_safe`](@ref), called on its output:

```julia
out = con2prim_safe(eos, in, opts, proj)       # level 1: projection
out = apply_policy(eos, in, out, proj, pol)    # level 2: physics
```

Replaces the state by the atmosphere -- at rest, at `ρ_min`, with `pol.s_atm`
and `pol.ye_atm` -- where 1.x would have: on `ρ_FLOORED`, for
`D < atm_trigger·ρ_min`, on `NONFINITE` and `NOT_CONVERGED`, and on `ρ_CEILED`
when `collapse_to_atmosphere` is set. It then adds `FLAG_POL_ATMOSPHERE` to
level 1's bits, which say why. Otherwise it returns `out` unchanged.

An application with its own rules replaces this function rather than
configuring it. It has `in` and `out` to compare, so needs no further flags.
"""
function apply_policy(eos::AbstractEOS{S}, in::Con2PrimIn{T}, out::Con2PrimSafeOut{T}, proj::ProjectionOptions{T},
                      pol::PolicyOptions{T}) where {S,T}
    f = out.policy_flags
    ρ_min = proj_ρ_min(eos, proj)
    atm = (f & (FLAG_POL_ρ_FLOORED | FLAG_POL_NONFINITE | FLAG_POL_NOT_CONVERGED)) != 0 ||
          !(in.D >= pol_atm_trigger(pol) * ρ_min) ||
          (pol.collapse_to_atmosphere && (f & FLAG_POL_ρ_CEILED) != 0)
    atm || return out

    yₑ = pol.ye_atm
    if !isfinite(yₑ)
        yₑ = (isfinite(in.D_Y) && isfinite(in.D) && in.D > zero(T)) ? in.D_Y / in.D : out.base.ye
    end
    yₑ, _ = clamp_yₑ(eos, isfinite(yₑ) ? yₑ : out.base.ye)
    sr = srange(eos, ρ_min, yₑ)
    s = isfinite(pol.s_atm) ? clamp(pol.s_atm, sr.s_min, sr.s_max) : sr.s_min

    B² = (isfinite(in.B²) && in.B² > zero(T)) ? in.B² : zero(T)
    b = out.base
    return package_out(eos, PrimState{T}(ρ_min, s, yₑ, zero(T)), in, B², T(NaN), b.result, b.iters_newton,
                       b.iters_fallback, f | FLAG_POL_ATMOSPHERE, out.solved)
end

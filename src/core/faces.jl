# The solves behind the projection in `con2prim_safe`.
#
# Each face of the valid range fixes one primitive and frees one conservative
# (CODE.md, "State policy"). What a set of active faces leaves to solve is one
# of three one-dimensional problems:
#
#   - along an entropy edge, s = s_edge(D/cosh w), find the w that keeps S;
#   - at fixed ρ and w, find the s that keeps τ;
#   - at fixed ρ, with w following S, find the s that keeps τ.
#
# Every one is a bracketed find whose bracket lies inside the valid range, so
# whatever iterate it stops at is a valid state on its face. Running out of
# iterations costs accuracy in the conservative the face keeps, never validity.

"""
    FaceState

A state on one or more faces of the valid range, as a face solve leaves it.
`φ = tanh w - V` is the momentum residual of the input `S` at this state: where
`S` is freed, `φ < 0` says the state carries less momentum than the input.
"""
struct FaceState{T<:Real}
    ρ::T
    s::T
    w::T
    u::T
    φ::T
    flags::UInt32
end

"""
    velocity_direction(S_par, S_perp, B², z) -> (cos_vB, V)

The direction relative to `B` that the momentum `(S_par, S_perp)` gives a state
of total enthalpy density `z`, and the speed it would need. A state built with
this `cos_vB` has momentum parallel to the input, so a face that lowers `S`
scales it rather than turning it.
"""
@inline function velocity_direction(S_par::T, S_perp::T, B²::T, z::T) where {T}
    sp = isfinite(S_par) ? S_par : zero(T)
    spp = (isfinite(S_perp) && S_perp > zero(T)) ? S_perp : zero(T)
    cos_vB = zero(T)
    V = zero(T)
    if z > zero(T) && isfinite(z)
        vp = sp / z
        vq = spp / (z + B²)
        Vr = safe_sqrt(vp * vp + vq * vq)
        if isfinite(Vr) && Vr > zero(T)
            cos_vB = clamp(vp / Vr, -one(T), one(T))
            V = Vr
        end
    end
    return cos_vB, V
end

"""
    w_from_momentum(a, S_par, S_perp, B²)

The rapidity at which a state with `ρh = a` carries the momentum
`(S_par, S_perp)`. Pure arithmetic.

With `y = sinh² w` the momentum relation is
`y/(1+y) = (σ_par/(1+y))² + (σ_perp/(1+y+b))²`, with `σ = S/a` and `b = B²/a`.
The difference of the two sides is increasing and concave in `y`, so the root
is unique, and Newton from either side lands at or below it and then climbs to
it monotonically. The bracket
`[0, y_hydro]`, with `y_hydro` the hydrodynamic root with all of `S` counted
(the field only lowers the right-hand side), guards the rounding. Working in
`y` rather than `W²` keeps the digits of a slow flow, and in ratios keeps
`Float32` from overflowing.
"""
function w_from_momentum(a::T, S_par::T, S_perp::T, B²::T) where {T}
    σp = S_par / a
    σq = (isfinite(S_perp) && S_perp > zero(T)) ? S_perp / a : zero(T)
    b = B² / a
    σ² = σp * σp + σq * σq
    (isfinite(σ²) && σ² > zero(T)) || return zero(T)

    lo = zero(T)
    hi = T(2) * σ² / (one(T) + safe_sqrt(one(T) + T(4) * σ²))
    y = hi
    for _ in 1:SEED_SCALAR_ITERS
        q = one(T) + y
        qb = q + b
        rp = σp / q
        rq = σq / qb
        F = y / q - rp * rp - rq * rq
        F > zero(T) ? (hi = y) : (lo = y)
        F == zero(T) && break
        dF = one(T) / (q * q) + T(2) * rp * rp / q + T(2) * rq * rq / qb
        y_next = y - F / dF
        (isfinite(y_next) && lo <= y_next <= hi) || (y_next = T(0.5) * (lo + hi))
        y_next == y && break
        y = y_next
    end
    return asinh(safe_sqrt(y))
end

# ---------------------------------------------------------------------------
# Along an entropy edge
# ---------------------------------------------------------------------------

"""
    edge_entropy(eos, ρ, yₑ, upper, shift)

An edge of the physical window at `ρ`, moved outward by `shift` times the
window's width: `s_max + shift·width` or `s_min - shift·width`.
"""
@inline function edge_entropy(eos::AbstractEOS{S}, ρ::T, yₑ::T, upper::Bool, shift::T) where {S,T}
    sr = srange(eos, ρ, yₑ)
    d = shift * (sr.s_max - sr.s_min)
    return upper ? sr.s_max + d : sr.s_min - d
end

"""
    edge_residuals(eos, D, τ, yₑ, S_par, S_perp, B², upper, shift, w, u_prev, τ_floor_rel)

`residuals` on an entropy edge: at rapidity `w`, with `s` the edge of the
physical window at the density `D/cosh w` actually written, not at some fixed
density. `con2prim_safe` uses the edge itself (`shift = 0`); `check_con_state`
moves it outward by the face slack.
"""
@inline function edge_residuals(eos::AbstractEOS{S}, D::T, τ::T, yₑ::T, S_par::T, S_perp::T, B²::T,
                                upper::Bool, shift::T, w::T, u_prev::T, τ_floor_rel::T) where {S,T}
    s = edge_entropy(eos, D / cosh(w), yₑ, upper, shift)
    return residuals(eos, D, τ, yₑ, S_par, S_perp, B², s, w, u_prev, τ_floor_rel)
end

"""
The face root find's variable: `f₁/cosh w = tanh w - V`. It has `f₁`'s sign and
roots, and is what `f1_converged` tests, but unlike `f₁` it stays monotone where
`V > 1`.
"""
@inline edge_φ(r::Residuals) = r.f1 / r.coshw

"""
    EdgeEnds

The two entropy edges evaluated at both ends of the valid rapidity interval
`[w_lo, w_hi]`: the four evaluations the projection's diagnosis decides from.
"""
struct EdgeEnds{T<:Real}
    min_lo::Residuals{T}
    min_hi::Residuals{T}
    max_lo::Residuals{T}
    max_hi::Residuals{T}
end

function edge_ends(eos::AbstractEOS{S}, D::T, τ::T, yₑ::T, S_par::T, S_perp::T, B²::T, shift::T,
                   w_lo::T, w_hi::T, u::T, opts::Con2PrimOptions{T}) where {S,T}
    # One call site in a loop rather than four: a GPU compiler inlines every
    # call, and each copy of `residuals` carries a whole EOS evaluation.
    rs = MVector{4,Residuals{T}}(undef)
    @inbounds for k in 1:4
        rs[k] = edge_residuals(eos, D, τ, yₑ, S_par, S_perp, B², k > 2, shift, isodd(k) ? w_lo : w_hi, u,
                               opts.τ_floor_rel)
        u = rs[k].pt.u_solved
    end
    return @inbounds EdgeEnds{T}(rs[1], rs[2], rs[3], rs[4])
end

"""
    nan_edge_ends(T)

Placeholder for paths that decide without the diagnosis, so that the variable
holding the edges is always defined.
"""
@inline function nan_edge_ends(::Type{T}) where {T}
    n = T(NaN)
    pt = EOSPoint{T}(n, n, n, n, n, n, n, n, n, n, n, n, Int32(0), UInt32(0))
    r = Residuals{T}(n, n, n, n, pt, n, n, n, n, n, n, n, n, n, n)
    return EdgeEnds{T}(r, r, r, r)
end

"""
    edge_root(eos, D, τ, yₑ, S_par, S_perp, B², upper, shift, ra, rb, opts) -> Residuals

The root of `φ` along an entropy edge between `ra.w` and `rb.w`, where
`φ(ra) ≤ 0 ≤ φ(rb)`: the state on the edge that keeps `D` and `S`.

Illinois rather than Newton, because `residuals`' Jacobian is taken at fixed `s`
and the edge's `s` moves with `w`. Where `φ` has several roots in the bracket,
which happens where a table has `cs² ≥ 1`, this returns whichever it reaches.
Returns the iterate with the smallest `|φ|`; every iterate lies on the edge.
"""
function edge_root(eos::AbstractEOS{S}, D::T, τ::T, yₑ::T, S_par::T, S_perp::T, B²::T, upper::Bool, shift::T,
                   ra::Residuals{T}, rb::Residuals{T}, opts::Con2PrimOptions{T}) where {S,T}
    fa = edge_φ(ra)
    fb = edge_φ(rb)
    best = abs(fa) <= abs(fb) ? ra : rb
    best_f = abs(edge_φ(best))
    f1_converged(best, opts.tol) && return best

    a, b = ra.w, rb.w
    u = best.pt.u_solved
    side = 0
    for _ in 1:opts.max_iter_1d
        w = fb != fa ? (a * fb - b * fa) / (fb - fa) : T(0.5) * (a + b)
        (isfinite(w) && a < w < b) || (w = T(0.5) * (a + b))
        a < w < b || break

        r = edge_residuals(eos, D, τ, yₑ, S_par, S_perp, B², upper, shift, w, u, opts.τ_floor_rel)
        f = edge_φ(r)
        u = r.pt.u_solved
        if abs(f) < best_f
            best = r
            best_f = abs(f)
        end
        f1_converged(r, opts.tol) && return r

        # Illinois: halve the stale side's value when the same side is kept
        # twice, as in con2prim's outer solve.
        if f < zero(T)
            a, fa = w, f
            side == -1 && (fb *= T(0.5))
            side = -1
        else
            b, fb = w, f
            side == 1 && (fa *= T(0.5))
            side = 1
        end
    end
    return best
end

# ---------------------------------------------------------------------------
# At a fixed density
# ---------------------------------------------------------------------------

"""
    fixed_ρ_point(eos, ρ, s, yₑ, w_fixed, free_w, S_par, S_perp, B², u)
        -> (pt, w, τ, φ)

The state at `(ρ, s)`: its rapidity (`w_fixed`, or the one carrying the input
momentum if `free_w`), the `τ` that `prim2con` gives it with the input
momentum's direction, and `φ`.
"""
@inline function fixed_ρ_point(eos::AbstractEOS{S}, ρ::T, s::T, yₑ::T, w_fixed::T, free_w::Bool, S_par::T,
                               S_perp::T, B²::T, u::T) where {S,T}
    pt = evaluate(eos, ρ, s, yₑ, u)
    a = ρ * pt.h
    w = free_w ? w_from_momentum(a, S_par, S_perp, B²) : w_fixed
    coshw = cosh(w)
    cos_vB, V = velocity_direction(S_par, S_perp, B², a * coshw * coshw)
    τ = prim2con_at(pt, ρ, yₑ, w, B², cos_vB).τ
    return pt, w, τ, tanh(w) - V
end

"""
    s_solve_at_fixed_ρ(eos, ρ, w_fixed, free_w, yₑ, τ, S_par, S_perp, B², u, opts) -> FaceState

The entropy at which a state of density `ρ` has energy `τ`, searched in the
physical window at `ρ`. With `free_w` the rapidity follows the input momentum
(`S` and `τ` kept, `D` freed: the ρ_max face); otherwise it is `w_fixed` (`τ`
kept, `S` freed: the w_cap and ρ_min faces).

If even the window's coldest state is too energetic, the result is on the
s_min face as well, with `τ` raised, and `S_FLOORED`; if even its hottest has
too little energy, on the s_max face with `S_CEILED`. Otherwise an Illinois
find to `|Δτ| ≤ tol·max(τ, τ_floor_rel·D)` returns the iterate with the
smallest residual.
"""
function s_solve_at_fixed_ρ(eos::AbstractEOS{S}, ρ::T, w_fixed::T, free_w::Bool, yₑ::T, τ::T,
                                      S_par::T, S_perp::T, B²::T, u::T, opts::Con2PrimOptions{T}) where {S,T}
    D_ref = ρ * cosh(w_fixed)
    denom = τ > opts.τ_floor_rel * D_ref ? τ : opts.τ_floor_rel * D_ref
    tol = opts.tol

    sr = srange(eos, ρ, yₑ)
    a, b = sr.s_min, sr.s_max
    ga = gb = T(NaN)
    # NaN until an evaluation is finite, so a solve that never gets a finite
    # residual is reported as such rather than as the window's lower edge.
    best = FaceState{T}(ρ, T(NaN), w_fixed, u, T(NaN), UInt32(0))
    best_g = T(Inf)
    side = 0
    # The two window ends, then the Illinois iterates, through one evaluation
    # site: a GPU compiler inlines every call.
    for k in 1:(Int(opts.max_iter_1d) + 2)
        s = if k == 1
            a
        elseif k == 2
            b
        else
            sk = gb != ga ? (a * gb - b * ga) / (gb - ga) : T(0.5) * (a + b)
            (isfinite(sk) && a < sk < b) ? sk : T(0.5) * (a + b)
        end
        k > 2 && !(a < s < b) && break

        pt, w, τm, φ = fixed_ρ_point(eos, ρ, s, yₑ, w_fixed, free_w, S_par, S_perp, B², u)
        u = pt.u_solved
        g = (τm - τ) / denom
        F = FaceState{T}(ρ, s, w, u, φ, UInt32(0))
        if k == 1
            # Too energetic even at the window's coldest state: on the s_min
            # face as well, with τ raised.
            g >= -tol && return FaceState{T}(ρ, s, w, u, φ, g > tol ? FLAG_POL_S_FLOORED : UInt32(0))
            ga = g
        elseif k == 2
            # Too little energy even at its hottest: on the s_max face.
            g <= tol && return FaceState{T}(ρ, s, w, u, φ, g < -tol ? FLAG_POL_S_CEILED : UInt32(0))
            gb = g
        end
        if abs(g) < best_g
            best = F
            best_g = abs(g)
        end
        k > 2 || continue
        best_g <= tol && break

        if g < zero(T)
            a, ga = s, g
            side == -1 && (gb *= T(0.5))
            side = -1
        else
            b, gb = s, g
            side == 1 && (ga *= T(0.5))
            side = 1
        end
    end
    return best
end

"""
    face_solve(eos, ρ, w_fixed, free_w, w_cap, yₑ, τ, S_par, S_perp, B², u, opts) -> FaceState

The faces that fix ρ. With `free_w`, the ρ_max face: `S` and `τ` kept, `D`
lowered, a nested one-dimensional solve, outer in `s` over the window at `ρ`,
inner in `w` by pure arithmetic, since `h` is fixed at fixed `(ρ, s)`. If the
rapidity this needs exceeds `w_cap`, the w_cap face is active too, and the state
is re-solved at `(ρ, w_cap)` with `τ` kept and `S` lowered, adding `W_CEILED`.
Without `free_w`, the w_cap and ρ_min faces: `w = w_fixed`, `τ` kept, `S` freed.
"""
function face_solve(eos::AbstractEOS{S}, ρ::T, w_fixed::T, free_w::Bool, w_cap::T, yₑ::T, τ::T,
                              S_par::T, S_perp::T, B²::T, u::T, opts::Con2PrimOptions{T}) where {S,T}
    extra = UInt32(0)
    F = FaceState{T}(ρ, T(NaN), w_fixed, u, T(NaN), UInt32(0))
    for _ in 1:2
        F = s_solve_at_fixed_ρ(eos, ρ, w_fixed, free_w, yₑ, τ, S_par, S_perp, B², u, opts)
        (free_w && F.w > w_cap * (one(T) + face_slack(T))) || break
        free_w, w_fixed, u = false, w_cap, F.u
        extra = FLAG_POL_W_CEILED
    end
    return FaceState{T}(F.ρ, F.s, F.w, F.u, F.φ, F.flags | extra)
end

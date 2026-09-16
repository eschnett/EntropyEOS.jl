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
struct EOSPoint{T<:Real}
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
struct SRange{T<:Real}
    s_min::T
    s_max::T
end

"""
    UHighTailInfo

Diagnostics for the high-temperature tail at one seam point: how the causal
cap and the monotonicity floor interacted. Audit hook only; nothing in the
run-time path reads it.
"""
struct UHighTailInfo{T<:Real}
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
struct Track1D{T<:Real}
    f0::T
    f1::T
    f2::T
end

"""
Blend width and slope guards for one tail.

`m_floor` enforces monotonicity and is applied *first*; `m_cap` is the causal
cap. When the cap falls below the floor the floor wins, lexicographically.
"""
struct TailSpec{T<:Real}
    w::T
    m_floor::T
    m_cap::T
end

"""Everything `extended_sample` needs to know about one field's extensions."""
struct ExtSpec{T<:Real}
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

# ---------------------------------------------------------------------------
# Tail construction
#
# Each tail is a two-phase curvature ramp, unified across the low (d < 0) and
# high (d > 0) sides by the sign of the offset. Inside the blend cell the
# second derivative ramps linearly from its seam value to zero; beyond it the
# tail is the straight continuation, with f'' identically zero. That makes the
# result C² at the seam and finite everywhere, which is what the T-solve needs:
# a hard clamp would zero the derivative and stall Newton.
# ---------------------------------------------------------------------------

"""
    ramp_track(t, d, w)

The raw two-phase ramp, with no guards and no shape override. The guarded
variants below configure it purely by choosing which slope and curvature to
hand in.

Within phase 1 the slope is an affine, monotone function of the progress
`|d| - d²/2w`, so it never overshoots the interval between its two endpoint
values anywhere in the blend cell.
"""
@inline function ramp_track(t::Track1D{T}, d::T, w::T) where {T}
    ad = d < zero(T) ? -d : d
    sgn = d < zero(T) ? -one(T) : one(T)
    if ad <= w
        f0 = t.f0 + t.f1 * d + t.f2 * (T(0.5) * d * d - ad * ad * ad / (T(6) * w))
        f1 = t.f1 + t.f2 * (d - sgn * d * d / (T(2) * w))
        f2 = t.f2 * (one(T) - ad / w)
        return Track1D{T}(f0, f1, f2)
    else
        m = t.f1 + sgn * t.f2 * w * T(0.5)
        f_edge = t.f0 + t.f1 * sgn * w + t.f2 * w * w / T(3)
        return Track1D{T}(f_edge + m * (d - sgn * w), m, zero(T))
    end
end

"""
    phase2_slope(t, sgn, w)

The asymptotic slope `ramp_track` will produce for a track already carrying its
effective slope and curvature.
"""
@inline phase2_slope(t::Track1D{T}, sgn::T, w::T) where {T} = t.f1 + sgn * t.f2 * w * T(0.5)

"""
    floor_slope(t, sgn, w, m_floor)

The monotonicity guard, as a rewrite of the track.

A non-positive floor is a no-op, which is every track except the u-direction
primary one. Otherwise the effective slope is raised to at least the floor, and
the effective curvature is then capped so the asymptotic slope cannot fall back
below it.

Factored out rather than inlined into `generic_track` so that the causal cap and
the audit hook reuse exactly this arithmetic instead of a second copy of it.
"""
@inline function floor_slope(t::Track1D{T}, sgn::T, w::T, m_floor::T) where {T}
    m_floor > zero(T) || return t
    f1_eff = t.f1 < m_floor ? m_floor : t.f1
    cap = T(2) * (f1_eff - m_floor) / w      # >= 0
    f2_eff = t.f2
    if sgn < zero(T)
        f2_eff > cap && (f2_eff = cap)       # low side: m = f1 - f2·w/2
    else
        f2_eff < -cap && (f2_eff = -cap)     # high side: m = f1 + f2·w/2
    end
    return Track1D{T}(t.f0, f1_eff, f2_eff)
end

"""
    cap_slope(t, w, m_cap)

The causal slope cap: the only guard that ever *lowers* a slope. If the
asymptotic slope exceeds the cap, lower the curvature alone so that it equals
the cap exactly.

The slope at the seam is deliberately left alone, unlike the monotonicity
guard: the seam value and seam slope must keep matching the boundary spline
sample so the tail stays C¹ there and U and U_s stay continuous across it. The
price is that the slope inside the blend cell rides down towards the cap and can
transiently exceed it over that one cell, which the design anticipates.
"""
@inline function cap_slope(t::Track1D{T}, w::T, m_cap::T) where {T}
    phase2_slope(t, one(T), w) > m_cap || return t
    return Track1D{T}(t.f0, t.f1, T(2) * (m_cap - t.f1) / w)
end

"""Generic tail with the optional monotonicity guard."""
@inline function generic_track(t::Track1D{T}, d::T, w::T, m_floor::T) where {T}
    sgn = d < zero(T) ? -one(T) : one(T)
    return ramp_track(floor_slope(t, sgn, w, m_floor), d, w)
end

"""
    capped_track(t, d, w, m_floor, m_cap)

The tail with both high-side guards, used only for L's high-temperature primary
track.

The monotonicity floor is applied first and wins lexicographically: a causal cap
that fell below the floor is raised back to it. This ordering is load-bearing --
causality is never enforced at the price of the monotonicity the T-solve
depends on.
"""
@inline function capped_track(t::Track1D{T}, d::T, w::T, m_floor::T, m_cap::T) where {T}
    t = floor_slope(t, one(T), w, m_floor)
    hi = (m_floor > zero(T) && m_cap < m_floor) ? m_floor : m_cap
    return ramp_track(cap_slope(t, w, hi), d, w)
end

"""
    log_sample(b)

The log-space image `g = ln(f)` of a spline sample. Requires `f > 0`; callers
check that first.
"""
@inline function log_sample(b::BsplineEval3{T}) where {T}
    inv = one(T) / b.f
    gx = b.fx * inv
    gu = b.fu * inv
    gy = b.fy * inv
    return BsplineEval3{T}(log(b.f), gx, gu, gy, b.fxx * inv - gx * gx, b.fxu * inv - gx * gu, b.fuu * inv - gu * gu)
end

"""
    exp_sample(g)

The exact inverse of [`log_sample`](@ref). Applied to the tail-evolved log
sample, so the secondary and frozen tracks are mapped back with the evolved
value, as the composition requires.
"""
@inline function exp_sample(g::BsplineEval3{T}) where {T}
    f = exp(g.f)
    return BsplineEval3{T}(
        f, f * g.fx, f * g.fu, f * g.fy,
        f * (g.fxx + g.fx * g.fx), f * (g.fxu + g.fx * g.fu), f * (g.fuu + g.fu * g.fu),
    )
end

"""
    L_slope_cap(L_b, b_cap, shift_hat, inv_c²)

Convert a cap on `b = dln(ε̂)/du` into a cap on L's own asymptotic slope.

With `E = 10^L · inv_c²` and `ε̂ = E - shift_hat`, we have `b = ln10 · L_u · E/ε̂`,
so the cap in L-slope units is `b_cap · ε̂ / (ln10 · E)`. Returns zero -- meaning
no cap -- when `ε̂ ≤ 0`: there is no causal statement to make there, and the
monotonicity floor is then the only meaningful guard.
"""
@inline function L_slope_cap(L_b::T, b_cap::T, shift_hat::T, inv_c²::T) where {T}
    E = exp10(L_b) * inv_c²
    ε = E - shift_hat
    # Written as positive tests so a NaN falls through to "no cap".
    (ε > zero(T) && E > zero(T)) || return zero(T)
    return b_cap * ε / (ln10(T) * E)
end

"""
    xlow_log_ok(g, w, depth)

May a log-space low-density tail be built from this seam sample?

Its job is finiteness, not fidelity: the bound leaves ample headroom above any
physical seam entropy, while a seam value heading for zero sends the log slope
to minus infinity and would otherwise overflow. False for a non-finite slope,
which sends the caller back to the plain linear tail -- `<=` on a NaN is false,
which is exactly the intent.
"""
@inline function xlow_log_ok(g::BsplineEval3{T}, w::T, depth::T) where {T}
    m = phase2_slope(Track1D{T}(g.f, g.fx, g.fxx), -one(T), w)
    a_gx = g.fx < zero(T) ? -g.fx : g.fx
    a_m = m < zero(T) ? -m : m
    return (a_gx > a_m ? a_gx : a_m) * depth <= xlow_log_excursion_max(T)
end

# The axis a tail runs along is always known at the call site, so it is encoded
# in the function rather than in a runtime tag. Each of these splits the sample
# into the tracks the mixed-derivative composition needs, evolves them, and
# reassembles. The input may itself already be the output of the other one --
# that is the corner case.

"""
    apply_x_tail(b, d, spec)

Apply the density-direction tail. The curvature cap is never consulted here:
it is a temperature-direction statement, and no low-density counterpart exists.
`fuu` and `fy` are frozen, there being no third mixed derivative to evolve them
with.
"""
@inline function apply_x_tail(b::BsplineEval3{T}, d::T, spec::TailSpec{T}) where {T}
    f_out = generic_track(Track1D{T}(b.f, b.fx, b.fxx), d, spec.w, spec.m_floor)
    fu_out = generic_track(Track1D{T}(b.fu, b.fxu, zero(T)), d, spec.w, zero(T))
    return BsplineEval3{T}(f_out.f0, f_out.f1, fu_out.f0, b.fy, f_out.f2, fu_out.f1, b.fuu)
end

"""
    apply_u_tail(b, d, spec)

Apply the temperature-direction tail. `fxx` and `fy` are frozen.

The causal cap applies only on the high side and only where a cap was supplied,
which is L's high-temperature primary track alone; every other track takes the
uncapped path bit for bit.
"""
@inline function apply_u_tail(b::BsplineEval3{T}, d::T, spec::TailSpec{T}) where {T}
    f_in = Track1D{T}(b.f, b.fu, b.fuu)
    f_out = if spec.m_cap > zero(T) && d > zero(T)
        capped_track(f_in, d, spec.w, spec.m_floor, spec.m_cap)
    else
        generic_track(f_in, d, spec.w, spec.m_floor)
    end
    fx_out = generic_track(Track1D{T}(b.fx, b.fxu, zero(T)), d, spec.w, zero(T))
    return BsplineEval3{T}(f_out.f0, fx_out.f0, f_out.f1, b.fy, b.fxx, fx_out.f1, f_out.f2)
end

"""
    extended_sample(fld, x, u, y, spec)

Evaluate one fitted spline with its designed extensions.

`x` and `u` must already be clamped into the *extended* box by the caller; the
spec carries the *physical* box. An interior point falls through to a single
plain `bspline_eval3` call with no tail applied at all, so this is exactly
transparent to every interior evaluation -- bit for bit, which is one of the
package's sharpest tests.

Composition order is fixed and load-bearing: the raw sample is always taken at
the seam clamped to the physical box in *both* directions, the temperature tail
is applied first, and the density tail is then applied to that result. Each
log-space tail sits entirely inside its own step, so a corner query maps out of
log space after the first tail and back into it for the second. That costs one
extra rounding there and keeps the two operators independent.
"""
@inline function extended_sample(fld::BsplineView3{S}, x::T, u::T, y::T, spec::ExtSpec{P}) where {S,T,P}
    # The spec carries its own scalar type: the coefficients may be Float64
    # while the query coordinates are Float32 or a dual number.
    x_lo, x_hi = T(spec.x_lo), T(spec.x_hi)
    u_lo, u_hi = T(spec.u_lo), T(spec.u_hi)
    u_below = u < u_lo
    u_above = u > u_hi
    x_below = x < x_lo
    x_above = x > x_hi

    u_seam = u_below ? u_lo : (u_above ? u_hi : u)
    x_seam = x_below ? x_lo : (x_above ? x_hi : x)

    b = bspline_eval3(fld, x_seam, u_seam, y)

    if u_below || u_above
        d = u - (u_below ? u_lo : u_hi)
        if u_above && spec.u_high_log && b.f > zero(T)
            # The log-space entropy tail: the same machinery run on ln(σ). The
            # monotonicity floor transfers as m_floor/σ at the seam.
            m_floor_g = spec.u_m_floor > 0 ? T(spec.u_m_floor) / b.f : zero(T)
            g = apply_u_tail(log_sample(b), d, TailSpec{T}(T(fld.hu), m_floor_g, zero(T)))
            b = exp_sample(g)
        else
            m_cap = if u_above && spec.u_high_b_cap > 0
                L_slope_cap(b.f, T(spec.u_high_b_cap), T(spec.shift_hat), T(spec.inv_c²))
            else
                zero(T)
            end
            b = apply_u_tail(b, d, TailSpec{T}(T(fld.hu), T(spec.u_m_floor), m_cap))
        end
    end

    if x_below || x_above
        d = x - (x_below ? x_lo : x_hi)
        xspec = TailSpec{T}(T(fld.hx), zero(T), zero(T))
        done = false
        if x_below && spec.x_low_log && b.f > zero(T)
            # No monotonicity floor here: x is clamped, never iterated.
            g = log_sample(b)
            if xlow_log_ok(g, T(fld.hx), x_lo - T(spec.x_ext_lo))
                b = exp_sample(apply_x_tail(g, d, xspec))
                done = true
            end
        end
        done || (b = apply_x_tail(b, d, xspec))
    end

    return b
end

"""
    σ_u_high_alpha(σ, x_seam, u_hi, y, m_floor)

The entropy tail's asymptotic log growth rate `dln(σ)/du` -- exactly the slope
the log tail continues with, and the quantity the energy tail's causal clamp is
measured against, since the tail's sound speed is `b/α - 1`.

Returns zero when the log tail is inactive, which disables the clamp. `x_seam`
must already be clamped to the physical density box.
"""
@inline function σ_u_high_alpha(σ::BsplineView3{S}, x_seam::T, u_hi::T, y::T, m_floor::T) where {S,T}
    b = bspline_eval3(σ, x_seam, u_hi, y)
    b.f > zero(T) || return zero(T)
    g = log_sample(b)
    m_floor_g = m_floor > zero(T) ? m_floor / b.f : zero(T)
    t = floor_slope(Track1D{T}(g.f, g.fu, g.fuu), one(T), T(σ.hu), m_floor_g)
    return phase2_slope(t, one(T), T(σ.hu))
end

# ---------------------------------------------------------------------------
# Per-field extension parameters
# ---------------------------------------------------------------------------

"""Extension parameters for the entropy field."""
@inline function σ_ext_spec(v::EOSTableView{T}) where {T}
    return ExtSpec{T}(
        v.x_lo, v.x_hi, v.u_lo, v.u_hi, v.x_ext_lo, v.ext_slope_floor_σ,
        true, true, zero(T), v.shift_hat, v.inv_c²,
    )
end

"""Extension parameters for the energy field, given a causal cap on `b`."""
@inline function L_ext_spec(v::EOSTableView{S}, b_cap::P) where {S,P}
    T = promote_type(S, P)
    return ExtSpec{T}(
        T(v.x_lo), T(v.x_hi), T(v.u_lo), T(v.u_hi), T(v.x_ext_lo), T(v.ext_slope_floor_L),
        false, false, T(b_cap), T(v.shift_hat), T(v.inv_c²),
    )
end

"""
    u_high_b_cap(v, x_use, y)

The causal cap on `b = dln(ε̂)/du` at the high-temperature seam, namely
`(1 + cs²_ext_cap)·α`. Zero when the entropy log tail supplied no growth rate.
"""
@inline function u_high_b_cap(v::EOSTableView{S}, x_use::T, y::T) where {S,T}
    x_seam = clamp(x_use, T(v.x_lo), T(v.x_hi))
    α = σ_u_high_alpha(v.σ, x_seam, T(v.u_hi), y, T(v.ext_slope_floor_σ))
    return α > zero(T) ? (one(T) + T(v.cs²_ext_cap)) * α : zero(T)
end

# ---------------------------------------------------------------------------
# Entropy windows
# ---------------------------------------------------------------------------

"""
    srange(v, ρ★, yₑ)

The *physical* entropy window at fixed `(ρ★, Yₑ)`: the entropy at the coldest
and hottest tabulated temperature. Both coordinates are clamped to the physical
box first.

This is what any entropy clamp must be taken against, because the physical
domain is not rectangular in `(ρ, s)`.
"""
@inline function srange(v::EOSTableView{S}, ρ★::T, yₑ::T) where {S,T}
    x = clamp(safe_log10(ρ★), T(v.x_lo), T(v.x_hi))
    y = clamp(yₑ, T(v.y_lo), T(v.y_hi))
    return SRange{T}(bspline_eval3(v.σ, x, T(v.u_lo), y).f, bspline_eval3(v.σ, x, T(v.u_hi), y).f)
end

"""
    srange_extended(v, ρ★, yₑ)

The entropy window over the *extended* box, used to bracket the solver. Wider
than [`srange`](@ref), and every value in it is still finite, smooth and
monotone, which is what makes it safe to bisect on.
"""
@inline function srange_extended(v::EOSTableView{S}, ρ★::T, yₑ::T) where {S,T}
    x = clamp(safe_log10(ρ★), T(v.x_ext_lo), T(v.x_ext_hi))
    y = clamp(yₑ, T(v.y_lo), T(v.y_hi))
    spec = σ_ext_spec(v)
    return SRange{T}(
        extended_sample(v.σ, x, T(v.u_ext_lo), y, spec).f,
        extended_sample(v.σ, x, T(v.u_ext_hi), y, spec).f,
    )
end

"""
    sigma_extended(v, ρ★, u, yₑ)

The extended entropy field at a given log-temperature. Strictly increasing in
`u`, which is what the T-solve's bracketing relies on.
"""
@inline function sigma_extended(v::EOSTableView{S}, ρ★::T, u::T, yₑ::T) where {S,T}
    x = clamp(safe_log10(ρ★), T(v.x_ext_lo), T(v.x_ext_hi))
    y = clamp(yₑ, T(v.y_lo), T(v.y_hi))
    uu = clamp(u, T(v.u_ext_lo), T(v.u_ext_hi))
    return extended_sample(v.σ, x, uu, y, σ_ext_spec(v)).f
end

"""
    u_high_tail_info(v, ρ★, yₑ)

Recompute the high-temperature seam's clamp arithmetic at one point, exposing
what the extension derives internally. Audit hook; nothing in the run-time path
calls it.
"""
function u_high_tail_info(v::EOSTableView{S}, ρ★::T, yₑ::T) where {S,T}
    x_seam = clamp(safe_log10(ρ★), T(v.x_lo), T(v.x_hi))
    y = clamp(yₑ, T(v.y_lo), T(v.y_hi))

    α = σ_u_high_alpha(v.σ, x_seam, T(v.u_hi), y, T(v.ext_slope_floor_σ))
    b_cap = α > zero(T) ? (one(T) + T(v.cs²_ext_cap)) * α : zero(T)

    Lb = bspline_eval3(v.L, x_seam, T(v.u_hi), y)
    t = floor_slope(Track1D{T}(Lb.f, Lb.fu, Lb.fuu), one(T), T(v.L.hu), T(v.ext_slope_floor_L))
    m_L_raw = phase2_slope(t, one(T), T(v.L.hu))
    m_L_cap = b_cap > zero(T) ? L_slope_cap(Lb.f, b_cap, T(v.shift_hat), T(v.inv_c²)) : zero(T)

    Eh = exp10(Lb.f) * T(v.inv_c²)
    ε = Eh - T(v.shift_hat)
    b_raw = ε > zero(T) ? ln10(T) * m_L_raw * Eh / ε : zero(T)

    floor_wins = m_L_cap > zero(T) && v.ext_slope_floor_L > 0 && m_L_cap < T(v.ext_slope_floor_L)
    hi = floor_wins ? T(v.ext_slope_floor_L) : m_L_cap
    clamped = m_L_cap > zero(T) && m_L_raw > hi
    return UHighTailInfo{T}(α, b_raw, b_cap, m_L_raw, m_L_cap, clamped, floor_wins)
end

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

"""
    evaluate(v, ρ★, s, yₑ, u_guess)

Evaluate the potential `U(ρ★, s, Yₑ)` and the derivatives a solver needs.

`u_guess` is the previous call's `u_solved`, or `NaN` for no guess. Warm-start
state is threaded explicitly rather than cached, so this is pure and re-entrant
and safe to call in parallel across grid points.

The density treatment is deliberately asymmetric. Below the table a designed
extension applies and the point is merely flagged; above it the point keeps a
hard out-of-bounds flag, because a converged state there is invalid outright --
the tail exists only so the iteration stays finite rather than to make the
answer meaningful.
"""
function evaluate(v::EOSTableView{S}, ρ★::T, s::T, yₑ::T, u_guess::T) where {S,T}
    flags = UInt32(0)

    # Locate ρ against the physical and extended boxes.
    x = safe_log10(ρ★)
    x_use = if x < v.x_lo
        flags |= FLAG_EXT_ρ_LOW
        x < v.x_ext_lo ? T(v.x_ext_lo) : x
    elseif x > v.x_hi
        flags |= FLAG_OOB_ρ_HIGH
        x > v.x_ext_hi ? T(v.x_ext_hi) : x
    else
        x
    end

    # Yₑ has no extension: it is hard-clamped and flagged.
    y = yₑ
    if y < v.y_lo
        y = T(v.y_lo)
        flags |= FLAG_CLAMP_YE
    elseif y > v.y_hi
        y = T(v.y_hi)
        flags |= FLAG_CLAMP_YE
    end

    # Solve σ_ext(x_use, u, y) = s for u on the extended bracket. The u-tail's
    # slope floor keeps σ_ext strictly increasing in u, so the bracket is
    # globally invertible and the safeguarded iteration cannot fail.
    spec = σ_ext_spec(v)
    σ_lo = extended_sample(v.σ, x_use, T(v.u_ext_lo), y, spec)
    σ_hi = extended_sample(v.σ, x_use, T(v.u_ext_hi), y, spec)

    u = zero(T)
    iters = Int32(0)
    if s <= σ_lo.f
        u = T(v.u_ext_lo)          # below even the extended bracket
    elseif s >= σ_hi.f
        u = T(v.u_ext_hi)
    else
        lo = T(v.u_ext_lo)
        hi = T(v.u_ext_hi)
        uu = if !isnan(u_guess)
            clamp(u_guess, lo, hi)
        else
            # A secant estimate from the endpoints beats a plain midpoint start.
            clamp(lo + (hi - lo) * (s - σ_lo.f) / (σ_hi.f - σ_lo.f), lo, hi)
        end

        s_scale = abs(s) > one(T) ? abs(s) : one(T)
        converged = false
        while iters < v.max_iter
            iters += one(Int32)
            e = extended_sample(v.σ, x_use, uu, y, spec)
            g = e.f - s

            # Maintain the bracket by sign; σ_ext increases with u.
            g < zero(T) ? (lo = uu) : (hi = uu)

            if abs(g) <= tsolve_residual_tol(T) * s_scale
                converged = true
                break
            end

            # Newton, accepted only if it stays inside the bracket and the
            # slope is positive there; bisect otherwise.
            u_next = zero(T)
            newton_ok = false
            if e.fu > zero(T)
                u_next = uu - g / e.fu
                newton_ok = lo < u_next < hi
            end
            newton_ok || (u_next = T(0.5) * (lo + hi))

            du = u_next - uu
            uu = u_next

            u_scale = abs(uu) > one(T) ? abs(uu) : one(T)
            if abs(du) <= tsolve_step_tol(T) * u_scale
                converged = true
                break
            end
        end
        converged || (flags |= FLAG_MAXITER)
        u = uu
    end

    # The extension flags describe where the solve *landed*, not where the
    # input was: with the extensions in place the solve almost always finds a
    # genuine root out there rather than pinning at the box edge, so these mark
    # "the answer is in the designed extension zone" and not "the input was
    # clamped".
    if u < v.u_lo
        flags |= FLAG_EXT_S_LOW
    elseif u > v.u_hi
        flags |= FLAG_EXT_S_HIGH
    end

    return eval_at(v, x_use, u, y, flags, iters)
end

evaluate(v::EOSTableView{S}, ρ★, s, yₑ, u_guess) where {S} =
    evaluate(v, promote(float(ρ★), float(s), float(yₑ), float(u_guess))...)

"""
    eval_at(v, x_use, u, y, flags, iters)

The chain rule at an already-located point, in log-density `x_use` and
log-temperature `u`.

Three stages: undo the log₁₀ energy fit analytically; convert the log-axis
partials to physical ones; then invert `s ↔ T` by the implicit function
theorem, which is what turns the table's `ε(ρ,T,Yₑ)` into a genuine potential
`U(ρ,s,Yₑ)`. A final rescaling by κ re-zeroes the energy so that `U ≥ 0`.
"""
function eval_at(v::EOSTableView{S}, x_use::T, u::T, y::T, flags::UInt32, iters::Int32) where {S,T}
    σ = extended_sample(v.σ, x_use, u, y, σ_ext_spec(v))
    # Above the seam L's tail slope is held to the causal bound, measured
    # against the entropy tail's own growth rate. One extra sample, on this
    # path only.
    b_cap = u > v.u_hi ? u_high_b_cap(v, x_use, y) : zero(T)
    Lv = extended_sample(v.L, x_use, u, y, L_ext_spec(v, b_cap))

    λ = ln10(T)
    ρ_eff = exp10(x_use)
    T̂_t = T(v.conv_t) * exp10(u)

    # Stage A, part 1: undo the log₁₀ fit. E = 10^L/c² = ε̂ + shift.
    Eh = exp10(Lv.f) * T(v.inv_c²)
    ε̂ = Eh - T(v.shift_hat)
    ε_x = λ * Eh * Lv.fx
    ε_u = λ * Eh * Lv.fu
    ε_y = λ * Eh * Lv.fy
    ε_xx = λ * Eh * (Lv.fxx + λ * Lv.fx * Lv.fx)
    ε_uu = λ * Eh * (Lv.fuu + λ * Lv.fu * Lv.fu)
    ε_xu = λ * Eh * (Lv.fxu + λ * Lv.fx * Lv.fu)

    # Stage A, part 2: log-axis partials to physical ones.
    inv_lρ = one(T) / (λ * ρ_eff)
    inv_lρ² = inv_lρ * inv_lρ
    inv_lT = one(T) / (λ * T̂_t)
    inv_lT² = inv_lT * inv_lT
    inv_lρlT = one(T) / (λ * λ * ρ_eff * T̂_t)

    σ_ρ = σ.fx * inv_lρ
    σ_ρρ = (σ.fxx - λ * σ.fx) * inv_lρ²
    σ_T = σ.fu * inv_lT
    σ_TT = (σ.fuu - λ * σ.fu) * inv_lT²
    σ_ρT = σ.fxu * inv_lρlT
    σ_y = σ.fy

    ε_ρ = ε_x * inv_lρ
    ε_ρρ = (ε_xx - λ * ε_x) * inv_lρ²
    ε_T = ε_u * inv_lT
    ε_TT = (ε_uu - λ * ε_u) * inv_lT²
    ε_ρT = ε_xu * inv_lρlT

    # Stage B: the implicit function theorem, inverting σ(ρ,T,Yₑ) = s for T.
    T_s = one(T) / σ_T
    T_ρ = -σ_ρ / σ_T
    T_ρs = -(σ_ρT + σ_TT * T_ρ) / (σ_T * σ_T)
    T_ρρ = -(σ_ρρ + T(2) * σ_ρT * T_ρ + σ_TT * T_ρ * T_ρ) / σ_T

    Uh = ε̂
    Uh_s = ε_T * T_s
    Uh_ρ = ε_ρ + ε_T * T_ρ
    Uh_ρs = ε_ρT * T_s + ε_TT * T_ρ * T_s + ε_T * T_ρs
    Uh_ρρ = ε_ρρ + T(2) * ε_ρT * T_ρ + ε_TT * T_ρ * T_ρ + ε_T * T_ρρ
    μh = ε_y + ε_T * (-σ_y / σ_T)

    # Stage C: the κ re-zeroing, which is exact rather than an approximation --
    # per baryon, m_B★(1 + U) = m_B(1 + ε).
    κ = T(v.κ)
    U = (one(T) + Uh) / κ - one(T)
    U_ρ = Uh_ρ / κ
    U_s = Uh_s / κ
    U_ρρ = Uh_ρρ / κ
    U_ρs = Uh_ρs / κ
    μ̃ = μh / κ

    p = ρ_eff * ρ_eff * U_ρ
    h = one(T) + U + p / ρ_eff
    cs² = (T(2) * ρ_eff * U_ρ + ρ_eff * ρ_eff * U_ρρ) / h

    return EOSPoint{T}(U, U_ρ, U_s, U_ρρ, U_ρs, U_s, p, h, cs², exp10(u), μ̃, u, iters, flags)
end

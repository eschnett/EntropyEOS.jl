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
#
# Every iterate stays inside the EOS's density box, ρ = D/cosh(w) ≤ ρ_max, so
# w ≥ w_floor(D). The inner root is unique only where cs²·v² < 1, and above the
# box the extension is acausal; a search allowed to sweep up to ρ = D found
# extra roots there for fast states near the top of the box.

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

Solver knobs. Every default was measured at `Float64`, and the `Float32`
tolerance in this port's own accuracy study; see `defs.jl`, and the "Precision
and GPUs" page of the documentation for what Float32 costs.
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
    seed_passes=5,
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
`sr_ext` is the extended entropy window at `ρ`, which the Newton clamps its step
to.
"""
struct Residuals{T<:Real}
    s::T
    w::T
    ρ::T
    coshw::T
    pt::EOSPoint{T}
    sr_ext::SRange{T}
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

"""
    Con2PrimStage

What [`con2prim_start`](@ref) hands to [`con2prim_finish`](@ref): an iterate as
the `(s, w)` and the warm start `u` it was evaluated with, which reproduce it
bit for bit, rather than the iterate itself, which is several times larger. It
is the last Newton iterate if the Newton converged, and the best one otherwise.
"""
struct Con2PrimStage{T<:Real}
    s::T
    w::T
    u::T
    iters_newton::Int32
    converged::Bool
end

"""
    is_done(stage)

Is the solve complete after its first stage, so that the output
[`con2prim_start`](@ref) or [`con2prim_safe_start`](@ref) returned with it is
the answer?
"""
@inline is_done(st::Con2PrimStage) = st.converged

"""A manufactured cold start: an exact seed derived without any prior iterate."""
struct ColdSeed{T<:Real}
    s::T
    w::T
    u::T
    ρ::T
    z::T
end

"""
    f1_converged(r, tol)

Has the momentum residual converged?

`tol` bounds the *normalized* residual `tanh(w) - V`, not `f₁` itself. That
matters: `f₁` carries a factor of `cosh(w)`, so near the rapidity cap its own
double-precision floor sits well above any absolute tolerance, and a test
written against `f₁` directly could never be satisfied there.

Written as `|f₁| ≤ tol·cosh(w)` rather than dividing, so the hot path does one
multiply; `cosh(w) ≥ 1` always, so this is never the tighter test.
"""
@inline f1_converged(r::Residuals{T}, tol::T) where {T} = abs(r.f1) <= tol * r.coshw

"""
    scaled_norm(r)

The residual pair's size, with the momentum residual normalized to the energy
residual's scale.

Used wherever the two are *compared* or minimized together. A plain Euclidean
norm would be dominated by the `cosh(w)`-sized component and would happily
trade a real improvement in the energy residual for a cosmetic one in momentum.
"""
@inline function scaled_norm(r::Residuals{T}) where {T}
    n1 = abs(r.f1) / r.coshw
    n2 = abs(r.f2)
    return n1 > n2 ? n1 : n2
end

"""
    w_floor(eos, D, w_max)

The least rapidity at which a state with rest-mass density `D` lies inside the
EOS's density box, `ρ = D/cosh(w) ≤ ρ_max`: zero for `D ≤ ρ_max`, and never above
`w_max`.

Every iterate of the solver keeps `w ≥ w_floor`, so it never evaluates the EOS
above its box. There the table is extended only so that an evaluation stays
finite, and acausally; a search sweeping up to `ρ = D` found extra roots of the
inner solve there for fast states near the top of the box, although the state
inside it was unique (CODE.md, "Deliberate deviations from the C++").

The box ends at `ρ_max` widened by the projection's `face_slack`: a state that
`con2prim_safe` writes onto that face must solve again, and `acosh` of its
`D/ρ_max` can round to a hair above its own rapidity.
"""
@inline function w_floor(eos::AbstractEOS, D::T, w_max::T) where {T}
    ρ_max = ρ_ceiling(eos, T)
    D > ρ_max || return zero(T)
    w = acosh(D / ρ_max)
    return w < w_max ? w : w_max
end

"""The top of the density box as [`w_floor`](@ref) sees it, widened by `face_slack`."""
@inline ρ_ceiling(eos::AbstractEOS, ::Type{T}) where {T} = exp10(T(last(logρ_bounds(eos)))) * (one(T) + face_slack(T))

"""
    above_floor(eos, D, w)

Is `w ≥ 0` at or above [`w_floor`](@ref)`(eos, D, ⋅)`? A sufficient test, not a
necessary one, that needs no `acosh`.

A warm start almost always passes, and then needs the floor only if the Newton
steps. About 8% of states have `D > ρ_max`, which on a GPU puts one in nearly
every warp; skipping their `acosh` made the first stage of a warm `con2prim`
3–5% faster on an H200 (CODE.md, "Discipline in `core/`").
"""
@inline function above_floor(eos::AbstractEOS, D::T, w::T) where {T}
    ρ_max = ρ_ceiling(eos, T)
    D > ρ_max || return true
    # cosh's Taylor series to w⁸. Every term is positive, so the sum is a lower
    # bound, within 11% of cosh up to w = 6.
    w² = w * w
    c = one(T) + w² * (inv(T(2)) + w² * (inv(T(24)) + w² * (inv(T(720)) + w² * inv(T(40320)))))
    return D <= ρ_max * c
end

"""
    clamp_w(w, w_lo, w_max)

A Newton step's rapidity, held in `[w_lo, w_max]`. With the bracket open at rest
(`w_lo = 0`) a negative step is reflected to `tiny_w` rather than onto zero; on
the `ρ_max` face it stops on the face.
"""
@inline clamp_w(w::T, w_lo::T, w_max::T) where {T} =
    w < w_lo ? (w_lo > zero(T) ? w_lo : tiny_w(T)) : (w > w_max ? w_max : w)

"""
    residuals(eos, D, τ, yₑ, S_par, S_perp, B², s, w, u_prev, τ_floor_rel)

Evaluate both residuals and the analytic Jacobian at one trial `(s, w)`.

`u_prev` threads the EOS temperature solve's warm start between calls, which is
what makes the whole iteration cheap.
"""
function residuals(eos::AbstractEOS{S}, D::T, τ::T, yₑ::T, S_par::T, S_perp::T, B²::T, s::T, w::T,
                   u_prev::T, τ_floor_rel::T) where {S,T}
    coshw = cosh(w)
    sinhw = sinh(w)
    tanhw = tanh(w)
    half_sinh = sinh(T(0.5) * w)

    ρ = D / coshw
    pt, sr_ext = evaluate_windowed(eos, ρ, s, yₑ, u_prev)

    h, U, p, cs² = pt.h, pt.U, pt.p, pt.cs²
    U_ρ, U_s, U_ρs = pt.U_ρ, pt.U_s, pt.U_ρs

    z = D * h * coshw
    zpB² = z + B²
    v_par = S_par / z
    v_perp = S_perp / zpB²
    V² = v_par * v_par + v_perp * v_perp
    V = safe_sqrt(V²)

    f1 = sinhw - coshw * V

    # The same cancellation-free construction prim2con uses, evaluated at the
    # trial point rather than at the truth. The two must stay identical.
    τ_model = T(2) * D * half_sinh * half_sinh + ρ * U * coshw * coshw + p * sinhw * sinhw +
              T(0.5) * B² * (one(T) + V²) - T(0.5) * B² * v_par * v_par
    denom = τ > τ_floor_rel * D ? τ : τ_floor_rel * D
    f2 = (τ_model - τ) / denom

    ρ_w = -ρ * tanhw
    z_w = z * (one(T) - cs²) * tanhw
    z_s = D * coshw * (U_s + ρ * U_ρs)
    p_w = -ρ * h * cs² * tanhw
    p_s = ρ * ρ * U_ρs

    dvpar_dw = -(v_par / z) * z_w
    dvpar_ds = -(v_par / z) * z_s
    dvperp_dw = -(v_perp / zpB²) * z_w
    dvperp_ds = -(v_perp / zpB²) * z_s

    # Carry V·dV rather than dV: the numerator is second order in V, so the
    # V → 0 limit is finite and correct without a special case.
    VdV_dw = v_par * dvpar_dw + v_perp * dvperp_dw
    VdV_ds = v_par * dvpar_ds + v_perp * dvperp_ds
    Vfloor = V > tiny_denom(T) ? V : tiny_denom(T)
    dV_dw = VdV_dw / Vfloor
    dV_ds = VdV_ds / Vfloor

    df1_dw = coshw - sinhw * V - coshw * dV_dw
    df1_ds = -coshw * dV_ds

    # Term by term, remembering that ρ depends on w alone.
    dτ_dw = D * sinhw + ρ_w * (U + ρ * U_ρ) * coshw * coshw + T(2) * ρ * U * coshw * sinhw +
            p_w * sinhw * sinhw + T(2) * p * sinhw * coshw + B² * VdV_dw - B² * v_par * dvpar_dw
    dτ_ds = ρ * U_s * coshw * coshw + p_s * sinhw * sinhw + B² * VdV_ds - B² * v_par * dvpar_ds

    return Residuals{T}(s, w, ρ, coshw, pt, sr_ext, z, v_par, v_perp, V, f1, f2, df1_ds, df1_dw,
                        dτ_ds / denom, dτ_dw / denom)
end

# Promoting fallback, so a caller may differentiate with respect to one
# argument while the rest stay plain floats.
residuals(eos::AbstractEOS{S}, D, τ, yₑ, S_par, S_perp, B², s, w, u_prev, τ_floor_rel) where {S} =
    residuals(eos, promote(D, τ, yₑ, S_par, S_perp, B², s, w, u_prev, τ_floor_rel)...)

"""
    inner_solve_w(eos, D, τ, yₑ, S_par, S_perp, B², s, w_lo, w_max, τ_floor_rel, w_start, u_prev,
                  max_iter_1d, tol)

Solve `f₁(w; s) = 0` on `[w_lo, w_max]`, returning `(residuals, iterations)`.
`w_lo` is [`w_floor`](@ref), the rapidity below which the state would lie above
the density box.

At a root of `f₁` the crossing rises wherever `cs²·v² < 1`, so inside a causal
box the root is unique and a bracketed safeguarded Newton finds it. That is the
reason the formulation uses rapidity in the first place, and the reason the
bracket ends at the box: above it the extension is acausal.

The root may lie at or beyond the `ρ_max` face, and then `f₁ ≥ 0` on the face.
The face is tested once, when the iteration would otherwise bisect towards it,
and if so the solve returns the face rather than bisect its budget away.
"""
function inner_solve_w(eos::AbstractEOS{S}, D::T, τ::T, yₑ::T, S_par::T, S_perp::T, B²::T, s::T,
                       w_lo::T, w_max::T, τ_floor_rel::T, w_start::T, u_prev::T, max_iter_1d::Integer,
                       tol::T) where {S,T}
    lo = w_lo
    hi = w_max
    w = clamp(w_start, lo, hi)
    r = residuals(eos, D, τ, yₑ, S_par, S_perp, B², s, w, u_prev, τ_floor_rel)

    # The face has not been tested yet; at rest there is none.
    face_open = w_lo > zero(T)
    iters = Int32(0)
    while !f1_converged(r, tol) && iters < max_iter_1d
        iters += one(Int32)
        # f₁ increases with w, so its sign says which way the root lies.
        r.f1 < zero(T) ? (lo = w) : (hi = w)
        # On the face with the root at or beyond it: nothing inside the box to find.
        (w == w_lo && w_lo > zero(T) && r.f1 >= zero(T)) && break

        w_next = zero(T)
        newton_ok = false
        if r.df1_dw != zero(T) && isfinite(r.df1_dw)
            w_next = w - r.f1 / r.df1_dw
            newton_ok = isfinite(w_next) && lo < w_next < hi
        end
        if !newton_ok
            if face_open && lo == w_lo && r.f1 >= zero(T)
                # Evaluated through the loop's own call, not a second one: a
                # GPU compiler inlines every call site.
                w_next = w_lo
                face_open = false
            else
                w_next = T(0.5) * (lo + hi)
            end
        end

        w = w_next
        r = residuals(eos, D, τ, yₑ, S_par, S_perp, B², s, w, r.pt.u_solved, τ_floor_rel)

        # Once the bracket collapses to adjacent floats no further iteration can
        # improve anything.
        lo < hi || break
    end
    return r, iters
end

"""
    ScanScratch{N,T}()

The bracket scan's stack scratch: an `MVector{N,T}` whose bounds check does not
capture the buffer. `MVector`'s own check hands the buffer to `BoundsError`;
under `--check-bounds=yes`, where `@inbounds` is ignored, that lets it escape
and puts it on the heap. Checking against `Base.OneTo(N)` instead puts only the
range in the error. Without bounds checks the two are the same code.
"""
struct ScanScratch{N,T}
    v::MVector{N,T}
end
@inline ScanScratch{N,T}() where {N,T} = ScanScratch{N,T}(MVector{N,T}(undef))

Base.@propagate_inbounds function Base.getindex(a::ScanScratch{N,T}, i::Int) where {N,T}
    @boundscheck checkbounds(Base.OneTo(N), i)
    v = a.v
    return GC.@preserve v unsafe_load(Base.unsafe_convert(Ptr{T}, pointer_from_objref(v)), i)
end

Base.@propagate_inbounds function Base.setindex!(a::ScanScratch{N,T}, x, i::Int) where {N,T}
    @boundscheck checkbounds(Base.OneTo(N), i)
    v = a.v
    GC.@preserve v unsafe_store!(Base.unsafe_convert(Ptr{T}, pointer_from_objref(v)), convert(T, x), i)
    return a
end

"""
    sort_scan!(s, g, n)

Insertion sort of the scan's `(s, g)` pairs by `s`. Hand-rolled rather than
delegated, to keep the fallback path allocation-free.
"""
@inline function sort_scan!(s, g, n::Int)
    @inbounds for i in 2:n
        si, gi = s[i], g[i]
        j = i - 1
        while j >= 1 && s[j] > si
            s[j + 1] = s[j]
            g[j + 1] = g[j]
            j -= 1
        end
        s[j + 1] = si
        g[j + 1] = gi
    end
    return nothing
end

"""
    bracket_scan(eos, D, τ, yₑ, S_par, S_perp, B², w_lo, w_max, τ_floor_rel, n_scan, s_in, w_in,
                 w_seed0, u_seed0, max_iter_1d, tol; nmax = Val(BRACKET_SCAN_MAX))

Find a sign-changing interval in `s` for the outer solve.

Endpoint signs are not enough: the outer function is non-monotone near the
extension seams, and the extended entropy bracket can span ten decades. So the
search is a multi-point scan, half of it uniform across the physical range and
half a geometric ladder of *relative* offsets around `s_in`.

The split matters and was measured. The global half finds a single monotone
root whatever `s_in` is, which is what a crude cold seed needs. The local half
concentrates resolution near a good `s_in`, which is what a Newton that stalled
near the root needs -- and "near" there is a relative statement, so a ladder
sized to the local entropy *span* would be orders of magnitude too coarse.

Every candidate gets a full-precision inner solve. A loosened scan-only
tolerance was tried in the C++ and reverted: near a genuine near-degenerate
root an under-converged inner solve can flip the sign of the outer residual and
manufacture a false bracket that masks the real one.

`nmax` sizes the stack scratch. It is a `Val` so a GPU caller can shrink the
per-thread local memory without touching the algorithm.

The scan stays inside the density box, `w ≥ w_lo`, and starts at its least
entropy. If even that candidate's root lies beyond the `ρ_max` face, so does
every other's: on the face `φ = tanh w − V` rises with `s`, since `z` does.
The scan then stops there, unbracketed, rather than solve the rest.
"""
function bracket_scan(eos::AbstractEOS{S}, D::T, τ::T, yₑ::T, S_par::T, S_perp::T, B²::T, w_lo::T, w_max::T,
                      τ_floor_rel::T, n_scan::Integer, s_in::T, w_in::T, w_seed0::T, u_seed0::T,
                      max_iter_1d::Integer, tol::T, ::Val{NMAX}=Val(BRACKET_SCAN_MAX)) where {S,T,NMAX}
    n = clamp(Int(n_scan), 4, NMAX)

    ρ_a = D / cosh(w_lo)       # at rest, or on the ρ_max face
    ρ_b = D / cosh(w_max)      # at the rapidity cap

    ext_a = srange_extended(eos, ρ_a, yₑ)
    ext_b = srange_extended(eos, ρ_b, yₑ)
    s_ext_lo = min(ext_a.s_min, ext_b.s_min)
    s_ext_hi = max(ext_a.s_max, ext_b.s_max)

    phys_a = srange(eos, ρ_a, yₑ)
    phys_b = srange(eos, ρ_b, yₑ)
    s_phys_lo = min(phys_a.s_min, phys_b.s_min)
    s_phys_hi = max(phys_a.s_max, phys_b.s_max)

    # Stack scratch: local and non-escaping, so it is promoted out of the heap.
    s_cand = ScanScratch{NMAX,T}()
    g_cand = ScanScratch{NMAX,T}()
    nc = 0
    @inbounds begin
        nc += 1
        s_cand[nc] = s_ext_lo
        nc += 1
        s_cand[nc] = s_phys_lo

        n_interior = n - 4
        n_global = (n_interior + 1) ÷ 2
        n_local = n_interior - n_global

        for i in 0:(n_global - 1)
            frac = n_global > 1 ? T(i) / T(n_global - 1) : T(0.5)
            nc += 1
            s_cand[nc] = s_phys_lo + frac * (s_phys_hi - s_phys_lo)
        end

        # The ladder is emitted in ± pairs so both sides reach the full span.
        # A one-sided alternating ladder halves each side's reach and was
        # measured worse: coverage beats resolution here, because the outer
        # solve resolves a wide bracket cheaply while a missed root is an
        # outright failure.
        if s_in > zero(T) && isfinite(s_in)
            n_pair = (n_local + 1) ÷ 2
            ratio = n_pair > 1 ? (scan_delta_max(T) / scan_delta_min(T))^(one(T) / T(n_pair - 1)) : one(T)
            δ = scan_delta_min(T)
            emitted = 0
            for _ in 1:n_pair
                emitted >= n_local && break
                nc += 1
                s_cand[nc] = clamp(s_in * (one(T) + δ), s_ext_lo, s_ext_hi)
                emitted += 1
                if emitted < n_local
                    nc += 1
                    s_cand[nc] = clamp(s_in * (one(T) - δ), s_ext_lo, s_ext_hi)
                    emitted += 1
                end
                δ *= ratio
            end
        else
            # A multiplicative offset degenerates at or below zero entropy,
            # which is legitimate on tables whose entropy axis reaches zero.
            # Fall back to a window sized by the local physical span.
            ρ_in = D / cosh(clamp(w_in, w_lo, w_max))
            sr_in = srange(eos, ρ_in, yₑ)
            half_width = T(0.6) * (sr_in.s_max - sr_in.s_min)
            lo = clamp(s_in - half_width, s_ext_lo, s_ext_hi)
            hi = clamp(s_in + half_width, s_ext_lo, s_ext_hi)
            for i in 0:(n_local - 1)
                frac = n_local > 1 ? T(i) / T(n_local - 1) : T(0.5)
                nc += 1
                s_cand[nc] = lo + frac * (hi - lo)
            end
        end

        nc += 1
        s_cand[nc] = s_phys_hi
        nc += 1
        s_cand[nc] = s_ext_hi

        w_seed, u_seed = w_seed0, u_seed0
        for i in 1:nc
            r, _ = inner_solve_w(eos, D, τ, yₑ, S_par, S_perp, B², s_cand[i], w_lo, w_max, τ_floor_rel,
                                 w_seed, u_seed, max_iter_1d, tol)
            g_cand[i] = r.f2
            w_seed = r.w
            u_seed = r.pt.u_solved
            # Candidate 1 is s_ext_lo, the least entropy of all.
            if i == 1 && w_lo > zero(T) && r.w == w_lo && r.f1 >= zero(T) && !f1_converged(r, tol)
                return BracketScanResult{T}(false, s_cand[1], s_cand[1], s_cand[1])
            end
        end

        sort_scan!(s_cand, g_cand, nc)

        # Take the first sign-changing pair whose midpoint is closest to s_in:
        # simple and deterministic.
        #
        # A pair of equal entropies brackets nothing, but occurs routinely:
        # the uniform half's endpoints duplicate s_phys_lo and s_phys_hi, and
        # the ladder clamps onto the extended edges. When the root sits on such
        # a candidate -- a state on an edge of the physical window -- the two
        # copies' solves differ only by their warm starts, can disagree in
        # sign, and the pair's midpoint is then exactly s_in, so it would win.
        pick = 0
        best_dist = zero(T)
        for i in 1:(nc - 1)
            s_cand[i] < s_cand[i + 1] || continue
            glo, ghi = g_cand[i], g_cand[i + 1]
            ((glo <= 0 && ghi >= 0) || (glo >= 0 && ghi <= 0)) || continue
            dist = abs(T(0.5) * (s_cand[i] + s_cand[i + 1]) - s_in)
            if pick == 0 || dist < best_dist
                pick = i
                best_dist = dist
            end
        end

        if pick > 0
            return BracketScanResult{T}(true, s_cand[pick], s_cand[pick + 1], s_cand[pick])
        end
        # No sign change anywhere: report the closest-to-zero point so the
        # caller's failure still carries the scan's evidence.
        best_i = 1
        best_g = abs(g_cand[1])
        for i in 2:nc
            ag = abs(g_cand[i])
            if ag < best_g
                best_g = ag
                best_i = i
            end
        end
        return BracketScanResult{T}(false, s_cand[1], s_cand[1], s_cand[best_i])
    end
end

"""
    seed_z_solve(D, E, p, S_perp, B², n_iter)

Solve the energy relation for the total enthalpy density, given a lagged
pressure. Touches no EOS at all -- pure arithmetic.

With `q = z + B²` the energy relation and the perpendicular momentum
projection close on each other without reference to rapidity, giving the cubic
`H(q) = q - A + B²S_⊥²/(2q²) = 0` with `A = E + p + B²/2`. `H` has a single
minimum at `∛(B²S_⊥²)` and increases above it, and the physical root always
lies on that increasing branch, so both bracket ends are exact physical bounds
rather than tuned constants. With no field this degenerates to `q = E + p`,
the exact hydrodynamic result, recovered rather than special-cased.
"""
function seed_z_solve(D::T, E::T, p::T, S_perp::T, B²::T, n_iter::Integer) where {T}
    A = E + p + T(0.5) * B²

    # Formed factor-wise so an intermediate product cannot overflow at the
    # magnitudes real tables reach.
    q_branch = cbrt(B²) * cbrt(S_perp) * cbrt(S_perp)
    lo = D + B²
    q_branch > lo && (lo = q_branch)
    hi = A > lo ? A : lo

    # Start at the upper bound, not the midpoint. With no field the residual is
    # exactly q - A, so the first point *is* the root and the test below exits
    # with the exact answer; from the midpoint every Newton step would be
    # rejected for landing on the bracket endpoint, leaving the result to
    # bisection alone. With a field, A overestimates the root, which is the
    # side Newton descends from monotonically.
    q = hi
    if hi > lo
        for _ in 1:n_iter
            # B²S⊥²/(2q²) is carried as ½B²(S⊥/q)², where S⊥/q is a velocity. The
            # textbook form squares and cubes conserved quantities, which reach
            # 1e20 and overflow Float32; the resulting Inf drove the bracket to
            # its lower end and the seed to w_max on every magnetized state.
            vq = S_perp / q
            m = T(0.5) * B² * vq * vq
            H = q - A + m
            H < zero(T) ? (lo = q) : (hi = q)
            abs(H) <= seed_z_tol(T) * A && break
            dH = one(T) - T(2) * m / q
            q_next = zero(T)
            newton_ok = false
            if dH > zero(T)
                q_next = q - H / dH
                newton_ok = isfinite(q_next) && lo < q_next < hi
            end
            newton_ok || (q_next = T(0.5) * (lo + hi))
            (q_next == q || !(lo < hi)) && break
            q = q_next
        end
    end

    z = q - B²
    return z > D ? z : D     # z = D·h·cosh(w) ≥ D exactly
end

"""
    cold_seed(eos, D, τ, yₑ, S_par, S_perp, B², w_lo, w_max, n_pass, n_s_iter)

Manufacture a starting point with no prior iterate.

Every step is an exact relation and the only lagged quantity is the pressure,
which starts at zero: energy gives the enthalpy density, momentum gives the
rapidity, the energy identity gives the internal energy, and the guaranteed
monotone entropy solve gives the entropy. Being field-aware is what makes this
work on magnetized states, where a hydrodynamic seed is badly wrong. Like every
iterate, each pass keeps its rapidity in `[w_lo, w_max]`, inside the density box.
"""
function cold_seed(eos::AbstractEOS{S}, D::T, τ::T, yₑ::T, S_par::T, S_perp::T, B²::T, w_lo::T, w_max::T,
                   n_pass::Integer, n_s_iter::Integer) where {S,T}
    E = τ + D
    S_perp_pos = S_perp > zero(T) ? S_perp : zero(T)
    V_max = tanh(w_max)

    npass = clamp(Int(n_pass), 1, 8)
    nsit = clamp(Int(n_s_iter), 2, 40)

    p = zero(T)
    u_prev = T(NaN)
    out = ColdSeed{T}(zero(T), zero(T), T(NaN), D, D)

    for _ in 1:npass
        z = seed_z_solve(D, E, p, S_perp_pos, B², SEED_SCALAR_ITERS)
        q = z + B²

        v_par = S_par / z
        v_perp = S_perp_pos / q
        V = safe_sqrt(v_par * v_par + v_perp * v_perp)
        isfinite(V) || (V = zero(T))
        # V at or above the cap means the lagged pressure is not yet consistent
        # with a state inside the rapidity bound; report the cap and let the
        # next pass fix it.
        w = V < V_max ? clamp(atanh(V), w_lo, w_max) : w_max

        coshw = cosh(w)
        sinhw = sinh(w)
        half_sinh = sinh(T(0.5) * w)
        ρ = D / coshw

        # Re-express the velocity at the rapidity actually adopted. A no-op on
        # the ordinary path, but it keeps the magnetic terms below consistent
        # on the capped branch, where V itself may exceed one.
        Vc = tanh(w)
        V > zero(T) && (v_par *= Vc / V)

        ρW² = D * coshw
        ε = (τ - T(2) * D * half_sinh * half_sinh - p * sinhw * sinhw - T(0.5) * B² * (one(T) + Vc * Vc) +
             T(0.5) * B² * v_par * v_par) / ρW²
        ε > zero(T) || (ε = zero(T))     # also catches a non-finite quotient

        s, pt = entropy_from_energy(eos, ρ, yₑ, ε, u_prev, nsit)
        u_prev = pt.u_solved
        out = ColdSeed{T}(s, w, pt.u_solved, ρ, z)

        p = (isfinite(pt.p) && pt.p > zero(T)) ? pt.p : zero(T)
    end
    return out
end

"""Assemble the shared output fields from a converged or best-effort iterate."""
@inline function fill_out(r::Residuals{T}, yₑ::T, result::C2PResult.T, iters_newton::Int32,
                          iters_fallback::Int32) where {T}
    return Con2PrimOut{T}(r.ρ, r.s, yₑ, r.w, cosh(r.w), r.v_par, r.v_perp, r.pt, result,
                          iters_newton, iters_fallback, r.pt.flags)
end

"""The placeholder output of a first stage that is not done: no EOS call, all NaN."""
@inline function nan_c2p_out(::Type{T}) where {T}
    n = T(NaN)
    pt = EOSPoint{T}(n, n, n, n, n, n, n, n, n, n, n, n, Int32(0), UInt32(0))
    return Con2PrimOut{T}(n, n, n, n, n, n, n, pt, C2PResult.failed_max_iter, Int32(0), Int32(0), UInt32(0))
end

"""
    con2prim(eos, in, opts; s_guess = NaN, w_guess = NaN, u_guess = NaN)

Recover primitive variables from conserved ones.

A damped 2×2 Newton on `(s, w)` with the analytic Jacobian, backed by a nested
one-dimensional fallback that is globally convergent by construction. Supplying
guesses from a previous step -- the usual case in a hydro evolution -- skips the
cold-start machinery entirely and typically converges in one or two Newton
steps.

"Damped" means *clamped*, not backtracked. Backtracking was tried and measured
harmful: near a locally ill-conditioned Jacobian a halved step barely moves and
the iteration stalls, whereas the full clamped step crosses the bad region and
recovers quadratic convergence beyond it. Taking the full step unconditionally
is what raises the warm-start Newton success rate from 86% to 99%, and the
fallback is the safety net for the remainder.

A failure still returns a fully populated best-effort state.

The search stays inside the EOS's density box: no iterate has `ρ > ρ_max`. A
state whose solution lies above the box is therefore not solved but reported as
`failed_no_bracket`, with its best iterate at or below the face.

This is [`con2prim_start`](@ref) followed, where needed, by
[`con2prim_finish`](@ref), and gives the same result bit for bit. On a GPU the
two stages can run as separate kernels.
"""
function con2prim(eos::AbstractEOS{S}, in::Con2PrimIn{T}, opts::Con2PrimOptions{T}, s_guess::T=T(NaN),
                  w_guess::T=T(NaN), u_guess::T=T(NaN)) where {S,T}
    out, st = con2prim_start(eos, in, opts, s_guess, w_guess, u_guess)
    is_done(st) && return out
    return con2prim_finish(eos, in, opts, st)
end

"""
    con2prim_start(eos, in, opts, s_guess = NaN, w_guess = NaN, u_guess = NaN) -> (out, stage)

The first stage of [`con2prim`](@ref): the cold seed, if a guess is missing,
and the Newton. If [`is_done`](@ref)`(stage)`, `out` is `con2prim`'s result;
otherwise `out` is a placeholder of NaNs, and [`con2prim_finish`](@ref) on
`stage` gives the result.

On a GPU a warp runs as long as its slowest state, and the fallback costs many
times a Newton solve, so a few states in it hold up most warps. Running the
first stage over all states and the second over only those not done, as two
kernels, keeps them apart:

```julia
out[i], st[i] = con2prim_start(eos, cin[i], opts)               # kernel 1, every i
idx = findall(!is_done, st)                                      # compaction, on the device
out[idx[j]] = con2prim_finish(eos, cin[idx[j]], opts, st[idx[j]])  # kernel 2, every j
```
"""
function con2prim_start(eos::AbstractEOS{S}, in::Con2PrimIn{T}, opts::Con2PrimOptions{T}, s_guess::T=T(NaN),
                        w_guess::T=T(NaN), u_guess::T=T(NaN)) where {S,T}
    yₑ = in.D_Y / in.D

    # The cold seed is computed only when a guess is missing, so a fully warm
    # call never pays for it.
    cold_s = isnan(s_guess)
    cold_w = isnan(w_guess)
    s = s_guess
    w = cold_w ? zero(T) : clamp(w_guess, zero(T), opts.w_max)
    u_start = u_guess

    if cold_s || cold_w
        w_lo = w_floor(eos, in.D, opts.w_max)
        seed = cold_seed(eos, in.D, in.τ, yₑ, in.S_par, in.S_perp, in.B², w_lo, opts.w_max, opts.seed_passes,
                         opts.seed_s_iters)
        cold_s && (s = seed.s)
        w = clamp(cold_w ? seed.w : w, w_lo, opts.w_max)
        # The seed's converged temperature is also the best available warm start
        # for the first evaluation.
        isnan(u_start) && (u_start = seed.u)
    elseif !above_floor(eos, in.D, w)
        # Nor for the floor's acosh, if the guess is already inside the box.
        w = max(w, w_floor(eos, in.D, opts.w_max))
    end

    r = residuals(eos, in.D, in.τ, yₑ, in.S_par, in.S_perp, in.B², s, w, u_start, opts.τ_floor_rel)
    iters = Int32(0)
    converged = f1_converged(r, opts.tol) && abs(r.f2) <= opts.tol

    # Remember the best iterate by scaled norm and hand *that* to the fallback,
    # not the last one. Unconditional clamped steps are what make this
    # necessary: an iterate oscillating at the precision floor, or one caught
    # mid-excursion when the budget runs out, leaves the final position far from
    # the best the trajectory reached.
    #
    # `best_norm` may start as NaN from a non-finite first residual; the update
    # is written as `!(n >= best_norm)` so that is replaced by the first finite
    # iterate rather than latching forever.
    #
    # Each iterate is remembered with the warm start it was evaluated with, so
    # that con2prim_finish can rebuild it exactly.
    u_last = u_start
    r_best = r
    u_best = u_start
    best_norm = scaled_norm(r)

    while !converged && iters < opts.max_iter_newton
        iters += one(Int32)

        # Cramer's rule on the 2×2 system.
        a, b, c, d = r.df1_ds, r.df1_dw, r.df2_ds, r.df2_dw
        det = a * d - b * c
        (det == zero(T) || !isfinite(det)) && break      # singular: fall through

        ds = (b * r.f2 - d * r.f1) / det
        dw = (c * r.f1 - a * r.f2) / det

        ds_max = T(0.25) * (r.sr_ext.s_max - r.sr_ext.s_min)
        ds = clamp(ds, -ds_max, ds_max)
        dw = clamp(dw, -one(T), one(T))

        # Recomputed rather than carried, so a warm start that never steps never
        # computes it. A flag saying whether it was known yet, live across the
        # residual's evaluation, cost the fused kernel 2% on an H200.
        w_next = clamp_w(w + dw, w_floor(eos, in.D, opts.w_max), opts.w_max)
        s_next = s + ds
        u_next = r.pt.u_solved
        r_next = residuals(eos, in.D, in.τ, yₑ, in.S_par, in.S_perp, in.B², s_next, w_next, u_next,
                           opts.τ_floor_rel)

        (isfinite(r_next.f1) && isfinite(r_next.f2)) || break

        s, w, r, u_last = s_next, w_next, r_next, u_next
        converged = f1_converged(r, opts.tol) && abs(r.f2) <= opts.tol

        n_now = scaled_norm(r)
        if isfinite(n_now) && !(n_now >= best_norm)
            r_best = r
            u_best = u_last
            best_norm = n_now
        end
    end

    if converged
        return fill_out(r, yₑ, C2PResult.converged_newton, iters, Int32(0)),
               Con2PrimStage{T}(r.s, r.w, u_last, iters, true)
    end
    return nan_c2p_out(T), Con2PrimStage{T}(r_best.s, r_best.w, u_best, iters, false)
end

"""
    con2prim_finish(eos, in, opts, stage) -> Con2PrimOut

The second stage of [`con2prim`](@ref), for a `stage` from
[`con2prim_start`](@ref): the nested one-dimensional fallback, started from the
Newton's best iterate. For a stage that [`is_done`](@ref) it only rebuilds the
converged result.
"""
function con2prim_finish(eos::AbstractEOS{S}, in::Con2PrimIn{T}, opts::Con2PrimOptions{T},
                         st::Con2PrimStage{T}) where {S,T}
    yₑ = in.D_Y / in.D
    iters = st.iters_newton
    # Evaluated as the first stage evaluated it, so this is its iterate, bit for bit.
    r = residuals(eos, in.D, in.τ, yₑ, in.S_par, in.S_perp, in.B², st.s, st.w, st.u, opts.τ_floor_rel)
    st.converged && return fill_out(r, yₑ, C2PResult.converged_newton, iters, Int32(0))
    r_best = r
    w_lo = w_floor(eos, in.D, opts.w_max)

    # --- Nested one-dimensional fallback ---------------------------------
    s_anchor = r_best.s
    w_anchor = r_best.w
    u_anchor = r_best.pt.u_solved

    scan = bracket_scan(eos, in.D, in.τ, yₑ, in.S_par, in.S_perp, in.B², w_lo, opts.w_max, opts.τ_floor_rel,
                        opts.bracket_scan, s_anchor, w_anchor, w_anchor, u_anchor, opts.max_iter_1d,
                        opts.tol)

    if !scan.bracketed
        best, _ = inner_solve_w(eos, in.D, in.τ, yₑ, in.S_par, in.S_perp, in.B², scan.s_best, w_lo, opts.w_max,
                                opts.τ_floor_rel, w_anchor, u_anchor, opts.max_iter_1d, opts.tol)
        # Report whichever of the scan's best point and the Newton's best
        # iterate sits closer to a root. The scan point satisfies the momentum
        # residual by construction but can sit far from the true entropy, while
        # a Newton that stalled near the root is often the better answer for a
        # caller inspecting the failed state -- and for the policy layer, which
        # repairs from it.
        rep = scaled_norm(r_best) < scaled_norm(best) ? r_best : best
        return fill_out(rep, yₑ, C2PResult.failed_no_bracket, iters, Int32(0))
    end

    # Re-solve at the chosen endpoints from the original warm start, since the
    # scan returns only its winning entropies and not their residuals.
    r_lo, _ = inner_solve_w(eos, in.D, in.τ, yₑ, in.S_par, in.S_perp, in.B², scan.s_lo, w_lo, opts.w_max,
                            opts.τ_floor_rel, w_anchor, u_anchor, opts.max_iter_1d, opts.tol)
    r_hi, _ = inner_solve_w(eos, in.D, in.τ, yₑ, in.S_par, in.S_perp, in.B², scan.s_hi, w_lo, opts.w_max,
                            opts.τ_floor_rel, r_lo.w, r_lo.pt.u_solved, opts.max_iter_1d, opts.tol)

    slo, shi = scan.s_lo, scan.s_hi
    glo, ghi = r_lo.f2, r_hi.f2
    # An endpoint that is already a root is a bracket, whatever the other
    # endpoint's sign. It is also exactly the endpoint whose sign the fresh warm
    # start is most likely to flip.
    bracketed = (glo <= 0 && ghi >= 0) || (glo >= 0 && ghi <= 0) || abs(glo) <= opts.tol || abs(ghi) <= opts.tol

    if !bracketed
        # Rare. Either the fresh warm start landed on a genuinely different
        # outer function, or an endpoint is a root whose residual sits at the
        # energy residual's resolution floor, above the tolerance, and flipped
        # sign; see CODE.md, "Open items".
        best = abs(glo) <= abs(ghi) ? r_lo : r_hi
        rep = scaled_norm(r_best) < scaled_norm(best) ? r_best : best
        return fill_out(rep, yₑ, C2PResult.failed_no_bracket, iters, Int32(0))
    end

    r_cur = abs(glo) <= abs(ghi) ? r_lo : r_hi
    outer_converged = abs(glo) <= opts.tol || abs(ghi) <= opts.tol
    r_bracket_lo, r_bracket_hi = r_lo, r_hi
    w_seed, u_seed = r_hi.w, r_hi.pt.u_solved
    outer_iters = Int32(0)

    # Illinois-modified regula falsi. Plain regula falsi stalls badly when the
    # outer function is strongly convex across the bracket, because one endpoint
    # is then never replaced. Halving the stale side's value whenever the same
    # side is retained twice preserves the sign -- so the bracket and the
    # convergence test are unaffected -- while pulling the secant estimate
    # toward the stale side and restoring superlinear convergence.
    stale_side = 0
    prev_s_next = T(NaN)
    while !outer_converged && outer_iters < opts.max_iter_1d
        outer_iters += one(Int32)

        s_next = zero(T)
        secant_ok = false
        if ghi != glo
            s_next = shi - ghi * (shi - slo) / (ghi - glo)
            smin, smax = min(slo, shi), max(slo, shi)
            secant_ok = isfinite(s_next) && smin < s_next < smax
        end
        secant_ok || (s_next = T(0.5) * (slo + shi))

        # Stagnation: the deflation above can make the secant division
        # numerically degenerate and reproduce the same point forever, without
        # the bracket itself having collapsed. The current residual already
        # holds the value there, so accept it rather than burn the budget.
        if s_next == prev_s_next
            outer_converged = true
            break
        end
        prev_s_next = s_next

        r_next, _ = inner_solve_w(eos, in.D, in.τ, yₑ, in.S_par, in.S_perp, in.B², s_next, w_lo, opts.w_max,
                                  opts.τ_floor_rel, w_seed, u_seed, opts.max_iter_1d, opts.tol)
        w_seed, u_seed = r_next.w, r_next.pt.u_solved
        r_cur = r_next
        g_next = r_next.f2

        isfinite(g_next) || break
        if abs(g_next) <= opts.tol
            outer_converged = true
            break
        end

        if (g_next < 0 && glo < 0) || (g_next > 0 && glo > 0)
            stale_side == 1 && (ghi *= T(0.5))
            slo, glo, r_bracket_lo = s_next, g_next, r_next
            stale_side = 1
        else
            stale_side == 2 && (glo *= T(0.5))
            shi, ghi, r_bracket_hi = s_next, g_next, r_next
            stale_side = 2
        end

        # Once the bracket collapses to adjacent floats nothing further can
        # shrink it, and a single unit in the last place can move the energy
        # residual by more than the tolerance in the entropy-sensitive regions.
        if !(slo < shi)
            r_cur = abs(glo) <= abs(ghi) ? r_bracket_lo : r_bracket_hi
            outer_converged = true
            break
        end
    end

    # The outer solve is limited to representable resolution in s, which is not
    # always enough where the energy residual is steep. A coupled Newton
    # correction is not so limited -- it computes a continuous step rather than
    # a bracket midpoint -- so a handful of them can land on a strictly better
    # representable value. Stops as soon as a step fails to improve.
    for _ in 1:5
        a, b, c, d = r_cur.df1_ds, r_cur.df1_dw, r_cur.df2_ds, r_cur.df2_dw
        det = a * d - b * c
        (det == zero(T) || !isfinite(det)) && break

        ds = (b * r_cur.f2 - d * r_cur.f1) / det
        dw = (c * r_cur.f1 - a * r_cur.f2) / det
        s_try = r_cur.s + ds
        w_try = clamp_w(r_cur.w + dw, w_lo, opts.w_max)
        r_try = residuals(eos, in.D, in.τ, yₑ, in.S_par, in.S_perp, in.B², s_try, w_try,
                          r_cur.pt.u_solved, opts.τ_floor_rel)
        n_try = scaled_norm(r_try)
        (isfinite(n_try) && n_try < scaled_norm(r_cur)) || break
        r_cur = r_try
    end

    # A Newton iterate closer to a root than the fallback's own best point is
    # the better thing to report -- but only when the fallback did *not*
    # converge. Substituting into a converged fallback could swap in a point
    # near a different root of the same state, which is exactly the silent
    # wrong-root class this solver was tuned to eliminate.
    accepted = outer_converged && f1_converged(r_cur, opts.tol)
    if !accepted && scaled_norm(r_best) < scaled_norm(r_cur)
        r_cur = r_best
        accepted = f1_converged(r_cur, opts.tol) && abs(r_cur.f2) <= opts.tol
    end
    result = accepted ? C2PResult.converged_fallback : C2PResult.failed_max_iter
    return fill_out(r_cur, yₑ, result, iters, outer_iters)
end

con2prim(eos::AbstractEOS{S}, in::Con2PrimIn{T}, opts::Con2PrimOptions{P}, args...) where {S,T,P} =
    con2prim(eos, in, Con2PrimOptions{T}(; tol=T(opts.tol), max_iter_newton=opts.max_iter_newton,
                                         max_iter_1d=opts.max_iter_1d, w_max=T(opts.w_max),
                                         τ_floor_rel=T(opts.τ_floor_rel), bracket_scan=opts.bracket_scan,
                                         seed_passes=opts.seed_passes, seed_s_iters=opts.seed_s_iters),
             args...)

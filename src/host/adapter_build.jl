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

# ---------------------------------------------------------------------------
# Monotonicity audit
#
# A bounded worst-offender list, so that memory stays constant however many
# points violate. This matters because the build deliberately does not require
# a pre-repaired table: it reports what it finds rather than refusing.
# ---------------------------------------------------------------------------

const MAX_WORST = 10

mutable struct AuditAccum{T<:AbstractFloat}
    min_value::T
    violation_count::Int
    worst::Vector{AuditLoc{T}}
end

AuditAccum{T}() where {T} = AuditAccum{T}(T(Inf), 0, AuditLoc{T}[])

"""Offer a location, evicting the least negative incumbent once full."""
function offer!(a::AuditAccum{T}, loc::AuditLoc{T}) where {T}
    if length(a.worst) < MAX_WORST
        push!(a.worst, loc)
        return a
    end
    max_idx, max_val = 1, a.worst[1].value
    for i in 2:length(a.worst)
        if a.worst[i].value > max_val
            max_val = a.worst[i].value
            max_idx = i
        end
    end
    loc.value < max_val && (a.worst[max_idx] = loc)
    return a
end

function consider!(a::AuditAccum{T}, x, u, y, value) where {T}
    a.min_value = min(a.min_value, value)
    if value <= 0
        a.violation_count += 1
        offer!(a, AuditLoc{T}(x, u, y, value))
    end
    return a
end

"""Fold one accumulator into another, preserving the overall ten worst."""
function merge_from!(dst::AuditAccum{T}, src::AuditAccum{T}) where {T}
    dst.min_value = min(dst.min_value, src.min_value)
    dst.violation_count += src.violation_count
    for loc in src.worst
        offer!(dst, loc)
    end
    return dst
end

function finalize_audit(a::AuditAccum{T}) where {T}
    return MonotonicityAudit{T}(a.min_value, a.violation_count, sort(a.worst; by=l -> l.value))
end

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

"""
    check_uniform_axis(axis, name, tol)

The adapter needs uniformly spaced axes: the spline evaluator locates a cell by
arithmetic on `(x - x0)/h` rather than by searching, so a non-uniform axis would
silently be evaluated at the wrong place.
"""
function check_uniform_axis(axis::AbstractVector, name::AbstractString, tol::Real)
    n = length(axis)
    h = (axis[end] - axis[begin]) / (n - 1)
    h > 0 || throw(ArgumentError("build_eos: axis '$name' is not strictly increasing"))
    for i in 2:n
        rel = abs((axis[i] - axis[i - 1]) - h) / h
        rel > tol && throw(
            ArgumentError(
                "build_eos: axis '$name' is not uniform (relative spacing deviation $rel at index $i exceeds $tol)",
            ),
        )
    end
    return nothing
end

function check_field_present_finite(t::RawTable, name::AbstractString)
    has_field(t, name) || throw(ArgumentError("build_eos: table has no field '$name'"))
    all(isfinite, field(t, name)) || throw(ArgumentError("build_eos: field '$name' has a non-finite value"))
    return nothing
end

# ---------------------------------------------------------------------------
# Refined-grid scans
#
# Both scans sample the fitted *splines*, never the raw data: a C² spline can
# undershoot its own data minimum, and it is the spline that will be evaluated
# at run time, so it is the spline whose minimum sets the energy floor.
# ---------------------------------------------------------------------------

"""
One pass over a refined grid spanning the physical box, tracking the minimum of
`ε̂` for the κ floor and the monotonicity of both fields in `u`.
"""
# The per-slice bodies are separate functions rather than inline loop bodies.
# Inside `Threads.@threads` an inline body becomes a closure, and a scalar
# accumulator assigned there can be boxed into the *enclosing* frame -- shared
# by every thread, which both allocates per iteration and races. Hoisting the
# work into a function makes the accumulators unambiguously thread-local.

function scan_slice!(σ_acc::AuditAccum{T}, L_acc::AuditAccum{T}, σv::BsplineView3{T},
                     Lv::BsplineView3{T}, y, x0, dx, rx, u0, du, ru, shift_hat, inv_c²) where {T}
    ε_min = T(Inf)
    for ju in 0:(ru - 1)
        u = u0 + ju * du
        for ix in 0:(rx - 1)
            x = x0 + ix * dx
            σ = bspline_eval3(σv, x, u, y)
            L = bspline_eval3(Lv, x, u, y)
            consider!(σ_acc, x, u, y, σ.fu)
            consider!(L_acc, x, u, y, L.fu)
            ε̂ = exp10(L.f) * inv_c² - shift_hat
            ε̂ < ε_min && (ε_min = ε̂)
        end
    end
    return ε_min
end

function scan_refined_grid(σv::BsplineView3{T}, Lv::BsplineView3{T}, x0, hx, nx, u0, hu, nu, y0, hy, ny,
                           refine, shift_hat, inv_c²) where {T}
    rx, ru, ry = (nx - 1) * refine + 1, (nu - 1) * refine + 1, (ny - 1) * refine + 1
    dx, du, dy = hx / refine, hu / refine, hy / refine

    # One accumulator per Yₑ slice, merged afterwards in slice order. The
    # partition is fixed rather than thread-dependent, so the result is the same
    # however many threads run it -- which matters because κ, derived from the
    # minimum below, is part of the EOS identity.
    ε_mins = fill(T(Inf), ry)
    σ_accs = [AuditAccum{T}() for _ in 1:ry]
    L_accs = [AuditAccum{T}() for _ in 1:ry]

    Threads.@threads for ky in 0:(ry - 1)
        @inbounds ε_mins[ky + 1] = scan_slice!(σ_accs[ky + 1], L_accs[ky + 1], σv, Lv, y0 + ky * dy,
                                               x0, dx, rx, u0, du, ru, shift_hat, inv_c²)
    end

    # Reduce with the same `<` test the inner loop uses, so a NaN never
    # displaces a real minimum; `min` would propagate it instead.
    ε_min = T(Inf)
    σ_total = AuditAccum{T}()
    L_total = AuditAccum{T}()
    for k in 1:ry
        @inbounds ε_mins[k] < ε_min && (ε_min = ε_mins[k])
        merge_from!(σ_total, @inbounds σ_accs[k])
        merge_from!(L_total, @inbounds L_accs[k])
    end
    return ε_min, AdapterAudit{T}(finalize_audit(σ_total), finalize_audit(L_total))
end

"""
The minimum of `ε̂` over the *extended* box, sampled through the same designed
tails `evaluate` uses.

This must run: the energy field's low-temperature tail can dip `ε̂` below the
table's own minimum, and the floor has to cover everywhere the solver can
actually land. It is safe to run before κ is known, because κ only relabels the
density origin and the causal clamp depends on `ε̂` and the entropy growth rate
alone.
"""
function ext_scan_slice(Lv::BsplineView3{T}, σv::BsplineView3{T}, y, x_ext_lo, dx, rx, u_ext_lo, du, ru,
                        x_lo, x_hi, u_lo, u_hi, slope_floor_σ, slope_floor_L, cs²_ext_cap, shift_hat,
                        inv_c²) where {T}
    ε_min = T(Inf)
    for ju in 0:(ru - 1)
        u = u_ext_lo + ju * du
        u_above = u > u_hi
        for ix in 0:(rx - 1)
            x = x_ext_lo + ix * dx
            b_cap = zero(T)
            if u_above
                α = σ_u_high_alpha(σv, clamp(x, x_lo, x_hi), u_hi, y, slope_floor_σ)
                α > 0 && (b_cap = (one(T) + cs²_ext_cap) * α)
            end
            # This scan evaluates L, whose tails are never log-space ones.
            spec = ExtSpec{T}(x_lo, x_hi, u_lo, u_hi, x_ext_lo, slope_floor_L, false, false, b_cap,
                              shift_hat, inv_c²)
            L = extended_sample(Lv, x, u, y, spec)
            ε̂ = exp10(L.f) * inv_c² - shift_hat
            ε̂ < ε_min && (ε_min = ε̂)
        end
    end
    return ε_min
end

function scan_extended_eps_floor(Lv::BsplineView3{T}, σv::BsplineView3{T}, x0, hx, nx, u0, hu, nu, y0, hy,
                                ny, refine, ext_cells, slope_floor_σ, slope_floor_L, cs²_ext_cap,
                                shift_hat, inv_c²) where {T}
    x_lo, x_hi = x0, x0 + (nx - 1) * hx
    u_lo, u_hi = u0, u0 + (nu - 1) * hu
    x_ext_lo = x_lo - ext_cells * hx
    u_ext_lo = u_lo - ext_cells * hu

    rx = (nx - 1 + 2ext_cells) * refine + 1
    ru = (nu - 1 + 2ext_cells) * refine + 1
    ry = (ny - 1) * refine + 1
    dx, du, dy = hx / refine, hu / refine, hy / refine

    # A pure minimum reduction over a fixed partition, so the threaded result is
    # bit-identical to the serial one.
    ε_mins = fill(T(Inf), ry)
    Threads.@threads for ky in 0:(ry - 1)
        @inbounds ε_mins[ky + 1] = ext_scan_slice(Lv, σv, y0 + ky * dy, x_ext_lo, dx, rx, u_ext_lo, du,
                                                  ru, x_lo, x_hi, u_lo, u_hi, slope_floor_σ,
                                                  slope_floor_L, cs²_ext_cap, shift_hat, inv_c²)
    end

    ε_min = T(Inf)
    for k in 1:ry
        @inbounds ε_mins[k] < ε_min && (ε_min = ε_mins[k])
    end
    return ε_min
end

# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------

"""
    build_eos(table, opts = BuildOptions())

Build the adapter from a raw table: validate, fit both splines, derive the
baryon-mass rescaling κ, and record a monotonicity audit.

Throws if an axis is non-uniform or shorter than four points, if `entropy` or
`logenergy` is missing or non-finite, or if the `energy_shift` attribute is
absent. It does not repair anything -- that is an offline step, and
[`check_table`](@ref) is what diagnoses a table that still needs it.
"""
function build_eos(t::RawTable{T}, opts::BuildOptions=BuildOptions()) where {T}
    nρ_, nT_, nYₑ_ = nρ(t), nT(t), nYₑ(t)
    nρ_ >= 4 || throw(ArgumentError("build_eos: ρ axis has fewer than 4 points"))
    nT_ >= 4 || throw(ArgumentError("build_eos: T axis has fewer than 4 points"))
    nYₑ_ >= 4 || throw(ArgumentError("build_eos: Yₑ axis has fewer than 4 points"))

    check_uniform_axis(t.logρ, "logrho", opts.uniform_tol)
    check_uniform_axis(t.logT, "logtemp", opts.uniform_tol)
    check_uniform_axis(t.Yₑ, "ye", opts.uniform_tol)
    check_field_present_finite(t, "logenergy")
    check_field_present_finite(t, "entropy")
    has_attribute(t, "energy_shift") ||
        throw(ArgumentError("build_eos: table has no 'energy_shift' attribute"))

    x0 = t.logρ[1]
    hx = (t.logρ[end] - t.logρ[1]) / (nρ_ - 1)
    u0 = t.logT[1]
    hu = (t.logT[end] - t.logT[1]) / (nT_ - 1)
    y0 = t.Yₑ[1]
    hy = (t.Yₑ[end] - t.Yₑ[1]) / (nYₑ_ - 1)

    σ_raw = fit_bspline_3d(field(t, "entropy"), x0, hx, u0, hu, y0, hy)
    L_raw = fit_bspline_3d(field(t, "logenergy"), x0, hx, u0, hu, y0, hy)

    inv_c² = T(1 / (C_LIGHT_CM_S * C_LIGHT_CM_S))
    shift_hat = T(energy_shift(t) * inv_c²)
    conv_t = T(MEV_TO_ERG / (opts.m_B_table_g * C_LIGHT_CM_S * C_LIGHT_CM_S))

    ε_min_box, audit = scan_refined_grid(σ_raw, L_raw, x0, hx, nρ_, u0, hu, nT_, y0, hy, nYₑ_,
                                         opts.refine, shift_hat, inv_c²)
    ε_min_ext = scan_extended_eps_floor(L_raw, σ_raw, x0, hx, nρ_, u0, hu, nT_, y0, hy, nYₑ_,
                                        opts.refine, opts.ext_cells, T(opts.ext_slope_floor_σ),
                                        T(opts.ext_slope_floor_L), T(opts.cs²_ext_cap), shift_hat, inv_c²)
    ε̂_min = min(ε_min_box, ε_min_ext)

    ε_floor = min(zero(T), ε̂_min - (T(opts.κ_margin_abs) + T(opts.κ_margin_rel) * abs(ε̂_min)))
    κ = one(T) + ε_floor
    m_B_star_g = κ * opts.m_B_table_g
    x0_star = x0 + log10(κ)

    # The rescaling is a relabelling of the density origin -- same coefficients,
    # no refit.
    σ_final = BsplineView3(σ_raw.c, x0_star, hx, u0, hu, y0, hy)
    L_final = BsplineView3(L_raw.c, x0_star, hx, u0, hu, y0, hy)

    x_lo, x_hi = x0_star, x0_star + (nρ_ - 1) * hx
    u_lo, u_hi = u0, u0 + (nT_ - 1) * hu
    y_lo, y_hi = y0, y0 + (nYₑ_ - 1) * hy
    # The extension's width is a count of physical grid cells, so it rides along
    # with the relabelling unchanged.
    x_ext_lo, x_ext_hi = x_lo - opts.ext_cells * hx, x_hi + opts.ext_cells * hx
    u_ext_lo, u_ext_hi = u_lo - opts.ext_cells * hu, u_hi + opts.ext_cells * hu

    view = EOSTableView(σ_final, L_final, κ, shift_hat, conv_t, inv_c², x_lo, x_hi, u_lo, u_hi, y_lo, y_hi,
                        x_ext_lo, x_ext_hi, u_ext_lo, u_ext_hi, T(opts.ext_slope_floor_σ),
                        T(opts.ext_slope_floor_L), T(opts.cs²_ext_cap), Int32(50))
    return EOSTable{T}(view, m_B_star_g, opts.m_B_table_g, audit)
end

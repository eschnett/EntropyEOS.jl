# Tensor-product uniform cubic B-spline evaluation: value and derivatives up to
# second order, in 1D and in the axis-separable 3D case the adapter needs.
#
# Translated from `entropy_eos/core/bspline_eval.hpp`.
#
# Data points sit at xᵢ = x0 + i*h for i = 0..n-1 (n >= 4). The spline carries
# n+2 control coefficients -- two more than data points -- fixed at fit time by
# not-a-knot end conditions, which make it reproduce any global cubic exactly.
# On cell i, with t the local coordinate in [0,1]:
#
#     S(x) = cᵢ b₀(t) + cᵢ₊₁ b₁(t) + cᵢ₊₂ b₂(t) + cᵢ₊₃ b₃(t)
#
# Outside the grid this evaluates the boundary cell's cubic -- smooth
# polynomial extrapolation, not a designed extension. Callers clamp and flag
# out-of-range arguments themselves; the adapter's tails do exactly that.

"""
    BsplineView1(c, x0, h)

A 1-D uniform cubic B-spline over `length(c) - 2` data points starting at `x0`
with spacing `h`.

Unlike the C++ original this does not store the data-point count: the
coefficient array knows its own length on every backend, which removes the
standing `n` versus `n+2` hazard.
"""
struct BsplineView1{T<:AbstractFloat,A<:AbstractVector{T}}
    c::A
    x0::T
    h::T
end

function BsplineView1(c::AbstractVector{T}, x0, h) where {T<:AbstractFloat}
    return BsplineView1{T,typeof(c)}(c, T(x0), T(h))
end

"""
    BsplineView3(c, x0, hx, u0, hu, y0, hy)

A 3-D tensor-product uniform cubic B-spline. `c` has size
`(nx+2, nu+2, ny+2)` with the x index fastest, which is Julia's native
column-major order and matches the C++ layout without any permutation.
"""
struct BsplineView3{T<:AbstractFloat,A<:AbstractArray{T,3}}
    c::A
    x0::T
    hx::T
    u0::T
    hu::T
    y0::T
    hy::T
end

function BsplineView3(c::AbstractArray{T,3}, x0, hx, u0, hu, y0, hy) where {T<:AbstractFloat}
    return BsplineView3{T,typeof(c)}(c, T(x0), T(hx), T(u0), T(hu), T(y0), T(hy))
end

"""Data-point counts along each axis (the coefficient array is two larger)."""
@inline npoints_x(v::BsplineView3) = size(v.c, 1) - 2
@inline npoints_u(v::BsplineView3) = size(v.c, 2) - 2
@inline npoints_y(v::BsplineView3) = size(v.c, 3) - 2
@inline npoints(v::BsplineView1) = length(v.c) - 2

"""Value and derivatives of a 1-D spline."""
struct BsplineEval1{T<:AbstractFloat}
    f::T
    fx::T
    fxx::T
end

"""
Value and the derivative set the adapter's chain rule needs.

Deliberately missing `fyy`, `fxy` and `fuy`: nothing downstream uses them, and
skipping them halves the per-axis basis work for the y axis.
"""
struct BsplineEval3{T<:AbstractFloat}
    f::T
    fx::T
    fu::T
    fy::T
    fxx::T
    fxu::T
    fuu::T
end

"""One axis's cell index and local coordinate for a query point."""
struct BsplineCell{T<:AbstractFloat}
    i::Int   # 1-based, clamped to 1:n-1
    t::T     # in [0,1] inside the grid, extrapolated outside
end

"""
    bspline_cell(x, x0, h, n)

Locate `x` within the grid. This is the single place where the C++'s 0-based
cell index becomes 1-based, so no further index shift appears anywhere in the
evaluation loops.
"""
@inline function bspline_cell(x::T, x0::T, h::T, n::Int) where {T<:AbstractFloat}
    xi = (x - x0) / h
    i = clamp(trunc_floor(xi) + 1, 1, n - 1)
    return BsplineCell{T}(i, xi - T(i - 1))
end

"""The four cubic B-spline basis functions at one local coordinate."""
struct Basis4{T<:AbstractFloat}
    b0::T
    b1::T
    b2::T
    b3::T
end

@inline function bspline_basis(t::T) where {T<:AbstractFloat}
    t2 = t * t
    t3 = t2 * t
    omt = one(T) - t
    return Basis4{T}(
        omt * omt * omt / T(6),
        (T(3) * t3 - T(6) * t2 + T(4)) / T(6),
        (T(-3) * t3 + T(3) * t2 + T(3) * t + one(T)) / T(6),
        t3 / T(6),
    )
end

@inline function bspline_dbasis(t::T) where {T<:AbstractFloat}
    t2 = t * t
    omt = one(T) - t
    return Basis4{T}(
        -omt * omt / T(2),
        (T(3) * t2 - T(4) * t) / T(2),
        (T(-3) * t2 + T(2) * t + one(T)) / T(2),
        t2 / T(2),
    )
end

@inline function bspline_d2basis(t::T) where {T<:AbstractFloat}
    return Basis4{T}(one(T) - t, T(3) * t - T(2), one(T) - T(3) * t, t)
end

"""
    bspline_eval1(v, x)

Value and first two derivatives of a 1-D spline at `x`.
"""
@inline function bspline_eval1(v::BsplineView1{T}, x::R) where {T,R<:AbstractFloat}
    cell = bspline_cell(R(x), R(v.x0), R(v.h), npoints(v))
    b = bspline_basis(cell.t)
    d1 = bspline_dbasis(cell.t)
    d2 = bspline_d2basis(cell.t)

    @inbounds c0 = v.c[cell.i + 0]
    @inbounds c1 = v.c[cell.i + 1]
    @inbounds c2 = v.c[cell.i + 2]
    @inbounds c3 = v.c[cell.i + 3]

    inv_h = one(R) / R(v.h)
    f = c0 * b.b0 + c1 * b.b1 + c2 * b.b2 + c3 * b.b3
    fx = (c0 * d1.b0 + c1 * d1.b1 + c2 * d1.b2 + c3 * d1.b3) * inv_h
    fxx = (c0 * d2.b0 + c1 * d2.b1 + c2 * d2.b2 + c3 * d2.b3) * (inv_h * inv_h)
    return BsplineEval1{R}(f, fx, fxx)
end

"""
    bspline_eval3(v, x, u, y)

Value and derivatives of the tensor-product spline at `(x, u, y)`, by
contracting the 4×4×4 block of coefficients around the containing cell with
the per-axis basis vectors.

Fixed cost: 64 coefficient reads and no data-dependent branches beyond the
three per-axis index clamps. The x axis is contracted innermost because it is
the fastest-varying one.
"""
@inline function bspline_eval3(v::BsplineView3{T}, x::R, u::R, y::R) where {T,R<:AbstractFloat}
    cx = bspline_cell(x, R(v.x0), R(v.hx), npoints_x(v))
    cu = bspline_cell(u, R(v.u0), R(v.hu), npoints_u(v))
    cy = bspline_cell(y, R(v.y0), R(v.hy), npoints_y(v))

    bx = bspline_basis(cx.t)
    dx = bspline_dbasis(cx.t)
    hx2 = bspline_d2basis(cx.t)
    bu = bspline_basis(cu.t)
    du = bspline_dbasis(cu.t)
    hu2 = bspline_d2basis(cu.t)
    by = bspline_basis(cy.t)
    dy = bspline_dbasis(cy.t)

    bxt = (bx.b0, bx.b1, bx.b2, bx.b3)
    dxt = (dx.b0, dx.b1, dx.b2, dx.b3)
    hxt = (hx2.b0, hx2.b1, hx2.b2, hx2.b3)
    but = (bu.b0, bu.b1, bu.b2, bu.b3)
    dut = (du.b0, du.b1, du.b2, du.b3)
    hut = (hu2.b0, hu2.b1, hu2.b2, hu2.b3)
    byt = (by.b0, by.b1, by.b2, by.b3)
    dyt = (dy.b0, dy.b1, dy.b2, dy.b3)

    f = zero(R)
    fx = zero(R)
    fu = zero(R)
    fy = zero(R)
    fxx = zero(R)
    fxu = zero(R)
    fuu = zero(R)

    @inbounds for r in 0:3
        iy = cy.i + r
        for q in 0:3
            iu = cu.i + q
            # Contract along x first for this (q, r): value, 1st, 2nd derivative.
            sf = zero(R)
            sdx = zero(R)
            shx = zero(R)
            for p in 0:3
                cc = v.c[cx.i + p, iu, iy]
                sf += bxt[p + 1] * cc
                sdx += dxt[p + 1] * cc
                shx += hxt[p + 1] * cc
            end

            wuy = but[q + 1] * byt[r + 1]
            f += wuy * sf
            fx += wuy * sdx
            fxx += wuy * shx
            fu += dut[q + 1] * byt[r + 1] * sf
            fy += but[q + 1] * dyt[r + 1] * sf
            fxu += dut[q + 1] * byt[r + 1] * sdx
            fuu += hut[q + 1] * byt[r + 1] * sf
        end
    end

    return BsplineEval3{R}(
        f,
        fx / R(v.hx),
        fu / R(v.hu),
        fy / R(v.hy),
        fxx / (R(v.hx) * R(v.hx)),
        fxu / (R(v.hx) * R(v.hu)),
        fuu / (R(v.hu) * R(v.hu)),
    )
end

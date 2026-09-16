# Moving a view between host and device.
#
# There is no counterpart to this in the C++, which needs a hand-written
# vendor-specific mirror object per backend to allocate device memory, copy the
# coefficient blobs, rebind raw pointers into a fresh view, and free it all
# again at the right time. Holding the arrays as fields makes all of that the
# array type's problem.
#
# One rule covers both hops. `adapt(CuArray, v)` gives a device-resident,
# host-callable view; passing *that* into a kernel triggers the backend's own
# adaptor, which turns it into a device-array view that is `isbits`. This is
# exactly the protocol KernelAbstractions uses, so a caller's KA kernel can take
# a view as an argument with nothing further required of this package.

function Adapt.adapt_structure(to, v::BsplineView1)
    return BsplineView1(Adapt.adapt(to, v.c), v.x0, v.h)
end

function Adapt.adapt_structure(to, v::BsplineView3)
    return BsplineView3(Adapt.adapt(to, v.c), v.x0, v.hx, v.u0, v.hu, v.y0, v.hy)
end

function Adapt.adapt_structure(to, v::EOSTableView)
    σ = Adapt.adapt(to, v.σ)
    L = Adapt.adapt(to, v.L)
    return EOSTableView(
        σ, L, v.κ, v.shift_hat, v.conv_t, v.inv_c²,
        v.x_lo, v.x_hi, v.u_lo, v.u_hi, v.y_lo, v.y_hi,
        v.x_ext_lo, v.x_ext_hi, v.u_ext_lo, v.u_ext_hi,
        v.ext_slope_floor_σ, v.ext_slope_floor_L, v.cs²_ext_cap, v.max_iter,
    )
end

# Outer constructor used by the adapt rule above: infers the type parameters
# from the already-adapted splines, and leaves the scalars alone.
function EOSTableView(
    σ::BsplineView3{T,A}, L::BsplineView3{T,A}, κ, shift_hat, conv_t, inv_c²,
    x_lo, x_hi, u_lo, u_hi, y_lo, y_hi, x_ext_lo, x_ext_hi, u_ext_lo, u_ext_hi,
    ext_slope_floor_σ, ext_slope_floor_L, cs²_ext_cap, max_iter,
) where {T,A}
    return EOSTableView{T,A}(
        σ, L, T(κ), T(shift_hat), T(conv_t), T(inv_c²),
        T(x_lo), T(x_hi), T(u_lo), T(u_hi), T(y_lo), T(y_hi),
        T(x_ext_lo), T(x_ext_hi), T(u_ext_lo), T(u_ext_hi),
        T(ext_slope_floor_σ), T(ext_slope_floor_L), T(cs²_ext_cap), Int32(max_iter),
    )
end

# Narrowing the scalar type is deliberately explicit and separate from
# `adapt`: a device hop must never change precision silently. Metal is
# Float32-only, so a Metal upload reads
# `adapt(MtlArray, narrow(EOSTableView(tbl), Float32))`.

"""
    narrow(v, S)

A copy of `v` with scalar type `S`. Note that the physics is only validated at
`Float64`; see the package documentation.
"""
function narrow(v::BsplineView3, ::Type{S}) where {S<:AbstractFloat}
    return BsplineView3(convert(Array{S,3}, v.c), S(v.x0), S(v.hx), S(v.u0), S(v.hu), S(v.y0), S(v.hy))
end

function narrow(v::EOSTableView, ::Type{S}) where {S<:AbstractFloat}
    return EOSTableView(
        narrow(v.σ, S), narrow(v.L, S), S(v.κ), S(v.shift_hat), S(v.conv_t), S(v.inv_c²),
        S(v.x_lo), S(v.x_hi), S(v.u_lo), S(v.u_hi), S(v.y_lo), S(v.y_hi),
        S(v.x_ext_lo), S(v.x_ext_hi), S(v.u_ext_lo), S(v.u_ext_hi),
        S(v.ext_slope_floor_σ), S(v.ext_slope_floor_L), S(v.cs²_ext_cap), v.max_iter,
    )
end

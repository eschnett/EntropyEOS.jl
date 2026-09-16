# Fitting uniform cubic B-spline coefficients.
#
# Translated from `entropy_eos/host/bspline_fit.{hpp,cpp}`.
#
# For n data points the spline has n+2 coefficients, determined by n
# interpolation conditions plus two not-a-knot end conditions:
#
#   interpolation, i = 1..n:  (cᵢ + 4cᵢ₊₁ + cᵢ₊₂)/6 = fᵢ
#   not-a-knot left:          c₁ - 4c₂ + 6c₃ - 4c₄ + c₅ = 0
#   not-a-knot right:         (the same stencil at the far end)
#
# Not-a-knot is what makes the spline reproduce any global cubic exactly.
#
# The 3-D fit is three independent axis passes with one matrix factorization
# per axis.
#
# Note the C++ `Bspline3` class has no counterpart here. It existed only to own
# the coefficient vector that `BsplineView3` pointed into; in Julia
# `BsplineView3` holds the array directly, so the fit simply returns one.

"""
    BandedLU{T}

Banded LU with partial pivoting, bandwidths `kl = ku = 4`.

Hand-written rather than delegated to LAPACK's `gbtrf!`: the storage layout
here carries `kl` extra workspace columns per row and is not LAPACK's, the
blocked LAPACK factorization pivots in a different order and would perturb the
fitted coefficients in the last few digits, and a hand-written version stays
generic in the scalar type.
"""
mutable struct BandedLU{T<:AbstractFloat}
    n::Int
    row_stride::Int          # 2*kl + ku + 1
    ab::Vector{T}
    pivot::Vector{Int}
    factored::Bool
end

const BANDED_KL = 4
const BANDED_KU = 4

"""
    BandedLU{T}(n)

An all-zero `n × n` banded system (`n >= 1`), ready to be filled in entry by
entry with `lu[row, col] = v` and then handed to [`factor!`](@ref).

Factor once, [`solve!`](@ref) many times: the not-a-knot system depends only on
the axis length, so one factorization serves every line along that axis.
"""
function BandedLU{T}(n::Integer) where {T<:AbstractFloat}
    n >= 1 || throw(ArgumentError("BandedLU: n must be >= 1 (got $n)"))
    m = Int(n)
    row_stride = 2 * BANDED_KL + BANDED_KU + 1
    return BandedLU{T}(m, row_stride, zeros(T, m * row_stride), zeros(Int, m), false)
end

Base.size(lu::BandedLU) = (lu.n, lu.n)
Base.eltype(::BandedLU{T}) where {T} = T

# Row-major band storage: row `row` occupies `row_stride` consecutive slots
# holding the diagonals `col - row = -kl .. kl+ku`. The true band is only
# `|col - row| <= kl`; the `kl` slots past it are workspace for the fill-in
# that partial pivoting pushes to the right of the band, which is why the
# stride is `2kl + ku + 1` and not LAPACK's `kl + ku + 1`.
@inline band_index(lu::BandedLU, row::Int, col::Int) = (row - 1) * lu.row_stride + (col - row + BANDED_KL) + 1

# Unchecked access over the whole extended band, including the fill-in
# workspace. Used by factor!/solve!; getindex/setindex! below are the checked
# public door and deliberately refuse the workspace columns.
@inline elem(lu::BandedLU, row::Int, col::Int) = @inbounds lu.ab[band_index(lu, row, col)]
@inline function setelem!(lu::BandedLU{T}, row::Int, col::Int, v::T) where {T}
    @inbounds lu.ab[band_index(lu, row, col)] = v
    return v
end

@inline function check_band(lu::BandedLU, row::Int, col::Int)
    (1 <= row <= lu.n && 1 <= col <= lu.n) || throw(BoundsError(lu, (row, col)))
    # An entry outside the band is structurally zero and has no storage at all,
    # so writing one would silently corrupt a neighbouring diagonal rather than
    # widen the matrix.
    abs(row - col) <= BANDED_KL ||
        throw(ArgumentError("BandedLU: entry ($row, $col) lies outside the stored band (kl = ku = $BANDED_KL)"))
    return nothing
end

"""
    lu[row, col], lu[row, col] = v

Checked access to an entry inside the band. An index outside the matrix raises
`BoundsError`; one inside the matrix but outside the band raises
`ArgumentError`, since such an entry is structurally zero and has no storage.
Writing is only meaningful before [`factor!`](@ref); reading afterwards returns
the corresponding entry of the L/U factors.
"""
function Base.getindex(lu::BandedLU, row::Integer, col::Integer)
    check_band(lu, Int(row), Int(col))
    return elem(lu, Int(row), Int(col))
end

function Base.setindex!(lu::BandedLU{T}, v, row::Integer, col::Integer) where {T}
    # After factor! the storage holds L and U, not the original matrix, so a
    # write would mean something quite different from what the caller intends.
    lu.factored && throw(ArgumentError("BandedLU: matrix already factored"))
    check_band(lu, Int(row), Int(col))
    return setelem!(lu, Int(row), Int(col), T(v))
end

"""
    factor!(lu)

Factor `lu` in place by banded Gaussian elimination with partial pivoting.

This is LAPACK `dgbtf2`'s *unblocked* algorithm: the pivot search for column
`j` covers only rows `j..j+kl`, the only ones that can still hold a nonzero
there, and every update stops at `ju`, the rightmost column any row involved
can reach. `ju` grows as pivoting shuffles rows rightwards -- that growth is
exactly the fill-in the extra workspace columns exist to hold.

Throws `ErrorException` if a zero pivot survives the search, and `ArgumentError`
if called twice.
"""
function factor!(lu::BandedLU{T}) where {T}
    lu.factored && throw(ArgumentError("BandedLU: already factored"))
    n = lu.n
    fill!(lu.pivot, 0)
    ju = 0   # rightmost column touched by any interchange or update so far

    for j in 1:n
        km = min(BANDED_KL, n - j)   # subdiagonal rows still available: j+1..j+km

        prow = j
        pval = abs(elem(lu, j, j))
        for i in 1:km
            v = abs(elem(lu, j + i, j))
            if v > pval
                pval = v
                prow = j + i
            end
        end
        if elem(lu, prow, j) == zero(T)
            error("BandedLU: singular matrix (zero pivot at column $j)")
        end
        lu.pivot[j] = prow

        i0 = prow - j
        ju = max(ju, min(j + BANDED_KU + i0, n))

        if prow != j
            for c in j:ju
                a = elem(lu, j, c)
                setelem!(lu, j, c, elem(lu, prow, c))
                setelem!(lu, prow, c, a)
            end
        end

        if km > 0
            diag = elem(lu, j, j)
            for i in 1:km
                setelem!(lu, j + i, j, elem(lu, j + i, j) / diag)
            end
            for c in (j + 1):ju
                ujc = elem(lu, j, c)
                if ujc != zero(T)
                    for i in 1:km
                        setelem!(lu, j + i, c, elem(lu, j + i, c) - elem(lu, j + i, j) * ujc)
                    end
                end
            end
        end
    end

    lu.factored = true
    return lu
end

"""
    solve!(lu, rhs)

Solve `A x = rhs` for the already-factored `lu`, overwriting `rhs` with `x`.
May be called any number of times -- that is the point of factoring separately.
"""
function solve!(lu::BandedLU{T}, rhs::AbstractVector{T}) where {T}
    lu.factored || throw(ArgumentError("BandedLU: factor! has not been called"))
    length(rhs) == lu.n || throw(ArgumentError("BandedLU: rhs length $(length(rhs)) != n = $(lu.n)"))
    n = lu.n

    # Forward: apply the recorded row interchanges, then the unit-lower-
    # triangular multipliers (L y = P b).
    @inbounds for j in 1:n
        p = lu.pivot[j]
        if p != j
            rhs[j], rhs[p] = rhs[p], rhs[j]
        end
        km = min(BANDED_KL, n - j)
        for i in 1:km
            rhs[j + i] -= elem(lu, j + i, j) * rhs[j]
        end
    end

    # Back-substitution (U x = y); `hi` mirrors factor!'s `ju` bound.
    @inbounds for j in n:-1:1
        s = rhs[j]
        hi = min(j + BANDED_KU + BANDED_KL, n)
        for c in (j + 1):hi
            s -= elem(lu, j, c) * rhs[c]
        end
        rhs[j] = s / elem(lu, j, j)
    end

    return rhs
end

# --- not-a-knot cubic B-spline fit -----------------------------------------

# Builds and factors the (n+2)×(n+2) not-a-knot system for n data points, rows
# ordered [not-a-knot left, interpolation 1..n, not-a-knot right]. The end rows
# impose third-derivative continuity at the first and last interior knots --
# "not a knot" -- which is what buys exact reproduction of a global cubic
# instead of the artificial curvature a natural spline would impose there.
function build_notaknot_factored(::Type{T}, n::Int) where {T<:AbstractFloat}
    lu = BandedLU{T}(n + 2)

    lu[1, 1] = 1
    lu[1, 2] = -4
    lu[1, 3] = 6
    lu[1, 4] = -4
    lu[1, 5] = 1

    sixth = one(T) / T(6)
    four_sixths = T(4) / T(6)
    for i in 1:n
        row = i + 1
        lu[row, i + 0] = sixth
        lu[row, i + 1] = four_sixths
        lu[row, i + 2] = sixth
    end

    rlast = n + 2
    lu[rlast, n - 2] = 1
    lu[rlast, n - 1] = -4
    lu[rlast, n + 0] = 6
    lu[rlast, n + 1] = -4
    lu[rlast, n + 2] = 1

    return factor!(lu)
end

# Fills `rhs` (length n+2) with one line of interpolation data and solves the
# already-factored not-a-knot system in place. The two end-condition rows are
# homogeneous, so their right-hand sides are zero.
function notaknot_solve!(rhs::AbstractVector{T}, lu::BandedLU{T}, f::AbstractVector) where {T}
    rhs[begin] = zero(T)
    rhs[end] = zero(T)
    i = firstindex(rhs) + 1
    for v in f
        rhs[i] = v
        i += 1
    end
    return solve!(lu, rhs)
end

"""
    fit_bspline_1d(f) -> Vector

Fit a uniform not-a-knot cubic B-spline to the `n >= 4` samples `f`, returning
the `n+2` coefficients `c` with `S(x0 + (i-1)h) == f[i]` for every `i`. Only
the sample count enters the system; the grid origin and spacing do not, so they
are not arguments.

The coefficients are exactly what [`BsplineView1`](@ref) and
[`BsplineView3`](@ref) consume.
"""
function fit_bspline_1d(f::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(f)
    n >= 4 || throw(ArgumentError("fit_bspline_1d: n must be >= 4 (got $n)"))
    lu = build_notaknot_factored(T, n)
    return notaknot_solve!(Vector{T}(undef, n + 2), lu, f)
end

"""
    fit_bspline_3d(data, x0, hx, u0, hu, y0, hy) -> BsplineView3

Tensor-product not-a-knot fit of `data`, an `(nx, nu, ny)` array of samples on
the uniform grid described by the origin/spacing pairs, with all extents `>= 4`.

Three passes: along x for every `(u, y)` line, then along u, then along y, each
pass factoring its axis's matrix once and reusing it for every line. Being
independent linear operators on separate axes the passes commute, so the order
is a cache-locality choice only -- x first because it is the fastest-varying
axis.

The returned view owns its `(nx+2, nu+2, ny+2)` coefficient array.
"""
function fit_bspline_3d(data::AbstractArray{T,3}, x0, hx, u0, hu, y0, hy) where {T<:AbstractFloat}
    nx, nu, ny = size(data)
    (nx >= 4 && nu >= 4 && ny >= 4) ||
        throw(ArgumentError("fit_bspline_3d: nx, nu, ny must all be >= 4 (got ($nx, $nu, $ny))"))
    nxp, nup, nyp = nx + 2, nu + 2, ny + 2

    # Pass 1: x. A fixed-(ju, ky) line is contiguous in column-major order.
    lu_x = build_notaknot_factored(T, nx)
    stage1 = Array{T,3}(undef, nxp, nu, ny)
    cx = Vector{T}(undef, nxp)
    for ky in 1:ny, ju in 1:nu
        notaknot_solve!(cx, lu_x, view(data, :, ju, ky))
        stage1[:, ju, ky] .= cx
    end

    # Pass 2: u, over the x-coefficients just produced.
    lu_u = build_notaknot_factored(T, nu)
    stage2 = Array{T,3}(undef, nxp, nup, ny)
    cu = Vector{T}(undef, nup)
    for ky in 1:ny, ix in 1:nxp
        notaknot_solve!(cu, lu_u, view(stage1, ix, :, ky))
        stage2[ix, :, ky] .= cu
    end

    # Pass 3: y.
    lu_y = build_notaknot_factored(T, ny)
    coeffs = Array{T,3}(undef, nxp, nup, nyp)
    cy = Vector{T}(undef, nyp)
    for iu in 1:nup, ix in 1:nxp
        notaknot_solve!(cy, lu_y, view(stage2, ix, iu, :))
        coeffs[ix, iu, :] .= cy
    end

    return BsplineView3(coeffs, x0, hx, u0, hu, y0, hy)
end

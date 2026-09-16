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

# Implementation lands in milestone M2:
#   BandedLU{T}(n), setindex!/getindex within the band, factor!, solve!
#   fit_bspline_1d(f) -> Vector{T}                    (length n+2)
#   fit_bspline_3d(data, x0, hx, u0, hu, y0, hy) -> BsplineView3{T,Array{T,3}}

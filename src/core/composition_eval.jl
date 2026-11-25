# The composition on the device: chemical potentials, mass fractions and the
# mean heavy nucleus, interpolated from the table's own datasets.
#
# This has no C++ counterpart. It is not part of the fitted potential: the
# fields are interpolated trilinearly on the raw table's grid, which keeps the
# mass fractions inside [0, 1] and is what leakage codes do. The conventions
# that differ between tables are resolved once, when the view is built
# (`host/composition_build.jl`), so that a kernel sees one convention for every
# table.

# Field order in the interleaved coefficient array.
const COMP_μ_E = 1
const COMP_μ̂ = 2
const COMP_X_N = 3
const COMP_X_P = 4
const COMP_X_α = 5
const COMP_X_H = 6
const COMP_X_L = 7
const COMP_Ā = 8
const COMP_Z̄ = 9
const COMP_NFIELDS = 9

"""
    Composition{T}

The composition at one state, from [`composition`](@ref). `isbits`.

  * `μ_e`, `μ̂`, `μ_ν` in MeV. `μ_e` includes the electron's rest mass. `μ̂`
    includes the neutron–proton mass difference `Q` ([`q_np`](@ref)) on every
    table, so that `μ_ν = μ_e − μ̂` exactly. The nucleons' own degeneracy,
    without rest masses, is `(μ̂ − Q)/T`.
  * `X_n`, `X_p`, `X_α`, `X_h`, `X_l`: mass fractions of free neutrons, free
    protons, α particles, heavy nuclei, and the light clusters other than α
    (0 on a table without them), each in [0, 1]. They are not renormalized.
  * `Ā`, `Z̄`: mass and charge number of the mean heavy nucleus. Where there
    are no heavy nuclei, `X_h = 0`, `Ā = 1` and `Z̄ = 0`.
"""
struct Composition{T<:Real}
    μ_e::T
    μ̂::T
    μ_ν::T
    X_n::T
    X_p::T
    X_α::T
    X_h::T
    X_l::T
    Ā::T
    Z̄::T
end

"""
    CompositionView

Kernel-side view of a [`CompositionTable`](@ref): the nine interpolated fields,
interleaved as `c[field, iρ, jT, kYₑ]` so that one lookup's corners are
contiguous, plus the grid and the scalars. `Adapt.adapt` moves it to a device
and [`narrow`](@ref) narrows it, as for an `EOSTableView`.
"""
struct CompositionView{T<:AbstractFloat,A<:AbstractArray{T,4}}
    c::A

    x0::T          # log10(ρ★ [g/cc]) of the first node: the table's, plus log10 κ
    hx::T
    u0::T          # log10(T [MeV])
    hu::T
    y0::T          # Yₑ
    hy::T

    κ::T           # the EOS's baryon-mass rescaling
    q_np::T        # m_n − m_p in MeV
    fingerprint::UInt64
end

"""
    q_np(cv) -> T

`Q = m_n − m_p` in MeV, the value the composition assumes: CODATA 2018,
1.29333236 MeV. It is in `μ̂` on every table; subtract it for the nucleons' own
degeneracy, `(μ̂ − Q)/T`, and nowhere else.
"""
q_np(cv::CompositionView) = cv.q_np

fingerprint(cv::CompositionView) = cv.fingerprint

"""
    matches(c, eos) -> Bool

Whether the composition `c` (a [`CompositionView`](@ref) or
[`CompositionTable`](@ref)) was built from the same raw table as `eos`, by
[`fingerprint`](@ref). `false` for an EOS without one, such as the analytic
EOSs. A composition paired with another table's EOS converts `ρ★` with the
wrong κ and reads the wrong composition, silently; a caller should refuse such
a pair.
"""
matches(cv::CompositionView, eos::EOSTableView) = cv.fingerprint != 0 && cv.fingerprint == eos.fingerprint
matches(::CompositionView, ::AbstractEOS) = false

# The cell along one axis and the position in it, for a fractional 0-based
# index `f` clamped to the axis. A non-finite `f` gives a valid cell and a
# non-finite position, so the result is NaN rather than an out-of-bounds read.
@inline clamp01(x::T) where {T<:Real} = clamp(x, zero(T), one(T))

@inline function comp_cell(f::T, n::Int) where {T<:Real}
    fc = clamp(f, zero(T), T(n - 1))
    i = clamp(trunc_floor(fc), 0, n - 2)
    return i, fc - T(i)
end

"""
    composition(cv, ρ★, T_MeV, yₑ) -> Composition

The composition at rescaled density `ρ★`, temperature `T_MeV` and electron
fraction `yₑ`. `T_MeV` is the temperature the recovery landed on,
`EOSPoint.T_MeV`, not `T̂`.

Trilinear in `(log₁₀ρ, log₁₀T, Yₑ)` on the raw table's grid, with
`log₁₀ρ = log₁₀(ρ★/κ)`; each coordinate is clamped to the table's axes, and
there is no extension. The result carries no flags: `EOSPoint.flags` already
marks a state outside the box. `Ā` and `Z̄` are `X_h`-weighted means over the
cell's corners, so that the tables' fill values where there are no heavy
nuclei never enter: `X_h·Ā` and `X_h·Z̄` are the trilinear interpolants of
those products.

Allocation-free and exception-free; runs in a device kernel.
"""
@inline function composition(cv::CompositionView{S}, ρ★::T, T_MeV::T, yₑ::T) where {S,T<:Real}
    fi = (safe_log10(ρ★) - T(cv.x0)) / T(cv.hx)
    fj = (safe_log10(T_MeV) - T(cv.u0)) / T(cv.hu)
    fk = (yₑ - T(cv.y0)) / T(cv.hy)
    return composition_index(cv, fi, fj, fk)
end

"""
    composition_index(cv, fi, fj, fk) -> Composition

[`composition`](@ref) at fractional 0-based grid indices: `fi = 0` is the
table's first density, `fi = nρ − 1` its last. At integer indices the result is
the stored node exactly, which is what lets a test compare against the
datasets bitwise. Not exported.
"""
@inline function composition_index(cv::CompositionView{S}, fi::T, fj::T, fk::T) where {S,T<:Real}
    c = cv.c
    nx, nu, ny = size(c, 2), size(c, 3), size(c, 4)
    i, tx = comp_cell(fi, nx)
    j, tu = comp_cell(fj, nu)
    k, ty = comp_cell(fk, ny)

    o = one(T)
    μe = μh = xn = xp = xa = xl = zero(T)
    den = numA = numZ = zero(T)
    Amin = Zmin = T(Inf)
    Amax = Zmax = T(-Inf)
    @inbounds for dk in 0:1
        wy = dk == 0 ? o - ty : ty
        for dj in 0:1
            wu = dj == 0 ? o - tu : tu
            for di in 0:1
                wx = di == 0 ? o - tx : tx
                w = wx * wu * wy
                ii, jj, kk = i + 1 + di, j + 1 + dj, k + 1 + dk
                μe += w * T(c[COMP_μ_E, ii, jj, kk])
                μh += w * T(c[COMP_μ̂, ii, jj, kk])
                xn += w * T(c[COMP_X_N, ii, jj, kk])
                xp += w * T(c[COMP_X_P, ii, jj, kk])
                xa += w * T(c[COMP_X_α, ii, jj, kk])
                xl += w * T(c[COMP_X_L, ii, jj, kk])
                wh = w * T(c[COMP_X_H, ii, jj, kk])
                A = T(c[COMP_Ā, ii, jj, kk])
                Z = T(c[COMP_Z̄, ii, jj, kk])
                den += wh
                numA += wh * A
                numZ += wh * Z
                # The range of the corners that contribute, to bound the ratio:
                # a subnormal denominator in Float32 can put it anywhere.
                contributes = wh > zero(T)
                Amin = ifelse(contributes, min(Amin, A), Amin)
                Amax = ifelse(contributes, max(Amax, A), Amax)
                Zmin = ifelse(contributes, min(Zmin, Z), Zmin)
                Zmax = ifelse(contributes, max(Zmax, Z), Zmax)
            end
        end
    end

    # No heavy nuclei: the fallback appears only next to X_h = 0 exactly.
    none = !(den >= floatmin(T))
    xh = ifelse(none, zero(T), min(den, o))
    Ā = ifelse(none, o, clamp(numA / den, Amin, Amax))
    Z̄ = ifelse(none, zero(T), clamp(numZ / den, Zmin, Zmax))
    return Composition{T}(μe, μh, μe - μh, clamp01(xn), clamp01(xp), clamp01(xa), xh, clamp01(xl), Ā, Z̄)
end

# Moving and narrowing, as for `EOSTableView` (see `core/adapt.jl`).
function Adapt.adapt_structure(to, cv::CompositionView)
    return CompositionView(Adapt.adapt(to, cv.c), cv.x0, cv.hx, cv.u0, cv.hu, cv.y0, cv.hy, cv.κ, cv.q_np,
                           cv.fingerprint)
end

function CompositionView(c::A, x0, hx, u0, hu, y0, hy, κ, q_np, fingerprint) where {T,A<:AbstractArray{T,4}}
    return CompositionView{T,A}(c, T(x0), T(hx), T(u0), T(hu), T(y0), T(hy), T(κ), T(q_np), UInt64(fingerprint))
end

function narrow(cv::CompositionView, ::Type{S}) where {S<:AbstractFloat}
    return CompositionView(convert(Array{S,4}, cv.c), S(cv.x0), S(cv.hx), S(cv.u0), S(cv.hu), S(cv.y0), S(cv.hy),
                           S(cv.κ), S(cv.q_np), cv.fingerprint)
end

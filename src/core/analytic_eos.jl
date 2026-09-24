# Analytic equations of state: an ideal gas, and a hybrid of a generalized
# piecewise polytrope with a thermal ideal-gas part.
#
# These have no C++ counterpart. They exist so that the solver and the policy
# layer can run the standard relativistic test problems -- shock tubes at
# Γ = 2, TOV stars on a polytrope -- which no tabulated nuclear EOS can host.
#
# Both are closed-form potentials U(ρ, s), so `evaluate` needs no inner solve,
# no clamping and no extension: the formulas hold everywhere, and the box only
# decides which flags are set. κ = 1, so ρ★ = ρ, and Yₑ enters only the bounds.
# The keyword constructors, which validate, live in host/analytic_eos.jl.

"""
The physical box shared by the analytic EOSs, with the entropy window's
bracketing extension precomputed.
"""
struct AnalyticBox{T<:AbstractFloat}
    ρ_lo::T
    ρ_hi::T
    x_lo::T        # log10(ρ_lo)
    x_hi::T
    y_lo::T
    y_hi::T
    s_min::T
    s_max::T
    s_ext_lo::T    # the bracketing window of srange_extended
    s_ext_hi::T
end

function narrow(b::AnalyticBox, ::Type{S}) where {S<:AbstractFloat}
    return AnalyticBox{S}(S(b.ρ_lo), S(b.ρ_hi), S(b.x_lo), S(b.x_hi), S(b.y_lo), S(b.y_hi),
                          S(b.s_min), S(b.s_max), S(b.s_ext_lo), S(b.s_ext_hi))
end

"""Supertype of the closed-form EOSs, which share their box and bounds."""
abstract type AnalyticEOS{T} <: AbstractEOS{T} end

logρ_bounds(eos::AnalyticEOS) = (eos.box.x_lo, eos.box.x_hi)
yₑ_bounds(eos::AnalyticEOS) = (eos.box.y_lo, eos.box.y_hi)
κ(::AnalyticEOS{T}) where {T} = one(T)

"""
    srange(eos::AnalyticEOS, ρ★, yₑ)

The entropy window the EOS was built with. It is the same at every density:
nothing in a closed-form potential makes the physical domain non-rectangular.
"""
@inline function srange(eos::AnalyticEOS{S}, ρ★::T, yₑ::T) where {S,T}
    return SRange{T}(T(eos.box.s_min), T(eos.box.s_max))
end

"""
    srange_extended(eos::AnalyticEOS, ρ★, yₑ)

The entropy window widened by a tenth of its width on each side, with the low
end kept positive. The bracket scan's local ladder is multiplicative in `s`,
so a bracketing window reaching zero entropy would degenerate it.
"""
@inline function srange_extended(eos::AnalyticEOS{S}, ρ★::T, yₑ::T) where {S,T}
    return SRange{T}(T(eos.box.s_ext_lo), T(eos.box.s_ext_hi))
end

# The table's flags, with the table's meaning: where the point lies relative to
# the physical box. Nothing here clamps.
@inline function box_flags(b::AnalyticBox{S}, ρ::T, s::T) where {S,T}
    flags = UInt32(0)
    ρ < b.ρ_lo && (flags |= FLAG_EXT_ρ_LOW)
    ρ > b.ρ_hi && (flags |= FLAG_OOB_ρ_HIGH)
    s < b.s_min && (flags |= FLAG_EXT_S_LOW)
    s > b.s_max && (flags |= FLAG_EXT_S_HIGH)
    return flags
end

# The derived quantities, by exactly the formulas the table uses, so that the
# thermodynamic identities hold to the same rounding on every EOS.
@inline function analytic_point(ρ::T, U::T, U_ρ::T, U_s::T, U_ρρ::T, U_ρs::T, flags::UInt32) where {T}
    p = ρ * ρ * U_ρ
    h = one(T) + U + p / ρ
    cs² = (T(2) * ρ * U_ρ + ρ * ρ * U_ρρ) / h
    # No table temperature and no inner solve; ∂U/∂Yₑ vanishes identically.
    return EOSPoint{T}(U, U_ρ, U_s, U_ρρ, U_ρs, U_s, p, h, cs², T(NaN), zero(T), T(NaN), Int32(0), flags)
end

# ---------------------------------------------------------------------------
# Ideal gas
# ---------------------------------------------------------------------------

"""
    IdealGasEOS{T}(; Γ, K_ref, s_ref, s_window, ρ_bounds, yₑ_bounds)

The Γ-law ideal gas as a potential in density and entropy:

    K(s) = K_ref · exp((Γ−1)(s − s_ref))
    U    = K(s) ρ^(Γ−1) / (Γ−1),    p = K(s) ρ^Γ = (Γ−1) ρ U

so `T̂ = U_s = (Γ−1)U = p/ρ`, `h = 1 + ΓU` and `cs² = Γp/(ρh)`. With `p = ρkT/m`,
`s` is the entropy per particle in units of `k_B`, up to the additive constant
fixed by `(K_ref, s_ref)`.

A polytrope `p = Kρ^Γ` is this EOS at the fixed entropy
[`polytropic_entropy`](@ref)`(eos, K)`. A purely barotropic EOS cannot be
offered, since the solver's Jacobian is singular when `U_s ≡ 0`.

Every argument is a required keyword, in any consistent unit system (κ = 1, so
`ρ★ = ρ`):

  * `s_window = (s_min, s_max)` is the physical entropy window, with
    `0 < s_min`. The bracket scan works in relative steps of `s`, so choose
    `s_ref` to keep `s` of order 1 to 10, as for an entropy in `k_B` per baryon.
  * `ρ_bounds = (ρ_min, ρ_max)` sets the policy's default atmosphere and
    density ceiling.
  * `yₑ_bounds` only bounds the passive `Yₑ`; the EOS does not depend on it.

The constructor throws an `ArgumentError` for invalid parameters, including any
that make the gas acausal (`cs² ≥ 1`) on the box, which needs `Γ > 2`.
`IdealGasEOS(; ...)` means `IdealGasEOS{Float64}(; ...)`.
"""
struct IdealGasEOS{T<:AbstractFloat} <: AnalyticEOS{T}
    Γ::T
    log_K_ref::T
    s_ref::T
    box::AnalyticBox{T}
end

function evaluate(eos::IdealGasEOS{S}, ρ★::T, s::T, yₑ::T, u_guess::T) where {S,T}
    g = T(eos.Γ) - one(T)
    # One exp of a sum of logs, so that cgs parameters (ρ^Γ ~ 1e45) cannot
    # overflow a Float32 intermediate when U itself is representable.
    U = exp(T(eos.log_K_ref) + g * (s - T(eos.s_ref) + safe_log(ρ★))) / g
    U_ρ = g * U / ρ★
    U_s = g * U
    U_ρρ = g * (g - one(T)) * U / (ρ★ * ρ★)
    U_ρs = g * g * U / ρ★
    return analytic_point(ρ★, U, U_ρ, U_s, U_ρρ, U_ρs, box_flags(eos.box, ρ★, s))
end

function narrow(eos::IdealGasEOS, ::Type{S}) where {S<:AbstractFloat}
    return IdealGasEOS{S}(S(eos.Γ), S(eos.log_K_ref), S(eos.s_ref), narrow(eos.box, S))
end

# ---------------------------------------------------------------------------
# Hybrid: generalized piecewise polytrope plus a thermal ideal gas
# ---------------------------------------------------------------------------

"""
One piece of a generalized piecewise polytrope, `p = Kρ^Γ + Λ`, stored scaled
to a reference density `ρ̄` so that `Kρ^(Γ−1) = c·(ρ/ρ̄)^(Γ−1)` stays in range
in Float32 even with cgs parameters.
"""
struct GPPPiece{T<:AbstractFloat}
    ρ_lo::T    # where the piece starts; 0 for the first
    ρ̄::T
    c::T       # K ρ̄^(Γ−1)
    Γ::T
    a::T
    Λ::T
end

narrow(q::GPPPiece, ::Type{S}) where {S<:AbstractFloat} =
    GPPPiece{S}(S(q.ρ_lo), S(q.ρ̄), S(q.c), S(q.Γ), S(q.a), S(q.Λ))

"""
    HybridEOS{T,N}(; ρ_breaks, K₀, Γs, Γ_th, K_th_ref, s_ref, s_window, ρ_bounds, yₑ_bounds)

A cold generalized piecewise polytrope plus a thermal ideal-gas part:

    U = U_cold(ρ) + U_th(ρ, s),   U_th = K_th(s) ρ^(Γ_th−1) / (Γ_th−1)
    K_th(s) = K_th_ref · exp((Γ_th−1)(s − s_ref))

so `p = p_cold + (Γ_th−1)ρU_th` and `T̂ = U_s = (Γ_th−1)U_th > 0`.

The cold part is the generalized piecewise polytrope of O'Boyle, Markakis,
Stergioulas & Read (2020), [Phys. Rev. D 102, 083027](https://doi.org/10.1103/PhysRevD.102.083027).
Piece `i`, with exponent `Γs[i]`, has

    p_cold = Kᵢ ρ^Γᵢ + Λᵢ,    U_cold = Kᵢ ρ^(Γᵢ−1)/(Γᵢ−1) + aᵢ − Λᵢ/ρ

The first piece starts at `ρ = 0` with `K₀` and `Λ = a = 0`, so `ε_cold(0) = 0`.
At each break the next piece's `K`, `Λ` and `a` follow from continuity of
`p`, `ε` *and* `dp/dρ` (their eqs. 4.6–4.8). The sound speed is therefore
continuous, unlike in the classic piecewise polytrope of Read et al. (2009),
whose jumps in `cs²` make the solver's Jacobian discontinuous. The same number
of parameters is free, but the `Kᵢ` are derived differently: parameter sets
fitted for classic piecewise polytropes do not carry over. Their Tables II and
III give fits for an SLy crust and for about 25 nuclear EOSs.

`ρ_breaks` has `N−1` entries, strictly increasing; `Γs` has `N`. Every `Γᵢ`
other than the first may be anything but 0 or 1, including negative with a
negative `Kᵢ`, as in the paper's crust fit. The remaining keywords are as for
[`IdealGasEOS`](@ref).

At low entropy `U_th ≪ U_cold`, and entropy is then only weakly determined by
the energy: a conservative-to-primitive inversion recovers `ρ`, `p` and the
velocity accurately, but `s` only to a relative accuracy of about
`ϵ·U/(s·U_s)`. This is physics, not a solver limitation, and `s_min` controls
how far into that regime the EOS reaches.

The constructor throws an `ArgumentError` for invalid or acausal parameters.
`HybridEOS(; ...)` means Float64.
"""
struct HybridEOS{T<:AbstractFloat,N} <: AnalyticEOS{T}
    pieces::NTuple{N,GPPPiece{T}}
    Γ_th::T
    log_K_th_ref::T
    s_ref::T
    box::AnalyticBox{T}
end

function evaluate(eos::HybridEOS{S,N}, ρ★::T, s::T, yₑ::T, u_guess::T) where {S,N,T}
    ρ = ρ★

    # Select the piece without dynamic indexing. A NaN density matches no
    # break, lands in the first piece and propagates.
    @inbounds q = eos.pieces[1]
    ρ̄, c, Γ, a, Λ = T(q.ρ̄), T(q.c), T(q.Γ), T(q.a), T(q.Λ)
    @inbounds for i in 2:N
        q = eos.pieces[i]
        in_piece = ρ >= T(q.ρ_lo)
        ρ̄ = ifelse(in_piece, T(q.ρ̄), ρ̄)
        c = ifelse(in_piece, T(q.c), c)
        Γ = ifelse(in_piece, T(q.Γ), Γ)
        a = ifelse(in_piece, T(q.a), a)
        Λ = ifelse(in_piece, T(q.Λ), Λ)
    end

    # Cold part. Kρ^(Γ−1), via the scaled form; exp(-Inf) = 0 covers ρ = 0.
    k = c * exp((Γ - one(T)) * safe_log(ρ / ρ̄))
    U_cold = k / (Γ - one(T)) + a - Λ / ρ
    U_cold_ρ = k / ρ + Λ / (ρ * ρ)
    U_cold_ρρ = (Γ - T(2)) * k / (ρ * ρ) - T(2) * Λ / (ρ * ρ * ρ)

    # Thermal part, as for the ideal gas.
    g = T(eos.Γ_th) - one(T)
    U_th = exp(T(eos.log_K_th_ref) + g * (s - T(eos.s_ref) + safe_log(ρ))) / g

    U = U_cold + U_th
    U_ρ = U_cold_ρ + g * U_th / ρ
    U_s = g * U_th
    U_ρρ = U_cold_ρρ + g * (g - one(T)) * U_th / (ρ * ρ)
    U_ρs = g * g * U_th / ρ
    return analytic_point(ρ, U, U_ρ, U_s, U_ρρ, U_ρs, box_flags(eos.box, ρ, s))
end

function narrow(eos::HybridEOS{T,N}, ::Type{S}) where {T,N,S<:AbstractFloat}
    return HybridEOS{S,N}(map(q -> narrow(q, S), eos.pieces), S(eos.Γ_th), S(eos.log_K_th_ref),
                          S(eos.s_ref), narrow(eos.box, S))
end

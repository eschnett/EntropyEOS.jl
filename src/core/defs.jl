# Shared definitions for the kernel-side code in `src/core/`.
#
# Translated from `entropy_eos/core/defs.hpp`.
#
# Everything here must be usable inside a GPU kernel: no allocation, no
# exceptions, no globals that are not compile-time constants.

# ---------------------------------------------------------------------------
# Flag bits
#
# Evaluation and policy code reports what happened at a point by OR-ing bits
# together, rather than by throwing or logging. `UInt32` matches the C++
# `unsigned`, and keeps `flags |= FLAG_MAXITER` from widening.
# ---------------------------------------------------------------------------

# Bits 0-5: what the EOS evaluation found. Bits 6-7 are reserved so that the
# policy bits below stay a contiguous maskable group.
const FLAG_CLAMP_YE     = UInt32(1) << 0   # Yₑ was clamped into range
const FLAG_EXT_S_LOW    = UInt32(1) << 1   # point used the low-entropy extension
const FLAG_EXT_S_HIGH   = UInt32(1) << 2   # point used the high-entropy extension
const FLAG_EXT_ρ_LOW    = UInt32(1) << 3   # point used the low-density extension
const FLAG_OOB_ρ_HIGH   = UInt32(1) << 4   # ρ above the table's high edge (no extension)
const FLAG_MAXITER      = UInt32(1) << 5   # the inner T-solve hit its iteration cap

# Bits 8-15: what the *policy layer* did to a state. Kept separate from the
# bits above so a caller can distinguish "the table extension was used" from
# "the state was repaired or excised" by masking.
const FLAG_POL_ATMOSPHERE = UInt32(1) << 8    # whole state replaced by the atmosphere
const FLAG_POL_CEILING    = UInt32(1) << 9    # a collapse ceiling was exceeded
const FLAG_POL_S_FLOORED  = UInt32(1) << 10   # s raised to the physical srange minimum
const FLAG_POL_S_CEILED   = UInt32(1) << 11   # s lowered to the physical srange maximum
const FLAG_POL_W_CAPPED   = UInt32(1) << 12   # rapidity clamped into [0, w_cap]
const FLAG_POL_ρ_CLAMPED  = UInt32(1) << 13   # ρ clamped without a full atmosphere reset
const FLAG_POL_YE_CLAMPED = UInt32(1) << 14   # Yₑ clamped into the table's range
const FLAG_POL_NONFINITE  = UInt32(1) << 15   # non-finite (or D <= 0) input was seen

"""
Mask of every policy bit: `flags & FLAG_POL_ANY` answers "did the policy layer
touch this point?" without enumerating the bits.
"""
const FLAG_POL_ANY = FLAG_POL_ATMOSPHERE | FLAG_POL_CEILING | FLAG_POL_S_FLOORED |
                     FLAG_POL_S_CEILED | FLAG_POL_W_CAPPED | FLAG_POL_ρ_CLAMPED |
                     FLAG_POL_YE_CLAMPED | FLAG_POL_NONFINITE

# ---------------------------------------------------------------------------
# Outcome enums
#
# `EnumX` rather than `@enum` so the members are namespaced (`Status.ok`) and
# do not leak into the module's flat namespace.
# ---------------------------------------------------------------------------

"""
Coarse outcome of an operation. `ok` means nothing notable happened,
`repaired` means the result is usable but something was flagged or adjusted,
`fatal` means the result must not be trusted.
"""
@enumx Status ok repaired fatal

"""
How `con2prim` terminated. Note that a failure still returns a fully populated
best-effort state; nothing is deliberately NaN-poisoned.
"""
@enumx C2PResult converged_newton converged_fallback failed_no_bracket failed_max_iter

# ---------------------------------------------------------------------------
# Domain-safe math
#
# The C++ relies on C semantics where Julia throws. `log10` of a negative
# number returns NaN in C and raises `DomainError` in Julia; the callers here
# feed it densities that may legitimately be non-positive and rely on the
# subsequent clamp to handle it. Same story for `sqrt`, and for truncating a
# non-finite float to an integer.
#
# `Base.isnan` and `Base.isfinite` are already exactly the self-comparison
# tricks the C++ hand-rolls (`v != v`, `(v - v) == 0`), so they are used
# directly. `Base.clamp` likewise passes NaN through unchanged, matching the
# C++ helper it replaces.
#
# `@fastmath` must never be applied to any of this: it folds the NaN and
# finiteness probes to the wrong answer. This is the same prohibition the C++
# states for `-ffast-math`.
# ---------------------------------------------------------------------------

"""
    safe_log10(x)

`log10` with C semantics: `-Inf` at zero, `NaN` below zero, never a throw.
"""
@inline safe_log10(x::T) where {T<:Real} = x < zero(T) ? T(NaN) : log10(x)

"""
    safe_log(x)

`log` with C semantics: `-Inf` at zero, `NaN` below zero, never a throw.
"""
@inline safe_log(x::T) where {T<:Real} = x < zero(T) ? T(NaN) : log(x)

"""
    safe_sqrt(x)

`sqrt` with C semantics: `NaN` below zero, never a throw.
"""
@inline safe_sqrt(x::T) where {T<:Real} = x < zero(T) ? T(NaN) : sqrt(x)

"""
    trunc_floor(x)

`static_cast<int>(std::floor(x))`, total on all inputs. Callers always clamp
the result into a valid index range, so the value returned for a non-finite
argument only has to be harmless, not meaningful.
"""
@inline trunc_floor(x::AbstractFloat) = unsafe_trunc(Int, floor(x))

# Generic fallback, reached by dual numbers under automatic differentiation.
# Unlike the float path this is not total -- it throws on a non-finite argument
# -- which is acceptable because AD is a host-side analysis tool, never a
# kernel path, and a non-finite coordinate cannot be differentiated anyway.
@inline trunc_floor(x::Real) = floor(Int, x)

# ---------------------------------------------------------------------------
# Numeric constants and tolerances
#
# These are functions of the scalar type rather than bare literals: a bare
# `Float64` literal multiplied into `Float32` arithmetic silently widens the
# whole computation back to `Float64`.
#
# Every value below was *measured* at `Float64` in the C++ (see the option
# comments in `core/con2prim.hpp`). The `Float64` methods therefore reproduce
# those decisions exactly. `con2prim_tol` also has a `Float32` method measured
# in this port's own accuracy study. The generic fallbacks scale with `eps(T)`
# so that any other narrower type produces something runnable -- they are not
# validated.
# ---------------------------------------------------------------------------

"""ln(10), to `Float64` precision, as the C++ `constexpr` is."""
@inline ln10(::Type{T}) where {T<:Real} =
    T(2.302585092994045684017991454684364207601101488628772976033)

"""Convergence tolerance for `con2prim`'s two normalized residuals."""
@inline con2prim_tol(::Type{Float64}) = 1.0e-12
# Measured (study/float32_accuracy.jl, docs/src/precision.md): evaluated in
# Float32, the energy residual has a noise floor of 2.0e-5 to 3.8e-5 at p99 on
# the real tables -- the log-shifted energy fit and the spline sums lose ~250
# ulps. A tolerance below that cannot be met: at 64 eps Newton stalled and
# 0.4-0.7% of warm starts failed. 512 eps is the smallest power of two above
# the floor on every table, and brings Float32 failure counts to within a small
# factor of Float64's.
@inline con2prim_tol(::Type{Float32}) = 512 * eps(Float32)
@inline con2prim_tol(::Type{T}) where {T<:AbstractFloat} = max(T(1.0e-12), T(64) * eps(T))
@inline con2prim_tol(::Type{T}) where {T<:Real} = T(con2prim_tol(Float64))

"""Relative floor on τ in the energy residual's normalization."""
@inline tau_floor_rel(::Type{Float64}) = 1.0e-16
@inline tau_floor_rel(::Type{T}) where {T<:AbstractFloat} = max(T(1.0e-16), eps(T))
@inline tau_floor_rel(::Type{T}) where {T<:Real} = T(tau_floor_rel(Float64))

"""Residual tolerance factor for the inner T-solve: `|g| <= tol * max(|s|, 1)`."""
@inline tsolve_residual_tol(::Type{Float64}) = 1.0e-12
@inline tsolve_residual_tol(::Type{T}) where {T<:AbstractFloat} = max(T(1.0e-12), T(64) * eps(T))
@inline tsolve_residual_tol(::Type{T}) where {T<:Real} = T(tsolve_residual_tol(Float64))

"""Step tolerance factor for the inner T-solve: `|du| <= tol * max(|u|, 1)`."""
@inline tsolve_step_tol(::Type{Float64}) = 1.0e-13
@inline tsolve_step_tol(::Type{T}) where {T<:AbstractFloat} = max(T(1.0e-13), T(8) * eps(T))
@inline tsolve_step_tol(::Type{T}) where {T<:Real} = T(tsolve_step_tol(Float64))

"""Relative residual at which the cold seed's EOS-free solve for `z` stops."""
@inline seed_z_tol(::Type{Float64}) = 1.0e-14
@inline seed_z_tol(::Type{T}) where {T<:AbstractFloat} = max(T(1.0e-14), T(4) * eps(T))
@inline seed_z_tol(::Type{T}) where {T<:Real} = T(seed_z_tol(Float64))

"""Rapidity a negative Newton step is reflected to, instead of exactly zero."""
@inline tiny_w(::Type{T}) where {T<:Real} = T(1.0e-10)

"""Bounds of the bracket scan's geometric ladder of relative offsets."""
@inline scan_delta_min(::Type{T}) where {T<:Real} = T(1.0e-5)
@inline scan_delta_max(::Type{T}) where {T<:Real} = T(0.6)

"""
Largest log excursion the low-density log-σ tail will accept before falling
back to the plain linear tail. Returns `false` for NaN inputs by construction,
which is intended.
"""
@inline xlow_log_excursion_max(::Type{T}) where {T<:Real} = T(40)

"""
Degeneracy guard for the perpendicular direction in `prim2con`.

`1e-300` underflows to exactly zero in `Float32`, which would turn the guard
off and let `Inf * 0` produce a NaN; scale off `floatmin` instead for any type
that cannot represent it.
"""
@inline perp_degenerate(::Type{Float64}) = 1.0e-300
@inline perp_degenerate(::Type{T}) where {T<:AbstractFloat} = T(1.0e4) * floatmin(T)
@inline perp_degenerate(::Type{T}) where {T<:Real} = T(perp_degenerate(Float64))

"""Guard against dividing by a vanishing velocity magnitude."""
@inline tiny_denom(::Type{Float64}) = 1.0e-300
@inline tiny_denom(::Type{T}) where {T<:AbstractFloat} = T(1.0e4) * floatmin(T)
@inline tiny_denom(::Type{T}) where {T<:Real} = T(tiny_denom(Float64))

"""Upper bound on the bracket scan's candidate count; sizes its stack scratch."""
const BRACKET_SCAN_MAX = 33

"""Iteration cap for the EOS-free cold-seed solve for `z`."""
const SEED_SCALAR_ITERS = 40

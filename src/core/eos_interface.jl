# The EOS interface: what prim2con, con2prim and the state policy require of an
# equation of state, and the two result types every EOS hands back.
#
# The solvers were written against the table, but they only ever touch it
# through `evaluate`, `srange`, `srange_extended` and the box bounds. Naming
# that surface lets an analytic EOS use the same solver and the same
# never-fails policy.

"""
    AbstractEOS{T}

An equation of state as a single thermodynamic potential `U(ρ★, s, Yₑ)`: the
specific internal energy as a function of density, specific entropy and
electron fraction. `T` is the storage type of its parameters; the working type
is whatever the call's arguments promote to.

A subtype must be `isbits` (or become so under `Adapt.adapt`), and must
provide, allocation-free and without ever throwing:

  * `evaluate(eos, ρ★::T, s::T, yₑ::T, u_guess::T) -> EOSPoint{T}` for any
    `T<:Real`, including `ForwardDiff.Dual`. `u_guess` is the previous call's
    `u_solved`, or `NaN` for none.
  * `srange(eos, ρ★, yₑ) -> SRange`: the physical entropy window, finite.
  * `srange_extended(eos, ρ★, yₑ) -> SRange`: a finite window containing
    `srange`, on which `U` is strictly increasing in `s`. The solver brackets
    on it.
  * `logρ_bounds(eos) -> (x_lo, x_hi)`: the physical box in `log10(ρ★)`.
  * `yₑ_bounds(eos) -> (y_lo, y_hi)`: the physical box in `Yₑ`.
  * `EntropyEOS.κ(eos)`: the baryon-mass rescaling, `ρ★ = κρ`.

Every returned `EOSPoint` must satisfy the thermodynamic identities
`p = ρ²U_ρ`, `h = 1 + U + p/ρ`, `cs² = (2ρU_ρ + ρ²U_ρρ)/h` and `T̂ = U_s`, with
`U_s > 0`. `U_ρs` must be consistent with the rest: con2prim's Jacobian uses
`cs²`, `U_s` and `U_ρs`. A barotropic EOS (`U_s ≡ 0`) cannot satisfy this --
the solver's Jacobian is singular there -- which is why a polytrope is offered
as an ideal gas at fixed entropy instead.
"""
abstract type AbstractEOS{T<:AbstractFloat} end

Base.eltype(::AbstractEOS{T}) where {T} = T

"""
    logρ_bounds(eos) -> (x_lo, x_hi)

The physical density box of `eos` in `x = log10(ρ★)`. The state policy's
default atmosphere and density ceiling are its two ends.
"""
function logρ_bounds end

"""
    yₑ_bounds(eos) -> (y_lo, y_hi)

The physical electron-fraction box of `eos`. The state policy clamps `Yₑ` to it.
"""
function yₑ_bounds end

"""
    EntropyEOS.κ(eos)

The baryon-mass rescaling: the EOS works in `ρ★ = κρ`, re-zeroed so that
`ρ★(1 + U) = ρ(1 + ε)`. It is 1 for the analytic EOSs. Deliberately not
exported, since `κ` is too common a name to take from a caller.
"""
function κ end

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
consistent, and conflating them is a silent physics bug. An analytic EOS has no
table temperature and returns `T_MeV = NaN`.

`u_solved` is `log10(T_MeV)`, threaded back out as the warm start for the next
call -- there is no hidden mutable state anywhere, which is what makes
evaluation re-entrant and safe to call in parallel across grid points. An EOS
that needs no inner solve returns `NaN`.
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

# Mixed argument types promote to one working type, so that a Float64 caller
# can evaluate a Float32 table, and a Dual caller a Float64 one.
evaluate(eos::AbstractEOS{S}, ρ★, s, yₑ, u_guess) where {S} =
    evaluate(eos, promote(float(ρ★), float(s), float(yₑ), float(u_guess))...)

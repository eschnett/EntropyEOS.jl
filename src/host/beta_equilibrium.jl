# Cold, neutrinoless β-equilibrium: the composition of a cold neutron star.
#
# Host-side, for initial data. Three choices shape this file; CODE.md
# ("β-equilibrium") has the measurements behind them.
#
# * In a star the composition is the one of least Gibbs energy per baryon at
#   fixed pressure, not of least energy at fixed density: gravity stacks phases
#   rather than mixing them. Where the composition changes, the density jumps
#   at constant pressure. So the result is a slice built once, keyed by
#   pressure, with its jumps explicit.
#
# * The selection uses the fitted EOS's own pressure and Gibbs enthalpy, which
#   is what a hydro code sees, rather than a convex hull of its energy: the
#   tables' energy is not consistent enough with their chemical potentials for
#   a hull's slopes to match the EOS's pressure.
#
# * The criterion μ_ν = 0 is read from the table's chemical potentials, not
#   from the fitted μ̃, which is off by MeV on the cold edge. The Gibbs energy
#   is corrected for the rest masses the tables' ε leaves out, so that its
#   Yₑ-derivative is that same μ_ν.

"""
    BetaEquilibrium{T}

One state of a [`ColdBetaSlice`](@ref), as [`beta_equilibrium`](@ref) returns it.

  * `ρ★`, `yₑ`, `s`: the state. `s = srange(eos, ρ★, yₑ).s_min`, the coldest
    tabulated temperature.
  * `point`: the EOS evaluated there, giving `p`, `U`, `h`, `cs²`, ….
  * `g`: the Gibbs free energy per unit rest mass,
    `1 + U − T̂s + p/ρ★ + c_E·Yₑ` with `c_E` the slice's `energy_offset` in
    these units. As `T → 0` it is `point.h` corrected for the rest masses the
    table's energy leaves out. It, not `point.h`, is continuous across a phase
    transition and non-decreasing along the slice, and `dg = dp/ρ★` there.
  * `flags`: `point.flags`, plus `FLAG_CLAMP_YE` if β-equilibrium lies outside
    the table's `Yₑ` range and `yₑ` is that range's edge.
"""
struct BetaEquilibrium{T<:Real}
    ρ★::T
    yₑ::T
    s::T
    point::EOSPoint{T}
    g::T
    flags::UInt32
end

"""
    ColdBetaJump

A phase transition on a [`ColdBetaSlice`](@ref): at pressure `p` the density
jumps from `ρ★_lo` to `ρ★_hi` and the composition from `yₑ_lo` to `yₑ_hi`. No
equilibrium state has a density strictly in between.

`g` is the Gibbs free energy there. `g_mismatch` is its jump, high side minus
low side: zero (to rounding) where two branches' `g` cross. It is nonzero only
where no crossing exists: a branch that appears at `p` with `g` already below
the current one's (seen once, on SRO LS220, at −1.2·10⁻⁷), or a branch that ends
at `p` still below the next. The slice's `g` is held at the low side's value
across a negative mismatch, so that it stays non-decreasing.
"""
struct ColdBetaJump
    p::Float64
    g::Float64
    g_mismatch::Float64
    ρ★_lo::Float64
    ρ★_hi::Float64
    yₑ_lo::Float64
    yₑ_hi::Float64
end

"""
    ColdBetaSlice

The cold, neutrinoless β-equilibrium states of a tabulated EOS, built by
[`cold_beta_slice`](@ref) and queried with [`beta_equilibrium`](@ref).

Its vertices are stored in order of increasing density, and along them `ρ★`,
`p` and `g` are non-decreasing; at a phase transition two vertices share `p`
and `g`.

  * `ρ★`, `yₑ`, `s`, `p`, `h`, `g`, `clamped`: the vertices' states (`h` is
    the EOS's enthalpy, `g` the Gibbs free energy of [`BetaEquilibrium`](@ref));
  * `jumps`: the phase transitions, in order of increasing pressure;
  * `munu_offset`: the constant `c` in `μ_ν = mu_e − muhat + c`, in MeV;
  * `energy_offset`: the correction `c_E` in `E = m_B ε + c_E·Yₑ`, in MeV per
    baryon: the energy whose `Yₑ`-derivative is `μ_ν`;
  * `munu_mismatch`: the fraction of the coldest row's nodes where the table's
    own `munu` differs from `mu_e − muhat + c` by more than 1 keV.
"""
struct ColdBetaSlice{V<:EOSTableView}
    eos::V
    logρ::Vector{Float64}       # raw log10 ρ
    Yₑ::Vector{Float64}
    μν::Matrix{Float64}         # μ_ν on the coldest row, nρ × nYₑ, MeV
    munu_offset::Float64
    energy_offset::Float64
    munu_mismatch::Float64
    ρ★::Vector{Float64}
    yₑ::Vector{Float64}
    s::Vector{Float64}
    p::Vector{Float64}
    h::Vector{Float64}
    g::Vector{Float64}
    clamped::Vector{Bool}
    jump_after::Vector{Bool}    # between vertex i and i + 1
    jumps::Vector{ColdBetaJump}
end

function Base.show(io::IO, sl::ColdBetaSlice)
    print(io, "ColdBetaSlice(", length(sl.ρ★), " vertices, ", length(sl.jumps), " jumps, ρ★ ∈ [",
          sl.ρ★[1], ", ", sl.ρ★[end], "], munu_offset = ", sl.munu_offset, " MeV, energy_offset = ",
          sl.energy_offset, " MeV)")
    return nothing
end

"""
    cold_beta_slice(table::RawTable, eos::EOSTable; refine = 8,
                    munu_offset = nothing, energy_offset = nothing) -> ColdBetaSlice

The cold, neutrinoless β-equilibrium slice of `eos`, which must have been built
from `table`.

**Criterion.** `μ_ν = mu_e − muhat + munu_offset = 0` on the table's coldest
row, interpolated bilinearly in `log10 ρ` and `Yₑ`. `munu_offset` defaults to
the median of `munu − (mu_e − muhat)` over that row: 0 on the Hempel and SRO
tables, `−Q = −1.293` MeV on the O'Connor–Ott LS220 table. Where the table's
own `munu` disagrees with that, as O'Connor–Ott LS220's does below about
10⁸ g/cm³, the disagreement is reported in `munu_mismatch` and not followed.

**Selection.** At `refine` densities per table cell, every upward zero crossing
of `μ_ν` in `Yₑ`, and a table edge where `μ_ν` points outward, is placed on the
fitted EOS's cold edge. Neighbouring samples' crossings that are each other's
nearest in `Yₑ` form branches, which are cut where the pressure stops rising.
At each pressure the branch with the least Gibbs free energy `g` is the
equilibrium; where that changes, `g` is equal on both branches at one pressure,
located by bisection on exact states, and the density jumps there.

**Energy correction.** The tables' ε leaves out rest masses that `μ_ν`
includes, so `g` uses `E = m_B ε + energy_offset·Yₑ`. By default
`energy_offset = munu_offset + mₑc²`, since the tables' `mu_e` includes the
electron's rest mass and their ε does not. That is checked: above 10¹³·⁵ g/cm³,
`(mu_e − muhat) − m_B ∂ε/∂Yₑ` must be `mₑc²` to within 20 keV. A table that
fails, as O'Connor–Ott LS220 does, throws `ArgumentError` unless
`energy_offset` is given explicitly.

"Cold" means the table's lowest temperature, not `T = 0`. The table must carry
`mu_e` and `muhat`, and `munu` unless `munu_offset` is given.
"""
function cold_beta_slice(table::RawTable, eos::EOSTable; refine::Integer=8, munu_offset=nothing,
                         energy_offset=nothing)
    v = eos.view
    check_beta_inputs(table, v)
    refine >= 1 || throw(ArgumentError("cold_beta_slice: refine must be >= 1"))
    logρ = Vector{Float64}(table.logρ)
    Yₑ = Vector{Float64}(table.Yₑ)
    mue, muh = field(table, "mu_e"), field(table, "muhat")
    d = [Float64(mue[i, 1, k]) - Float64(muh[i, 1, k]) for i in eachindex(logρ), k in eachindex(Yₑ)]
    all(isfinite, d) || throw(ArgumentError("cold_beta_slice: non-finite mu_e or muhat on the coldest row"))

    # The criterion's offset, from the table's own munu where it has one.
    has_munu = has_field(table, "munu")
    munu = has_munu ? Float64.(field(table, "munu")[:, 1, :]) : zeros(0, 0)
    c_μ = if munu_offset === nothing
        has_munu || throw(ArgumentError("cold_beta_slice: table has no 'munu'; pass munu_offset"))
        median_of(vec(munu .- d))
    else
        Float64(munu_offset)
    end
    mismatch = has_munu ? count(>(1.0e-3), abs.(munu .- (d .+ c_μ))) / length(d) : NaN
    c_E = if energy_offset === nothing
        δ, spread, n = electron_offset(table, d, eos.m_B_table_g)
        n >= 10 && abs(δ - M_ELECTRON_MEV) <= 0.02 && spread <= 0.05 ||
            throw(ArgumentError("cold_beta_slice: above 10^13.5 g/cm³, (mu_e − muhat) − m_B ∂ε/∂Yₑ has median " *
                                "$δ MeV and interquartile range $spread MeV over $n nodes, not the electron's " *
                                "rest mass $M_ELECTRON_MEV MeV; the table's energy and chemical potentials do not " *
                                "fit one convention. Pass energy_offset (MeV) explicitly."))
        c_μ + M_ELECTRON_MEV
    else
        Float64(energy_offset)
    end
    μν = d .+ c_μ
    ctx = BranchCtx(v, logρ, Yₑ, μν, log10(Float64(v.κ)), Float64(v.conv_t) / Float64(v.κ) * c_E)

    # Candidates at refined densities across the box, each on the fitted cold edge.
    cands = BetaCandidate[]
    first_of = Int[]                     # index of each sample's first candidate
    for x in refined_axis(logρ, refine)
        push!(first_of, length(cands) + 1)
        for (y, clamped) in mu_roots(ctx, x)
            push!(cands, beta_candidate(ctx, x + ctx.lκ, y, clamped))
        end
    end
    push!(first_of, length(cands) + 1)

    pieces = stable_pieces(cands, first_of)
    isempty(pieces) && throw(ArgumentError("cold_beta_slice: no branch of β-equilibrium states found"))
    verts, jump_after, jumps = gibbs_envelope(ctx, pieces)

    ρ★s, ps, gs = [c.ρ★ for c in verts], [c.pt.p for c in verts], [c.g for c in verts]
    # The two sides of a jump are found separately and agree in p only to
    # rounding; they are one pressure by construction. g is equal too at a
    # crossing, and is held at the low side's value where a branch born below
    # it would make it decrease (`g_mismatch` keeps the difference).
    for (i, j) in enumerate(findall(jump_after))
        abs(ps[j + 1] - ps[j]) <= 1.0e-9 * ps[j] && (ps[j + 1] = ps[j])
        jumps[i].g_mismatch <= 0 && (gs[j + 1] = gs[j])
        J = jumps[i]
        jumps[i] = ColdBetaJump(ps[j], gs[j], J.g_mismatch, J.ρ★_lo, J.ρ★_hi, J.yₑ_lo, J.yₑ_hi)
    end
    issorted(ρ★s) && issorted(ps) && issorted(gs) ||
        throw(ArgumentError("cold_beta_slice: the slice is not monotone in ρ★, p and g"))
    return ColdBetaSlice(v, logρ, Yₑ, μν, c_μ, c_E, mismatch, ρ★s, [c.y for c in verts], [c.s for c in verts],
                         ps, [c.pt.h for c in verts], gs, [c.clamped for c in verts], jump_after, jumps)
end

"""
    beta_equilibrium(slice::ColdBetaSlice; ρ★ | p | g) -> BetaEquilibrium

The cold β-equilibrium state at a given rescaled density `ρ★`, pressure `p`
(in `EOSPoint` units) or Gibbs free energy `g`; pass exactly one.

A TOV solve wants `p` or `g`: each gives one state, also across a phase
transition, where the state is the low-density phase below the jump and the
high-density phase at and above it. A density inside a jump's gap has no
equilibrium state and throws `ArgumentError`, as does a value outside the
slice. Between vertices the state is solved for on the branch the vertices lie
on, so `μ_ν = 0` there too.
"""
function beta_equilibrium(sl::ColdBetaSlice; ρ★=nothing, p=nothing, g=nothing)
    count(!isnothing, (ρ★, p, g)) == 1 ||
        throw(ArgumentError("beta_equilibrium: pass exactly one of ρ★, p and g"))
    key, arr, val = ρ★ !== nothing ? (:ρ★, sl.ρ★, Float64(ρ★)) :
                    p !== nothing ? (:p, sl.p, Float64(p)) : (:g, sl.g, Float64(g))
    n = length(arr)
    arr[1] <= val <= arr[n] ||
        throw(ArgumentError("beta_equilibrium: $key = $val is outside the slice's range [$(arr[1]), $(arr[n])]"))
    i = clamp(searchsortedlast(arr, val), 1, n - 1)
    val == arr[i + 1] && return vertex_state(sl, i + 1)
    val == arr[i] && return vertex_state(sl, i)
    if sl.jump_after[i]
        # Only a density, or a Gibbs energy across a jump that has a mismatch,
        # can fall strictly inside a jump.
        key === :ρ★ && throw(ArgumentError("beta_equilibrium: ρ★ = $val lies in the density gap of a phase " *
                                           "transition, between $(arr[i]) and $(arr[i + 1])"))
        return vertex_state(sl, i + 1)
    end
    ctx = slice_ctx(sl)
    key === :ρ★ && return result(branch_candidate(sl, ctx, i, log10(val)))
    # Bisection in log ρ★ on the branch between the two vertices; of the last
    # bracket, the end nearer in the key is returned.
    a, b = slice_candidate(sl, ctx, i), slice_candidate(sl, ctx, i + 1)
    f(c) = key === :p ? c.pt.p : c.g
    lo, hi = a, b
    for _ in 1:100
        mid = (log10(lo.ρ★) + log10(hi.ρ★)) / 2
        (mid == log10(lo.ρ★) || mid == log10(hi.ρ★)) && break
        st = between(ctx, a, b, mid)
        f(st) < val ? (lo = st) : (hi = st)
    end
    return result(val - f(lo) <= f(hi) - val ? lo : hi)
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The table must carry what the criterion reads, and be the one the EOS was
# fitted to: a cropped or different table would silently read the wrong row.
function check_beta_inputs(table::RawTable, v::EOSTableView)
    for f in ("mu_e", "muhat")
        has_field(table, f) || throw(ArgumentError("cold_beta_slice: table has no '$f' field"))
    end
    length(table.logρ) >= 2 && length(table.Yₑ) >= 2 ||
        throw(ArgumentError("cold_beta_slice: table needs at least two ρ and two Yₑ points"))
    lκ = log10(Float64(v.κ))
    same(a, b) = isapprox(Float64(a), Float64(b); atol=1.0e-8 * max(1.0, abs(Float64(b))))
    ok = same(table.logρ[1] + lκ, v.x_lo) && same(table.logρ[end] + lκ, v.x_hi) &&
         same(table.Yₑ[1], v.y_lo) && same(table.Yₑ[end], v.y_hi) && same(table.logT[1], v.u_lo)
    ok || throw(ArgumentError("cold_beta_slice: the table's axes do not match the EOS's box; pass the table the " *
                              "EOS was built from"))
    return nothing
end

function median_of(x::AbstractVector)
    s = sort(x)
    n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

# (mu_e − muhat) − m_B ∂ε/∂Yₑ on the coldest row above 10^13.5 g/cm³, where the
# tables are smooth enough for a Richardson-extrapolated difference quotient:
# its median, interquartile range and count, in MeV.
function electron_offset(table::RawTable, d::Matrix{Float64}, m_B_g::Float64)
    le = field(table, "logenergy")
    shift = Float64(energy_shift(table))
    Y = table.Yₑ
    vals = Float64[]
    for i in eachindex(table.logρ)
        table.logρ[i] >= 13.5 || continue
        ε(k) = exp10(Float64(le[i, 1, k])) - shift
        for k in 3:(length(Y) - 2)
            D1 = (ε(k + 1) - ε(k - 1)) / (Y[k + 1] - Y[k - 1])
            D2 = (ε(k + 2) - ε(k - 2)) / (Y[k + 2] - Y[k - 2])
            push!(vals, d[i, k] - (4D1 - D2) / 3 * m_B_g / MEV_TO_ERG)
        end
    end
    isempty(vals) && return NaN, NaN, 0
    s = sort(vals)
    q(f) = s[clamp(round(Int, f * (length(s) - 1)) + 1, 1, length(s))]
    return median_of(s), q(0.75) - q(0.25), length(s)
end

# The table's log ρ axis with `refine` points per cell.
function refined_axis(logρ::Vector{Float64}, refine::Integer)
    xs = Float64[]
    for i in 1:(length(logρ) - 1), r in 0:(refine - 1)
        push!(xs, logρ[i] + r * (logρ[i + 1] - logρ[i]) / refine)
    end
    push!(xs, logρ[end])
    return xs
end

# What placing and following candidates needs.
struct BranchCtx{V<:EOSTableView}
    v::V
    logρ::Vector{Float64}
    Yₑ::Vector{Float64}
    μν::Matrix{Float64}
    lκ::Float64
    cF::Float64          # energy_offset in U units per unit Yₑ
end

slice_ctx(sl::ColdBetaSlice) =
    BranchCtx(sl.eos, sl.logρ, sl.Yₑ, sl.μν, log10(Float64(sl.eos.κ)),
              Float64(sl.eos.conv_t) / Float64(sl.eos.κ) * sl.energy_offset)

# The candidates at raw log-density x: every upward zero crossing of μ_ν in Yₑ,
# with the root linear in its cell, and an edge at which μ_ν points outward.
# Each is (yₑ, clamped).
function mu_roots(ctx::BranchCtx, x::Float64)
    logρ, Yₑ, μν = ctx.logρ, ctx.Yₑ, ctx.μν
    x = clamp(x, logρ[1], logρ[end])
    i = clamp(searchsortedlast(logρ, x), 1, length(logρ) - 1)
    w = (x - logρ[i]) / (logρ[i + 1] - logρ[i])
    gk(k) = (1 - w) * μν[i, k] + w * μν[i + 1, k]
    n = length(Yₑ)
    out = Tuple{Float64,Bool}[]
    gk(1) > 0 && push!(out, (Yₑ[1], true))
    for k in 1:(n - 1)
        g0, g1 = gk(k), gk(k + 1)
        g0 <= 0 < g1 && push!(out, (Yₑ[k] + (Yₑ[k + 1] - Yₑ[k]) * (-g0 / (g1 - g0)), false))
    end
    gk(n) <= 0 && push!(out, (Yₑ[n], gk(n) < 0))
    return out
end

struct BetaCandidate
    ρ★::Float64
    y::Float64
    s::Float64
    pt::EOSPoint{Float64}
    g::Float64          # 1 + U − T̂s + p/ρ★ + c_E·Yₑ
    clamped::Bool
end

function beta_candidate(ctx::BranchCtx, x★::Float64, y::Float64, clamped::Bool)
    v = ctx.v
    ρ★ = exp10(x★)
    s = Float64(srange(v, ρ★, y).s_min)
    pt = evaluate(v, ρ★, s, y, Float64(v.u_lo))
    return BetaCandidate(ρ★, y, s, pt, 1 + pt.U - pt.T̂ * s + pt.p / ρ★ + ctx.cF * y, clamped)
end

# The candidate at log ρ★ = x★ on the branch whose Yₑ there is about y_guess.
function candidate_near(ctx::BranchCtx, x★::Float64, y_guess::Float64)
    y, clamped = argmin(r -> abs(r[1] - y_guess), mu_roots(ctx, x★ - ctx.lκ))
    return beta_candidate(ctx, x★, y, clamped)
end

# Between two states on one branch, the state at log ρ★ = x★.
function between(ctx::BranchCtx, a::BetaCandidate, b::BetaCandidate, x★::Float64)
    xa, xb = log10(a.ρ★), log10(b.ρ★)
    w = (x★ - xa) / (xb - xa)
    return candidate_near(ctx, x★, (1 - w) * a.y + w * b.y)
end

branch_candidate(sl::ColdBetaSlice, ctx::BranchCtx, i::Int, x★::Float64) =
    between(ctx, slice_candidate(sl, ctx, i), slice_candidate(sl, ctx, i + 1), x★)

function slice_candidate(sl::ColdBetaSlice, ctx::BranchCtx, i::Int)
    pt = evaluate(sl.eos, sl.ρ★[i], sl.s[i], sl.yₑ[i], Float64(sl.eos.u_lo))
    return BetaCandidate(sl.ρ★[i], sl.yₑ[i], sl.s[i], pt, sl.g[i], sl.clamped[i])
end

result(c::BetaCandidate) =
    BetaEquilibrium{Float64}(c.ρ★, c.y, c.s, c.pt, c.g, c.pt.flags | (c.clamped ? FLAG_CLAMP_YE : UInt32(0)))

function vertex_state(sl::ColdBetaSlice, i::Int)
    c = slice_candidate(sl, slice_ctx(sl), i)
    return result(c)
end

# A branch's stretch on which the pressure rises: its states in order of
# increasing density, their pressures, and whether it reaches the lowest or the
# highest sampled density (where the table's box, not the branch, ends it).
struct Piece
    c::Vector{BetaCandidate}
    p::Vector{Float64}
    lo_edge::Bool
    hi_edge::Bool
end

# Branches link candidates at neighbouring samples that are each other's
# nearest in Yₑ. Each branch is cut where its pressure stops rising, since no
# equilibrium state has dp/dρ ≤ 0; the pieces of two or more states are kept.
function stable_pieces(c::Vector{BetaCandidate}, first_of::Vector{Int})
    nxt = zeros(Int, length(c))
    prv = zeros(Int, length(c))
    for j in 1:(length(first_of) - 2)
        A = first_of[j]:(first_of[j + 1] - 1)
        B = first_of[j + 1]:(first_of[j + 2] - 1)
        (isempty(A) || isempty(B)) && continue
        for a in A
            b = argmin(k -> abs(c[k].y - c[a].y), B)
            argmin(k -> abs(c[k].y - c[b].y), A) == a && (nxt[a] = b; prv[b] = a)
        end
    end
    pieces = Piece[]
    nsamples = length(first_of) - 1
    sample(k) = searchsortedlast(first_of, k)
    keep!(cur) = length(cur) >= 2 &&
                 push!(pieces, Piece(c[cur], [c[k].pt.p for k in cur], sample(cur[1]) == 1, sample(cur[end]) == nsamples))
    for a in eachindex(c)
        prv[a] == 0 || continue
        cur = [a]
        k = a
        while nxt[k] != 0
            k2 = nxt[k]
            if c[k2].pt.p > c[k].pt.p
                push!(cur, k2)
            else
                keep!(cur)
                cur = [k2]
            end
            k = k2
        end
        keep!(cur)
    end
    return pieces
end

bracket(pc::Piece, p::Float64) = clamp(searchsortedlast(pc.p, p), 1, length(pc.p) - 1)

# g on a piece at pressure p, interpolated linearly between its states.
function piece_g(pc::Piece, p::Float64)
    k = bracket(pc, p)
    w = (p - pc.p[k]) / (pc.p[k + 1] - pc.p[k])
    return (1 - w) * pc.c[k].g + w * pc.c[k + 1].g
end

# The state on a piece at pressure p, by bisection in log ρ★ between the two
# states that bracket p. Of the last bracket, the end nearer in p is returned.
function piece_state(ctx::BranchCtx, pc::Piece, p::Float64)
    k = bracket(pc, p)
    a, b = pc.c[k], pc.c[k + 1]
    p <= a.pt.p && return a
    p >= b.pt.p && return b
    lo, hi = a, b
    for _ in 1:100
        mid = (log10(lo.ρ★) + log10(hi.ρ★)) / 2
        (mid == log10(lo.ρ★) || mid == log10(hi.ρ★)) && break
        st = between(ctx, a, b, mid)
        st.pt.p < p ? (lo = st) : (hi = st)
    end
    return p - lo.pt.p <= hi.pt.p - p ? lo : hi
end

# The least-g piece at each pressure, as runs joined at phase transitions.
#
# The range is where every piece that the box cuts off is defined: below the
# first pressure of a piece starting at the box's lowest density its g is
# unknown, and likewise at the top. Between neighbouring pressure breakpoints g
# is nearly linear on every piece, so the least one changes at most once there;
# each change is then located on the pieces' own states, where their g are
# equal. When two g cross at a small angle the exact crossing can lie an
# interval or two from the interpolated one, so the bracket is widened as
# needed.
function gibbs_envelope(ctx::BranchCtx, pieces::Vector{Piece})
    p_min = maximum((pc.p[1] for pc in pieces if pc.lo_edge); init=minimum(pc.p[1] for pc in pieces))
    p_max = minimum((pc.p[end] for pc in pieces if pc.hi_edge); init=maximum(pc.p[end] for pc in pieces))
    p_min < p_max || throw(ArgumentError("cold_beta_slice: the branches cut off by the box share no pressure"))
    P = sort!(unique!(filter(p -> p_min <= p <= p_max, vcat(p_min, p_max, (pc.p for pc in pieces)...))))
    pfirst = [pc.p[1] for pc in pieces]
    plast = [pc.p[end] for pc in pieces]
    active(pl, pr) = [q for q in eachindex(pieces) if pfirst[q] <= pl && plast[q] >= pr]
    ginterp(q, p) = piece_g(pieces[q], p)

    runs = Tuple{Int,Float64,Float64}[]   # (piece, start pressure, end pressure)
    cur, start = 0, P[1]
    function switch!(new, ps)
        push!(runs, (cur, start, ps))
        cur, start = new, ps
        return nothing
    end
    # Where `cur` loses to `new`, near interval m.
    function crossing(new, m)
        A, B = pieces[cur], pieces[new]
        Δ(p) = piece_state(ctx, A, p).g - piece_state(ctx, B, p).g
        lim_lo, lim_hi = max(A.p[1], B.p[1]), min(A.p[end], B.p[end])
        for k in 0:8
            lo = max(P[clamp(m - k, 1, length(P))], lim_lo)
            hi = min(P[clamp(m + 1 + k, 1, length(P))], lim_hi)
            lo < hi && Δ(lo) <= 0 <= Δ(hi) || continue
            for _ in 1:100
                mid = (lo + hi) / 2
                (mid == lo || mid == hi) && break
                Δ(mid) <= 0 ? (lo = mid) : (hi = mid)
            end
            return (lo + hi) / 2
        end
        # No crossing: `new` begins already below `cur` (a branch born at that
        # pressure), or `cur` ends still above `new`. Switch there, at one
        # pressure on exact states; g then jumps.
        P[clamp(m - 8, 1, length(P))] <= lim_lo <= P[clamp(m + 9, 1, length(P))] && Δ(lim_lo) > 0 && return lim_lo
        P[clamp(m - 8, 1, length(P))] <= lim_hi <= P[clamp(m + 9, 1, length(P))] && Δ(lim_hi) < 0 && return lim_hi
        # Otherwise the interpolated crossing.
        pl, pr = P[clamp(m, 1, length(P))], P[clamp(m + 1, 1, length(P))]
        dl, dr = ginterp(cur, pl) - ginterp(new, pl), ginterp(cur, pr) - ginterp(new, pr)
        return dl < dr ? clamp(pl + (pr - pl) * (-dl / (dr - dl)), pl, pr) : pl
    end
    for m in 1:(length(P) - 1)
        pl, pr = P[m], P[m + 1]
        act = active(pl, pr)
        isempty(act) && throw(ArgumentError("cold_beta_slice: no equilibrium branch covers p ∈ [$pl, $pr]"))
        bl = argmin(q -> ginterp(q, pl), act)
        br = argmin(q -> ginterp(q, pr), act)
        if cur == 0
            cur = bl
        elseif bl != cur
            # The current piece lost within the previous interval, or ended at pl.
            switch!(bl, cur in act ? crossing(bl, m - 1) : pl)
        end
        br != cur && switch!(br, crossing(br, m))
    end
    push!(runs, (cur, start, P[end]))

    verts = BetaCandidate[]
    jump_after = Bool[]
    jumps = ColdBetaJump[]
    function add!(st, jump)
        isempty(verts) || push!(jump_after, jump)
        push!(verts, st)
        return nothing
    end
    for (r, (q, ps, pe)) in enumerate(runs)
        pc = pieces[q]
        r == 1 && add!(piece_state(ctx, pc, ps), false)   # later runs start with the jump's high side
        for st in pc.c
            ps < st.pt.p < pe && st.ρ★ > verts[end].ρ★ && add!(st, false)
        end
        last = piece_state(ctx, pc, pe)
        last.ρ★ > verts[end].ρ★ && add!(last, false)
        if r < length(runs)
            lo = verts[end]
            hi = piece_state(ctx, pieces[runs[r + 1][1]], pe)
            add!(hi, true)
            push!(jumps, ColdBetaJump(pe, lo.g, hi.g - lo.g, lo.ρ★, hi.ρ★, lo.y, hi.y))
        end
    end
    # Rounding-level mismatches at a crossing count as none.
    for (i, j) in enumerate(jumps)
        abs(j.g_mismatch) <= 1.0e-12 * j.g &&
            (jumps[i] = ColdBetaJump(j.p, j.g, 0.0, j.ρ★_lo, j.ρ★_hi, j.yₑ_lo, j.yₑ_hi))
    end
    return verts, jump_after, jumps
end

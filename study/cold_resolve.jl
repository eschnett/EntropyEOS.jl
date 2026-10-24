# Behaviour study for the projection's face slack (CODE.md, "The contract"):
# how far past its faces does a cold re-solve of a projected state land, and
# how often does that make con2prim_safe flag it again?
#
# A state written onto a face comes back from con2prim only to the solver's
# conditioning, not to its tolerance: with a guess at the output the solve
# stops at the guess, but a cold start converges in the residuals and leaves s,
# ρ and w off by more. The slack has to cover that for an output to be a fixed
# point without a guess.
#
# States: valid ones with the 5% margins of test/test_real_tables.jl (w up to
# 6, magnetized in half), pushed off the valid range by up to four decades in
# D, τ and S. Each projected output is re-solved cold; its excess is how far
# the re-solved state lies past the nearest face it violates, relative to the
# window width for s and to the value for ρ and w (zero if inside). Re-flag
# counts are for con2prim_safe with the slack of the package as loaded, cold
# and with the output's primitives as the guess.
#
#   julia --project=study study/cold_resolve.jl --tables=/path/to/tables [--n=5000] [--only=LS220] [--float32]

include("common.jl")

const FLAGS = parse_flags(ARGS)
const N = parse(Int, get(FLAGS, "n", "5000"))

function sample_pushed(v, rng)
    xspan = v.x_hi - v.x_lo
    ρ = exp10(v.x_lo + xspan * (0.05 + 0.9rand(rng)))
    yₑ = v.y_lo + (v.y_hi - v.y_lo) * rand(rng)
    sr = srange(v, ρ, yₑ)
    s = sr.s_min + (sr.s_max - sr.s_min) * (0.05 + 0.9rand(rng))
    pt = E.evaluate(v, ρ, s, yₑ, NaN)
    B² = rand(rng) < 0.5 ? 0.0 : ρ * pt.h * exp10(-3 + 4rand(rng))
    c = prim2con(v, ρ, s, yₑ, 6rand(rng), B², 2rand(rng) - 1, pt.u_solved)
    k = rand(rng, 1:4)
    r() = exp10(-4 + 8rand(rng))
    fD = k in (1, 4) ? r() : 1.0
    fτ = k in (2, 4) ? r() : 1.0
    fS = k in (3, 4) ? r() : 1.0
    return Con2PrimIn(fD * c.D, fτ * c.τ, fD * c.D_Y, fS * c.S_par, fS * c.S_perp, c.B²)
end

"""How far `b` lies past the nearest violated face, relatively; zero if inside."""
function excess(v, proj, b)
    ρ_min, ρ_max, w_cap = E.proj_ρ_min(v, proj), E.proj_ρ_max(v, proj), E.proj_w_cap(proj)
    sr = srange(v, b.ρ, b.ye)
    W = sr.s_max - sr.s_min
    return max((sr.s_min - b.s) / W, (b.s - sr.s_max) / W, (ρ_min - b.ρ) / ρ_min, (b.ρ - ρ_max) / ρ_max,
               (b.w - w_cap) / w_cap, 0.0)
end

function study(name, v64, T)
    v = T === Float64 ? v64 : E.narrow(v64, T)
    opts = Con2PrimOptions{T}()
    proj = default_projection(v, exp10(T(v.x_lo)))
    rng = StableRNG(11)
    ex = Float64[]
    ncold = nguess = 0
    for _ in 1:N
        c = sample_pushed(v64, rng)
        cin = Con2PrimIn{T}(T(c.D), T(c.τ), T(c.D_Y), T(c.S_par), T(c.S_perp), T(c.B²))
        out = con2prim_safe(v, cin, opts, proj)
        out.policy_flags == 0 && continue
        k = out.cons
        c2 = Con2PrimIn{T}(k.D, k.τ, k.D_Y, k.S_par, k.S_perp, k.B²)
        b = con2prim(v, c2, opts)
        converged(b.result) && push!(ex, excess(v, proj, b))
        con2prim_safe(v, c2, opts, proj).policy_flags != 0 && (ncold += 1)
        con2prim_safe(v, c2, opts, proj, out.base.s, out.base.w, out.base.eos.u_solved).policy_flags != 0 &&
            (nguess += 1)
    end
    n = length(ex)
    @printf("| %s | %s | %d | %.1e | %.1e | %.1e | %d | %d | %d | %d |\n", name, T, n, pct(ex, 0.99), pct(ex, 0.999),
            maxv(ex), count(>(4 * E.con2prim_tol(T)), ex), count(>(E.face_slack(T)), ex), ncold, nguess)
end

function main()
    dir = get(FLAGS, "tables", get(ENV, "ENTROPYEOS_TABLE_DIR", ""))
    tabs = load_tables(dir; only=get(FLAGS, "only", nothing))
    types = haskey(FLAGS, "float32") ? (Float64, Float32) : (Float64,)
    @printf("n = %d pushed-off states per row; slack %s\n\n", N, join(["$T: $(E.face_slack(T))" for T in types], ", "))
    println("| table | type | projected, re-solve converged | excess p99 | p99.9 | max | > 4·tol | > slack | re-flagged cold | re-flagged with guess |")
    println("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (name, v) in tabs, T in types
        study(name, v, T)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

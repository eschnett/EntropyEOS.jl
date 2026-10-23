# Behaviour study: where con2prim fails on valid states, what con2prim_safe
# (the projection, level 1) makes of them, and whether a bisection in s rescues them
# (CODE.md, "Two-level projection (proposed)", on NOT_CONVERGED).
#
# The states are those of `test/test_real_tables.jl`: valid primitives inside a
# 5% margin of the box and the entropy window, w uniform in [0, 6], magnetized
# in 90% of cases. Each is solved warm (the truth perturbed by 1e-3) and cold
# (no guess). A cold failure is then
#
#   - solved at its true s by a cold inner solve, to see whether that lands on
#     the true root;
#   - bisected in s on G(s) = f₂ at the inner root, bracketed by the physical
#     window at the failed solve's density (as the diagnosis takes it) and at
#     the true density;
#   - passed to con2prim_safe with `default_projection`, to see its flags.
#
#   julia --project=study study/cold_failures.jl --tables=/path/to/tables [--n=20000] [--only=DD2]

include("common.jl")

const FLAGS = parse_flags(ARGS)
const N = parse(Int, get(FLAGS, "n", "20000"))

function sample_valid(v, rng; w_max_sample=6.0, sigma_max=1e4, margin=0.05)
    xspan = v.x_hi - v.x_lo
    xlo, xhi = v.x_lo + margin * xspan, v.x_hi - margin * xspan
    ρ = exp10(xlo + rand(rng) * (xhi - xlo))
    yₑ = v.y_lo + rand(rng) * (v.y_hi - v.y_lo)
    sr = srange(v, ρ, yₑ)
    sspan = sr.s_max - sr.s_min
    s = sr.s_min + margin * sspan + rand(rng) * (sspan - 2margin * sspan)
    w = rand(rng) * w_max_sample
    b2_zero = rand(rng) < 0.1
    σ = exp(log(1e-6) + rand(rng) * (log(sigma_max) - log(1e-6)))
    pt = E.evaluate(v, ρ, s, yₑ, NaN)
    return (ρ=ρ, s=s, yₑ=yₑ, w=w, B²=(b2_zero ? 0.0 : σ * ρ * pt.h), cos_vB=-1 + 2rand(rng), u=pt.u_solved)
end

inner(v, c, ye, s, opts) = E.inner_solve_w(v, c.D, c.τ, ye, c.S_par, c.S_perp, c.B², s, opts.w_max, opts.τ_floor_rel,
                                           0.0, NaN, opts.max_iter_1d, opts.tol)[1]

"""Bisection in s on f₂ at the inner root; `why` is :conv, :nobracket, :collapsed or :maxiter."""
function bisect(v, c, ye, a, b, opts; n=200)
    ra, rb = inner(v, c, ye, a, opts), inner(v, c, ye, b, opts)
    (ra.f2 <= 0 && rb.f2 >= 0) || (ra.f2 >= 0 && rb.f2 <= 0) || return (; ok=false, why=:nobracket, r=ra)
    r = ra
    for _ in 1:n
        m = (a + b) / 2
        r = inner(v, c, ye, m, opts)
        abs(r.f2) <= opts.tol && E.f1_converged(r, opts.tol) && return (; ok=true, why=:conv, r)
        sign(r.f2) == sign(ra.f2) ? (a, ra = m, r) : (b, rb = m, r)
        a < m < b || return (; ok=E.f1_converged(r, opts.tol), why=:collapsed, r)
    end
    return (; ok=false, why=:maxiter, r)
end

ok(o) = o.result === E.C2PResult.converged_newton || o.result === E.C2PResult.converged_fallback

function flagname(f)
    names = ((E.FLAG_POL_ATMOSPHERE, "ATMOSPHERE"), (E.FLAG_POL_NONFINITE, "NONFINITE"),
             (E.FLAG_POL_NOT_CONVERGED, "NOT_CONVERGED"), (E.FLAG_POL_ρ_FLOORED, "ρ_FLOORED"),
             (E.FLAG_POL_ρ_CEILED, "ρ_CEILED"), (E.FLAG_POL_YE_FLOORED, "YE_FLOORED"), (E.FLAG_POL_YE_CEILED, "YE_CEILED"),
             (E.FLAG_POL_S_FLOORED, "S_FLOORED"), (E.FLAG_POL_S_CEILED, "S_CEILED"), (E.FLAG_POL_W_CEILED, "W_CEILED"))
    s = join([n for (b, n) in names if f & b != 0], "|")
    return isempty(s) ? "0" : s
end

function main()
    dir = get(FLAGS, "tables", get(ENV, "ENTROPYEOS_TABLE_DIR", ""))
    tabs = filter(t -> t[1] != "synthetic", load_tables(dir; only=get(FLAGS, "only", nothing)))
    opts = E.Con2PrimOptions{Float64}()
    for (name, v) in tabs
        rng = StableRNG(4242)
        proj = E.default_projection(v, exp10(v.x_lo))
        nwarm = ncold = nrescued = 0
        safe = Dict{String,Int}()
        lines = String[]
        for _ in 1:N
            st = sample_valid(v, rng)
            c = E.prim2con(v, st.ρ, st.s, st.yₑ, st.w, st.B², st.cos_vB, st.u)
            cin = E.Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²)
            ok(E.con2prim(v, cin, opts, st.s * (1 + 1e-3), st.w + 1e-3, st.u + 1e-3)) || (nwarm += 1)
            o = E.con2prim(v, cin, opts)
            ok(o) && continue
            ncold += 1
            ye = c.D_Y / c.D
            w_ref = isfinite(o.w) ? clamp(o.w, 0.0, opts.w_max) : 0.0
            sr_ref = srange(v, c.D / cosh(w_ref), ye)
            sr_true = srange(v, st.ρ, ye)
            b_ref = bisect(v, cin, ye, sr_ref.s_min, sr_ref.s_max, opts)
            b_true = bisect(v, cin, ye, sr_true.s_min, sr_true.s_max, opts)
            back(b) = b.ok && abs(b.r.ρ - st.ρ) <= 1e-6 * st.ρ
            (back(b_ref) || back(b_true)) && (nrescued += 1)
            rt = inner(v, cin, ye, st.s, opts)
            sf = E.con2prim_safe(v, cin, opts, proj)
            fl = flagname(sf.policy_flags)
            safe[fl] = get(safe, fl, 0) + 1
            push!(lines, @sprintf("  %-17s ρ=%.2e s=%.3g Yₑ=%.2f W=%.1f | inner at true s: Δw=%.1e | bisect ref/true: %s/%s | safe: %s, ρ off by %.1e",
                                  string(o.result), st.ρ, st.s, st.yₑ, cosh(st.w), abs(rt.w - st.w), b_ref.why,
                                  b_true.why, fl, abs(sf.base.ρ - st.ρ) / st.ρ))
        end
        @printf("%s: warm fails %d/%d, cold fails %d/%d, rescued by bisection %d; con2prim_safe: %s\n", name, nwarm, N,
                ncold, N, nrescued, join(["$k $n" for (k, n) in sort(collect(safe))], ", "))
        foreach(println, lines)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

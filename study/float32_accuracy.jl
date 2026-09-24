# Float32 accuracy study, host side.
#
# Three configurations are compared against Float64 ground truth:
#
#   f64    Float64 table, Float64 arithmetic   (the validated reference)
#   mixed  Float32 table, Float64 arithmetic   (halves the table's memory traffic)
#   f32    Float32 table, Float32 arithmetic   (what Metal can run)
#
# and, for evaluate only, a Float64 table with Float32 arithmetic, which
# separates what the table's rounding costs from what the arithmetic costs.
#
# Inputs are always rounded to Float32 first and then handed to every
# configuration as the same exactly-representable values, so that input
# rounding and computational error are not confused. For con2prim the
# irreducible part is measured directly: the Float64 solver applied to the
# Float32-rounded conservatives ("floor") shows how much a hydro code loses by
# *storing* its state in Float32, before any solver runs at Float32.
#
# State sampling mirrors the C++ audit harness (`con2prim_audit.cpp`) and
# `test/test_real_tables.jl`: 5% margins on the density box and on the pointwise
# entropy window, a 10% zero-field coin, log-uniform magnetization, and a warm
# start perturbed by 1e-3. Uniform in entropy puts most states above ~80 MeV on
# a real table, though, so con2prim is also run on a second sampling that is
# uniform in log T instead -- which is where cold neutron-star matter lives.
#
#   julia -t 64 --project=study study/float32_accuracy.jl --tables=/path/to/tables \
#         [--n=1000000] [--only=LS220]

include("common.jl")

const FLAGS = parse_flags(ARGS)
const N = parse(Int, get(FLAGS, "n", "200000"))
const EPS32 = Float64(eps(Float32))

# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

const EVAL_FIELDS = (:U, :p, :h, :cs², :T_MeV, :T̂, :μ̃, :U_ρ, :U_ρρ, :U_ρs)

"""Points inside the audit box (`ext=false`) or across the whole extended box."""
function sample_eval_points(v, n, rng; ext)
    ρ = zeros(Float32, n); s = zeros(Float32, n); ye = zeros(Float32, n)
    for i in 1:n
        x = if ext
            v.x_ext_lo + rand(rng) * (v.x_hi - v.x_ext_lo)
        else
            m = 0.05 * (v.x_hi - v.x_lo)
            v.x_lo + m + rand(rng) * (v.x_hi - v.x_lo - 2m)
        end
        ρ64 = exp10(x)
        ye64 = v.y_lo + rand(rng) * (v.y_hi - v.y_lo)
        sr = ext ? E.srange_extended(v, ρ64, ye64) : srange(v, ρ64, ye64)
        m = ext ? 0.0 : 0.05 * (sr.s_max - sr.s_min)
        s64 = sr.s_min + m + rand(rng) * (sr.s_max - sr.s_min - 2m)
        ρ[i], s[i], ye[i] = ρ64, s64, ye64
    end
    return ρ, s, ye
end

function eval_study(name, v64, v32; ext)
    n = N
    rng = StableRNG(ext ? 4242 : 2424)
    ρ, s, ye = sample_eval_points(v64, n, rng; ext)
    truth = Vector{EOSPoint{Float64}}(undef, n)
    mixed = Vector{EOSPoint{Float64}}(undef, n)
    f32 = Vector{EOSPoint{Float32}}(undef, n)
    a32 = Vector{EOSPoint{Float32}}(undef, n)
    Threads.@threads :static for i in 1:n
        r, ss, y = Float64(ρ[i]), Float64(s[i]), Float64(ye[i])
        truth[i] = evaluate(v64, r, ss, y, NaN)
        mixed[i] = evaluate(v32, r, ss, y, NaN)
        f32[i] = evaluate(v32, ρ[i], s[i], ye[i], NaN32)
        a32[i] = evaluate(v64, ρ[i], s[i], ye[i], NaN32)
    end

    println("\n-- evaluate, $(ext ? "whole extended box" : "audit box (5% margins)"), $n points --")
    println("   relative error vs Float64, p50 / p99 / p999 / max")
    println("          [mixed: f32 table, f64 arith | f64 table, f32 arith | f32: both]")
    # μ̃ crosses zero, so it is measured against the temperature scale T̂
    # (both are energies per baryon over c²) rather than against itself.
    err(a, t, f) = f === :μ̃ ? abs(Float64(a.μ̃) - t.μ̃) / max(abs(t.μ̃), t.T̂) :
                   rel_diff(Float64(getfield(a, f)), getfield(t, f))
    for f in EVAL_FIELDS
        em = [err(mixed[i], truth[i], f) for i in 1:n]
        ea = [err(a32[i], truth[i], f) for i in 1:n]
        e3 = [err(f32[i], truth[i], f) for i in 1:n]
        @printf("   %-6s %s | %s | %s\n", f, qline(em), qline(ea), qline(e3))
    end
    eu = [abs(Float64(f32[i].u_solved) - truth[i].u_solved) for i in 1:n]
    @printf("   u abs  (f32) %s\n", qline(eu))
    it64 = sum(p -> p.iters, truth) / n; it32 = sum(p -> p.iters, f32) / n
    mx64 = count(p -> p.flags & FLAG_MAXITER != 0, truth); mx32 = count(p -> p.flags & FLAG_MAXITER != 0, f32)
    nf = count(p -> !(isfinite(p.U) && isfinite(p.p) && isfinite(p.cs²)), f32)
    @printf("   T-solve: mean iters f64 %.2f f32 %.2f; FLAG_MAXITER f64 %d f32 %d; non-finite f32 %d\n",
            it64, it32, mx64, mx32, nf)

    # Where the error lives: p99 per log-density band, the axis along which U
    # changes by orders of magnitude.
    if !ext
        edges = range(v64.x_lo, v64.x_hi; length=7)
        println("   p99 f32 relative error by log10 ρ* band:   U        p        cs²      T")
        for b in 1:6
            idx = [i for i in 1:n if edges[b] <= log10(Float64(ρ[i])) < edges[b+1]]
            isempty(idx) && continue
            q(f) = pct([rel_diff(Float64(getfield(f32[i], f)), getfield(truth[i], f)) for i in idx], 0.99)
            medU = pct([truth[i].U for i in idx], 0.5)
            @printf("     [%5.2f, %5.2f)  (median U %.1e)  %.1e  %.1e  %.1e  %.1e\n", edges[b], edges[b+1], medU,
                    q(:U), q(:p), q(:cs²), q(:T_MeV))
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# prim2con and con2prim
# ---------------------------------------------------------------------------

"""
Mirror of the C++ audit's `sample_state`, rounded to Float32. With
`logT=true` the entropy is instead set by a temperature drawn uniformly in
log T over the table's range, with the same 5% margins.
"""
function sample_audit_states(v, n, rng; logT=false)
    out = Vector{NTuple{6,Float32}}(undef, n)
    warm = Vector{NTuple{3,Float32}}(undef, n)
    for k in 1:n
        m = 0.05 * (v.x_hi - v.x_lo)
        ρ = exp10(v.x_lo + m + rand(rng) * (v.x_hi - v.x_lo - 2m))
        yₑ = v.y_lo + rand(rng) * (v.y_hi - v.y_lo)
        s = if logT
            um = 0.05 * (v.u_hi - v.u_lo)
            E.sigma_extended(v, ρ, v.u_lo + um + rand(rng) * (v.u_hi - v.u_lo - 2um), yₑ)
        else
            sr = srange(v, ρ, yₑ)
            sm = 0.05 * (sr.s_max - sr.s_min)
            sr.s_min + sm + rand(rng) * (sr.s_max - sr.s_min - 2sm)
        end
        w = 6 * rand(rng)
        b2_zero = rand(rng) < 0.1
        σ = exp(log(1e-6) + rand(rng) * (log(1e4) - log(1e-6)))
        pt = evaluate(v, ρ, s, yₑ, NaN)
        B² = b2_zero ? 0.0 : σ * ρ * pt.h
        cvB = -1 + 2rand(rng)
        out[k] = Float32.((ρ, s, yₑ, w, B², cvB))
        wr = 1e-3
        warm[k] = Float32.((s * (1 + wr * (-1 + 2rand(rng))), w + wr * (-1 + 2rand(rng)),
                            pt.u_solved + wr * (-1 + 2rand(rng))))
    end
    return out, warm
end

"""
Backward error of a recovered state: how far the conservatives it implies are
from the ones it was solved from, each in the scale a hydro code stores it at
-- D and τ relative to themselves, the momentum relative to E = τ + D, since it
may vanish.
"""
function backward_err(v, o, cin::Con2PrimIn)
    vv = hypot(Float64(o.v_par), Float64(o.v_perp))
    cvB = vv > 0 ? Float64(o.v_par) / vv : 0.0
    # Always Float64 arithmetic, whatever table `v` is.
    c = prim2con(v, Float64(o.ρ), Float64(o.s), Float64(o.ye), Float64(o.w), Float64(cin.B²), cvB, NaN)
    E_ = Float64(cin.τ) + Float64(cin.D)
    ΔS = hypot(c.S_par - cin.S_par, c.S_perp - cin.S_perp)
    return max(abs(c.D - cin.D) / cin.D, abs(c.τ - cin.τ) / cin.τ, ΔS / E_)
end

function report_solve(label, o, prims, n; floor=nothing)
    conv = [converged(x.result) for x in o]
    nfail = count(!, conv)
    byres = Dict(r => count(x -> x.result === r, o) for r in instances(E.C2PResult.T))
    nnb = byres[E.C2PResult.failed_no_bracket]; nmi = byres[E.C2PResult.failed_max_iter]
    eρ = [rt_err(Float64(o[i].ρ), Float64(prims[i][1])) for i in 1:n if conv[i]]
    es = [rt_err(Float64(o[i].s), Float64(prims[i][2])) for i in 1:n if conv[i]]
    ew = [rt_err(Float64(o[i].w), Float64(prims[i][4]), 1.0) for i in 1:n if conv[i]]
    itn = sum(x -> x.iters_newton, o) / n
    nfb = byres[E.C2PResult.converged_fallback]
    @printf("   %-12s fail %6d (%.4f%%; no_bracket %d, max_iter %d)  fallback %6d  <newton iters> %.2f\n", label, nfail,
            100nfail / n, nnb, nmi, nfb, itn)
    @printf("   %-12s   ρ  %s\n", "", qline(eρ))
    @printf("   %-12s   s  %s\n", "", qline(es))
    @printf("   %-12s   w  %s\n", "", qline(ew))
    if floor !== nothing
        # Distance to what the Float64 solver extracts from the same Float32
        # conservatives: the solver's own contribution, separated from the
        # conditioning of the problem.
        fo = floor
        both = [i for i in 1:n if conv[i] && converged(fo[i].result)]
        dρ = [rt_err(Float64(o[i].ρ), Float64(fo[i].ρ)) for i in both]
        ds = [rt_err(Float64(o[i].s), Float64(fo[i].s)) for i in both]
        @printf("   %-12s   vs floor: ρ %s\n", "", qline(dρ))
        @printf("   %-12s             s %s\n", "", qline(ds))
    end
    return nfail
end

function c2p_study(name, v64, v32; logT)
    n = N
    rng = StableRNG(logT ? 456789 : 987654)
    prims, warm = sample_audit_states(v64, n, rng; logT)
    cin64 = Vector{Con2PrimIn{Float64}}(undef, n)
    cin32 = Vector{Con2PrimIn{Float32}}(undef, n)
    p2c32 = Vector{Prim2ConOut{Float32}}(undef, n)
    Threads.@threads :static for k in 1:n
        ρ, s, yₑ, w, B², cvB = Float64.(prims[k])
        c = prim2con(v64, ρ, s, yₑ, w, B², cvB, NaN)
        cin64[k] = Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²)
        cin32[k] = Con2PrimIn(Float32.((c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²))...)
        p2c32[k] = prim2con(v32, prims[k]..., NaN32)
    end

    println("\n######## sampling: $(logT ? "uniform in log T" : "uniform in s (the C++ audit)") ########")
    println("\n-- prim2con, f32 vs Float64 at the same Float32 primitives, $n states --")
    for (f, lbl) in ((:D, "D"), (:τ, "τ"), (:S_par, "S_par"), (:S_perp, "S_perp"))
        e = [rel_diff(Float64(getfield(p2c32[k], f)), getfield(cin64[k], f)) for k in 1:n]
        @printf("   %-6s %s\n", lbl, qline(e))
    end
    # τ relative to the energy scale E = τ + D, which is what a hydro code's
    # own Float32 storage resolves.
    eτE = [abs(Float64(p2c32[k].τ) - cin64[k].τ) / (cin64[k].τ + cin64[k].D) for k in 1:n]
    @printf("   τ/E    %s\n", qline(eτE))

    o64 = Con2PrimOptions{Float64}()
    o32 = Con2PrimOptions{Float32}()
    up(c::Con2PrimIn{Float32}) = Con2PrimIn(Float64.((c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²))...)
    w64(k) = Float64.(warm[k])

    solves = Dict{String,Any}()
    t = Dict{String,Float64}()
    # Concretely typed result vectors, so the timing measures the solver.
    function timed(f, lbl)
        out = Vector{typeof(f(1))}(undef, n)
        t[lbl] = @elapsed Threads.@threads :static for k in 1:n
            @inbounds out[k] = f(k)
        end
        solves[lbl] = out
    end
    timed(k -> con2prim(v64, cin64[k], o64, w64(k)...), "f64 warm")
    timed(k -> con2prim(v64, cin64[k], o64), "f64 cold")
    timed(k -> con2prim(v64, up(cin32[k]), o64, w64(k)...), "floor warm")
    timed(k -> con2prim(v32, up(cin32[k]), o64, w64(k)...), "mixed warm")
    timed(k -> con2prim(v32, up(cin32[k]), o64), "mixed cold")
    timed(k -> con2prim(v32, cin32[k], o32, warm[k]...), "f32 warm")
    timed(k -> con2prim(v32, cin32[k], o32), "f32 cold")

    println("\n-- con2prim round trip vs ground truth, $n states; error quantiles p50 / p99 / p999 / max --")
    fl = solves["floor warm"]
    for lbl in ("f64 warm", "f64 cold", "floor warm", "mixed warm", "mixed cold", "f32 warm", "f32 cold")
        report_solve(lbl, solves[lbl], prims, n; floor=lbl in ("f64 warm", "floor warm", "f64 cold") ? nothing : fl)
    end

    # Against the Float64 table this includes the Float32 table's own rounding;
    # against the table the solver actually used it is the solver's residual
    # alone, which its tolerance bounds.
    println("\n-- backward error max(|ΔD|/D, |Δτ|/τ, |ΔS|/E) of converged solves, re-evaluated in Float64 arithmetic --")
    for lbl in ("floor warm", "mixed warm", "f32 warm", "f32 cold")
        o = solves[lbl]
        be = [backward_err(v64, o[k], cin32[k]) for k in 1:n if converged(o[k].result)]
        bo = [backward_err(lbl == "floor warm" ? v64 : v32, o[k], cin32[k]) for k in 1:n if converged(o[k].result)]
        @printf("   %-12s vs f64 table %s  | vs own table %s  (own p99 = %.0f eps32)\n", lbl, qline(be), qline(bo),
                pct(bo, 0.99) / EPS32)
    end

    # Where the Float32 solver's error lives, by temperature: cold matter is
    # where s is least well determined by τ.
    for lbl in ("f32 warm", "f32 cold")
    let o = solves[lbl], fo = solves["floor warm"]
        uT = [log10(evaluate(v64, Float64.(prims[k][1:3])..., NaN).T_MeV) for k in 1:n]
        wT = [Float64(prims[k][4]) for k in 1:n]
        for (axis, vals) in (("log10 T [MeV]", uT), ("rapidity w", wT))
        edges = range(minimum(vals), maximum(vals) + 1e-9; length=7)
        println("\n   $lbl by $axis:   n   fail   p99 ρ err   p99 s err   p99 s err (floor)")
        for b in 1:6
            idx = [k for k in 1:n if edges[b] <= vals[k] < edges[b+1]]
            isempty(idx) && continue
            ok = [k for k in idx if converged(o[k].result)]
            okf = [k for k in idx if converged(fo[k].result)]
            @printf("     [%5.2f, %5.2f) %7d %6d   %.2e    %.2e    %.2e\n", edges[b], edges[b+1], length(idx),
                    length(idx) - length(ok), pct([rt_err(Float64(o[k].ρ), Float64(prims[k][1])) for k in ok], 0.99),
                    pct([rt_err(Float64(o[k].s), Float64(prims[k][2])) for k in ok], 0.99),
                    pct([rt_err(Float64(fo[k].s), Float64(prims[k][2])) for k in okf], 0.99))
        end
        end
    end
    end

    # Is the default Float32 tolerance the right one? A tolerance below the
    # noise floor of the Float32 residual cannot be met: Newton stalls and the
    # fallback takes over, or nothing converges at all.
    println("\n   f32 tolerance sweep:  tol/eps32   fail   max_iter  fallback  <iters>  p99 ρ err   p999 ρ err  p99 backward  Mstates/s/thread")
    for warmstart in (true, false), k_eps in (16, 64, 128, 256, 512, 1024)
        oo = Con2PrimOptions{Float32}(; tol=k_eps * eps(Float32))
        o = Vector{Con2PrimOut{Float32}}(undef, n)
        tt = @elapsed Threads.@threads :static for k in 1:n
            o[k] = warmstart ? con2prim(v32, cin32[k], oo, warm[k]...) : con2prim(v32, cin32[k], oo)
        end
        ok = [k for k in 1:n if converged(o[k].result)]
        eρ = [rt_err(Float64(o[k].ρ), Float64(prims[k][1])) for k in ok]
        @printf("     %s   %5d    %6d   %6d   %6d    %5.2f    %.2e    %.2e    %.2e      %.3f\n", warmstart ? "warm" : "cold",
                k_eps, n - length(ok), count(x -> x.result === E.C2PResult.failed_max_iter, o),
                count(x -> x.result === E.C2PResult.converged_fallback, o), sum(x -> x.iters_newton, o) / n,
                pct(eρ, 0.99), pct(eρ, 0.999), pct([backward_err(v64, o[k], cin32[k]) for k in ok], 0.99),
                n / tt / Threads.nthreads() / 1e6)
    end

    # The never-fails layer at Float32: its contract (finite output, and the
    # returned conservatives re-solve to the returned primitives) must hold.
    pol32 = default_policy(v32, exp10(Float32(v32.x_lo)))
    nbad = Threads.Atomic{Int}(0)
    resolve = Vector{Float64}(undef, n)
    nflag = Threads.Atomic{Int}(0)
    Threads.@threads :static for k in 1:n
        sa = con2prim_safe(v32, cin32[k], o32, pol32)
        fin = isfinite(sa.base.ρ) && isfinite(sa.base.s) && isfinite(sa.base.w) && isfinite(sa.cons.τ)
        fin || Threads.atomic_add!(nbad, 1)
        sa.policy_flags != 0 && Threads.atomic_add!(nflag, 1)
        c = sa.cons
        re = con2prim(v32, Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²), o32, sa.base.s, sa.base.w,
                      sa.base.eos.u_solved)
        resolve[k] = converged(re.result) ? rt_err(Float64(re.ρ), Float64(sa.base.ρ)) : Inf
    end
    @printf("\n   con2prim_safe f32: non-finite %d, policy-touched %d, re-solve ρ err %s, re-solve failures %d\n",
            nbad[], nflag[], qline(filter(isfinite, resolve)), count(!isfinite, resolve))

    nt = Threads.nthreads()
    print("\n   throughput per thread (Mstates/s): ")
    for lbl in ("f64 warm", "mixed warm", "f32 warm", "f64 cold", "mixed cold", "f32 cold")
        @printf("%s %.3f  ", lbl, n / t[lbl] / nt / 1e6)
    end
    println()
    return nothing
end

println("EntropyEOS Float32 accuracy study: Julia $VERSION, $(Threads.nthreads()) threads, n = $N")
@printf("eps(Float32) = %.3e, Con2PrimOptions{Float32}().tol = %.3e\n", EPS32, Con2PrimOptions{Float32}().tol)
for (name, v64) in load_tables(get(FLAGS, "tables", ""); only=get(FLAGS, "only", nothing))
    v32 = E.narrow(v64, Float32)
    println("\n==================== $name ====================")
    @printf("κ = %.10f (Float32: %.10f), x ∈ [%.3f, %.3f], shift_hat = %.4e\n", v64.κ, v32.κ, v64.x_lo, v64.x_hi,
            v64.shift_hat)
    eval_study(name, v64, v32; ext=false)
    eval_study(name, v64, v32; ext=true)
    c2p_study(name, v64, v32; logT=false)
    c2p_study(name, v64, v32; logT=true)
    flush(stdout)
end

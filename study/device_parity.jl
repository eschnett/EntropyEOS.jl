# Device parity: the same kernels on a GPU and on the host, at one scalar type.
#
# A faithful mirror of the C++ `tests/test_device_cuda.cu`, so its numbers can
# be read side by side with the C++ H200 runs: the same state sampling (2%
# entropy margin, w uniform in [0, 6), half the states magnetized with B²/p
# log-uniform in [1e-3, 10], warm start at the exact truth), the same four
# passes (evaluate, cold con2prim, warm con2prim, con2prim_safe), and the same
# three gates (one floor is widened; see gate 2). What is claimed is "the device is indistinguishable from the
# host solver", not absolute solver quality -- that is the audits' business.
#
# One difference: the C++ harness repairs a real table in memory first. This
# port has no repair, so real tables are used as shipped (or pre-repaired on
# disk, as DD2 repaired is).
#
#   julia --project=study study/device_parity.jl --backend=cuda --type=Float64 \
#         --tables=/path/to/tables [--storage=Float32] [--n=1048576] [--only=LS220]
#
# `--type` is the arithmetic type; `--storage` (default: the same) is the
# table's coefficient type, so `--type=Float64 --storage=Float32` runs Float64
# arithmetic against a Float32 table.

using Adapt
using KernelAbstractions
include("common.jl")

const FLAGS = parse_flags(ARGS)
const BACKEND_NAME = get(FLAGS, "backend", "cuda")
const TYPES = Dict("Float64" => Float64, "Float32" => Float32)
const T = TYPES[get(FLAGS, "type", "Float64")]
const S = TYPES[get(FLAGS, "storage", string(T))]
const N = parse(Int, get(FLAGS, "n", string(2^20)))

if BACKEND_NAME == "cuda"
    using CUDA
    const BACKEND = CUDABackend()
    const DEVARRAY = CuArray
    device_name() = CUDA.name(CUDA.device()) * " (sm_" * string(CUDA.capability(CUDA.device())) * ")"
elseif BACKEND_NAME == "metal"
    using Metal
    const BACKEND = MetalBackend()
    const DEVARRAY = MtlArray
    device_name() = string(Metal.device())
else
    error("unknown backend $BACKEND_NAME")
end

@kernel function eval_k!(out, @Const(ρ), @Const(s), @Const(ye), v)
    i = @index(Global)
    @inbounds out[i] = evaluate(v, ρ[i], s[i], ye[i], typeof(ρ[i])(NaN))
end

@kernel function cold_k!(out, @Const(cin), v, opts)
    i = @index(Global)
    @inbounds out[i] = con2prim(v, cin[i], opts)
end

@kernel function warm_k!(out, @Const(cin), @Const(s), @Const(w), @Const(u), v, opts)
    i = @index(Global)
    @inbounds out[i] = con2prim(v, cin[i], opts, s[i], w[i], u[i])
end

# The residual pair at a given iterate, for diagnosing warm-start outliers.
@kernel function resid_k!(out, @Const(cin), @Const(s), @Const(w), @Const(u), v, τ_floor_rel)
    i = @index(Global)
    @inbounds c = cin[i]
    r = EntropyEOS.residuals(v, c.D, c.τ, c.D_Y / c.D, c.S_par, c.S_perp, c.B², s[i], w[i], u[i], τ_floor_rel)
    @inbounds out[i] = (r.f1 / r.coshw, r.f2, r.pt.u_solved, r.pt.U, r.pt.p, typeof(r.f2)(r.pt.iters))
end

@kernel function safe_k!(out, @Const(cin), v, opts, pol)
    i = @index(Global)
    @inbounds out[i] = con2prim_safe(v, cin[i], opts, pol)
end

"""Mirror of the C++ device harness's sampling, drawn in Float64 and rounded to `T`."""
function sample_states(v64, vT, n)
    rng = StableRNG(12345)
    ρ = zeros(T, n); s = zeros(T, n); ye = zeros(T, n); w = zeros(T, n)
    B² = zeros(T, n); cvB = zeros(T, n); u = zeros(T, n)
    cin = Vector{Con2PrimIn{T}}(undef, n)
    for i in 1:n
        x = v64.x_lo + (v64.x_hi - v64.x_lo) * rand(rng)
        ρ64 = exp10(x)
        ye64 = v64.y_lo + (v64.y_hi - v64.y_lo) * rand(rng)
        sr = srange(v64, ρ64, ye64)
        s64 = sr.s_min + (sr.s_max - sr.s_min) * (0.02 + 0.96 * rand(rng))
        w64 = 6 * rand(rng)
        c64 = 2 * rand(rng) - 1
        p64 = evaluate(v64, ρ64, s64, ye64, NaN).p
        ξ = rand(rng)
        b64 = iseven(i - 1) ? 0.0 : p64 * exp10(-3 + 4ξ)     # C++ i%2 with 0-based i
        ρ[i], s[i], ye[i], w[i], B²[i], cvB[i] = T(ρ64), T(s64), T(ye64), T(w64), T(b64), T(c64)
        # Everything past the draw is computed at T, as the C++ does in `real`.
        u[i] = evaluate(vT, ρ[i], s[i], ye[i], T(NaN)).u_solved
        pc = prim2con(vT, ρ[i], s[i], ye[i], w[i], B²[i], cvB[i], u[i])
        cin[i] = Con2PrimIn(pc.D, pc.τ, pc.D_Y, pc.S_par, pc.S_perp, pc.B²)
    end
    return (; ρ, s, ye, w, B², cvB, u, cin)
end

"""Threaded host pass; returns the results and the wall time."""
function host_pass(f, ::Type{R}, n) where {R}
    out = Vector{R}(undef, n)
    f(1)                                    # compile outside the timing
    t = @elapsed Threads.@threads :static for i in 1:n
        @inbounds out[i] = f(i)
    end
    return out, t
end

"""One warm-up launch, then a timed one."""
function device_pass(launch!)
    launch!()
    KernelAbstractions.synchronize(BACKEND)
    t = @elapsed begin
        launch!()
        KernelAbstractions.synchronize(BACKEND)
    end
    return t
end

function rt_stats(what, out, st)
    eρ = Float64[]; es = Float64[]; ew = Float64[]
    for i in eachindex(out)
        converged(out[i].result) || continue
        push!(eρ, rt_err(out[i].ρ, st.ρ[i]))
        push!(es, rt_err(out[i].s, st.s[i]))
        push!(ew, rt_err(out[i].w, st.w[i], 1.0))
    end
    r = (p99_ρ=pct(eρ, 0.99), max_ρ=maxv(eρ), p99_s=pct(es, 0.99), max_s=maxv(es),
         p99_w=pct(ew, 0.99), max_w=maxv(ew))
    @printf("%s round trip: rho p99 %.2e max %.2e | s p99 %.2e max %.2e | w p99 %.2e max %.2e\n",
            what, r.p99_ρ, r.max_ρ, r.p99_s, r.max_s, r.p99_w, r.max_w)
    return r
end

function run_table(name, v64)
    vT = S === Float64 ? v64 : E.narrow(v64, S)
    n = N
    println("\n=== $name ($T arithmetic, $S table) ===")
    @printf("eos: box x [%.4f, %.4f] u [%.4g, %.4g] y [%.4g, %.4g]\n", v64.x_lo, v64.x_hi, v64.u_lo, v64.u_hi,
            v64.y_lo, v64.y_hi)
    println("device: ", device_name(), ", $n states, workgroup 256, host threads ", Threads.nthreads())
    @printf("mirrored %.1f MB of coefficients\n", (length(vT.σ.c) + length(vT.L.c)) * sizeof(S) / 2^20)

    t0 = time()
    st = sample_states(v64, vT, n)
    @printf("sampled in %.1f s\n", time() - t0)
    opts = Con2PrimOptions{T}()
    pol = default_policy(vT, exp10(T(vT.x_lo)))
    nanT = T(NaN)

    # --- host passes
    pt_cpu, t_eval_cpu = host_pass(i -> evaluate(vT, st.ρ[i], st.s[i], st.ye[i], nanT), EOSPoint{T}, n)
    cold_cpu, t_cold_cpu = host_pass(i -> con2prim(vT, st.cin[i], opts), Con2PrimOut{T}, n)
    warm_cpu, t_warm_cpu = host_pass(i -> con2prim(vT, st.cin[i], opts, st.s[i], st.w[i], st.u[i]),
                                     Con2PrimOut{T}, n)
    safe_cpu, t_safe_cpu = host_pass(i -> con2prim_safe(vT, st.cin[i], opts, pol), Con2PrimSafeOut{T}, n)

    # --- device passes
    dv = Adapt.adapt(DEVARRAY, vT)
    dρ, ds, dye = DEVARRAY(st.ρ), DEVARRAY(st.s), DEVARRAY(st.ye)
    dw, du, dcin = DEVARRAY(st.w), DEVARRAY(st.u), DEVARRAY(st.cin)
    dpt = DEVARRAY{EOSPoint{T}}(undef, n)
    dcold = DEVARRAY{Con2PrimOut{T}}(undef, n)
    dwarm = DEVARRAY{Con2PrimOut{T}}(undef, n)
    dsafe = DEVARRAY{Con2PrimSafeOut{T}}(undef, n)
    t_eval_gpu = device_pass(() -> eval_k!(BACKEND, 256)(dpt, dρ, ds, dye, dv; ndrange=n))
    t_cold_gpu = device_pass(() -> cold_k!(BACKEND, 256)(dcold, dcin, dv, opts; ndrange=n))
    t_warm_gpu = device_pass(() -> warm_k!(BACKEND, 256)(dwarm, dcin, ds, dw, du, dv, opts; ndrange=n))
    t_safe_gpu = device_pass(() -> safe_k!(BACKEND, 256)(dsafe, dcin, dv, opts, pol; ndrange=n))
    pt_gpu, cold_gpu, warm_gpu, safe_gpu = Array(dpt), Array(dcold), Array(dwarm), Array(dsafe)

    failures = 0

    # Gate 1: failure *rate* parity. Boundary states flip in both directions
    # under ULP-level libm differences, so a per-state gate is wrong by
    # construction; the GPU total must be within 5% + 10 of the CPU's.
    fail_bar(conv) = round(Int, 1.05 * (n - conv)) + 10
    cc = count(o -> converged(o.result), cold_cpu); cg = count(o -> converged(o.result), cold_gpu)
    wc = count(o -> converged(o.result), warm_cpu); wg = count(o -> converged(o.result), warm_gpu)
    par_c = count(i -> converged(cold_cpu[i].result) && !converged(cold_gpu[i].result), 1:n)
    par_w = count(i -> converged(warm_cpu[i].result) && !converged(warm_gpu[i].result), 1:n)
    shown = 0
    for i in 1:n
        if converged(cold_cpu[i].result) != converged(cold_gpu[i].result) && shown < 5
            shown += 1
            @printf("  parity(cold) state %d: cpu %s gpu %s  D %.6e tau %.6e B2 %.3e w %.3f\n", i - 1,
                    cold_cpu[i].result, cold_gpu[i].result, st.cin[i].D, st.cin[i].τ, st.cin[i].B², st.w[i])
        end
    end
    @printf("convergence: cold cpu %d/%d gpu %d/%d (cpu-conv-but-gpu-not: %d); warm cpu %d/%d gpu %d/%d (cpu-conv-but-gpu-not: %d)\n",
            cc, n, cg, n, par_c, wc, n, wg, n, par_w)
    if n - cg > fail_bar(cc) || n - wg > fail_bar(wc)
        println("  GATE FAIL: GPU failure total above the CPU's + 5% + 10")
        failures += 1
    end

    # Gate 2: GPU round-trip errors held to the CPU's own distribution.
    #
    # The C++ gate is 10x + 1e-12 on both p99 and max. That is kept for p99,
    # but the max needs a floor of 100 solver tolerances: the warm start is the
    # host's own exact answer, so the host's max error is 2e-16, while the
    # device's temperature solve may land up to its step tolerance (1e-13 in
    # log T) away. Where U ∝ T⁴ that moves the energy residual by ~1e-12,
    # just over the con2prim tolerance, and the device takes one legitimate
    # Newton step to a root ~1e-11 away. Measured on H200: 1-2 such states per
    # million, max 5e-11.
    bar(k, c) = startswith(string(k), "max") ? 10c + 100 * Float64(opts.tol) : 10c + 1e-12
    for (what, oc, og) in (("cold", cold_cpu, cold_gpu), ("warm", warm_cpu, warm_gpu))
        rc = rt_stats("cpu $what", oc, st)
        rg = rt_stats("gpu $what", og, st)
        if any(getfield(rg, k) > bar(k, getfield(rc, k)) for k in keys(rc))
            println("  GATE FAIL: $what GPU round-trip above 10x the CPU's own distribution")
            failures += 1
        end
    end

    # The warm start is the exact truth, so a warm outlier means one side took
    # a Newton step the other did not. Show how close the starting residual
    # already was to the tolerance: that is what ULP-level libm differences
    # can tip either way.
    wout = [i for i in 1:n if converged(warm_gpu[i].result) &&
                              rt_err(warm_gpu[i].ρ, st.ρ[i]) > 1e-12 + 10 * rt_err(warm_cpu[i].ρ, st.ρ[i])]
    if !isempty(wout)
        @printf("  warm outliers: %d states with GPU ρ error > 1e-12 + 10x CPU's\n", length(wout))
        dres = DEVARRAY{NTuple{6,T}}(undef, n)
        resid_k!(BACKEND, 256)(dres, dcin, ds, dw, du, dv, opts.τ_floor_rel; ndrange=n)
        KernelAbstractions.synchronize(BACKEND)
        res_gpu = Array(dres)
        for i in wout[1:min(end, 5)]
            c = st.cin[i]
            r0 = E.residuals(vT, c.D, c.τ, c.D_Y / c.D, c.S_par, c.S_perp, c.B², st.s[i], st.w[i], st.u[i], opts.τ_floor_rel)
            g = res_gpu[i]
            @printf("    state %d: newton iters cpu %d gpu %d, ρ err cpu %.1e gpu %.1e; w %.3f B² %.2e\n", i - 1,
                    warm_cpu[i].iters_newton, warm_gpu[i].iters_newton, rt_err(warm_cpu[i].ρ, st.ρ[i]),
                    rt_err(warm_gpu[i].ρ, st.ρ[i]), st.w[i], c.B²)
            @printf("      start residual/tol  cpu f1 %+.3e f2 %+.3e | gpu f1 %+.3e f2 %+.3e\n", r0.f1 / r0.coshw / opts.tol,
                    r0.f2 / opts.tol, g[1] / opts.tol, g[2] / opts.tol)
            @printf("      EOS at start        cpu u %.17g U %.17g p %.17g iters %d\n", r0.pt.u_solved, r0.pt.U, r0.pt.p, r0.pt.iters)
            @printf("                          gpu u %.17g U %.17g p %.17g iters %d\n", g[3], g[4], g[5], Int(g[6]))
        end
    end

    # Report, never gated: |GPU - CPU| per quantity, and agreement rates.
    dp = [rel_diff(pt_cpu[i].p, pt_gpu[i].p) for i in 1:n]
    dcs = [rel_diff(pt_cpu[i].cs², pt_gpu[i].cs²) for i in 1:n]
    du_ = [rel_diff(pt_cpu[i].u_solved, pt_gpu[i].u_solved) for i in 1:n]
    both = [i for i in 1:n if converged(cold_cpu[i].result) && converged(cold_gpu[i].result)]
    dρ_ = [rel_diff(cold_cpu[i].ρ, cold_gpu[i].ρ) for i in both]
    dw_ = [rel_diff(cold_cpu[i].w, cold_gpu[i].w) for i in both]
    @printf("gpu-cpu deltas: eval p %.2e/%.2e cs2 %.2e/%.2e u %.2e/%.2e | cold rho %.2e/%.2e w %.2e/%.2e (p99/max)\n",
            pct(dp, 0.99), maxv(dp), pct(dcs, 0.99), maxv(dcs), pct(du_, 0.99), maxv(du_),
            pct(dρ_, 0.99), maxv(dρ_), pct(dw_, 0.99), maxv(dw_))
    agree(f) = 100 * count(f, 1:n) / n
    @printf("agreement: cold result %.4f%% flags %.4f%% | safe result %.4f%% policy_flags %.4f%%\n",
            agree(i -> cold_cpu[i].result === cold_gpu[i].result), agree(i -> cold_cpu[i].flags == cold_gpu[i].flags),
            agree(i -> safe_cpu[i].base.result === safe_gpu[i].base.result),
            agree(i -> safe_cpu[i].policy_flags == safe_gpu[i].policy_flags))

    # Gate 3: con2prim_safe never fails.
    bad = count(o -> !(isfinite(o.base.ρ) && isfinite(o.base.s) && isfinite(o.base.w) && isfinite(o.cons.τ)), safe_gpu)
    @printf("con2prim_safe: %d/%d non-finite outputs on GPU\n", bad, n)
    bad > 0 && (failures += 1)

    nt = Threads.nthreads()
    rate(t) = n / t / 1e6
    @printf("throughput (Mstates/s, %d states):\n", n)
    @printf("  %-14s %10s %10s %10s %8s\n", "", "cpu(1core)", "cpu(all)", "gpu", "gpu/core")
    for (lbl, tc, tg) in (("evaluate", t_eval_cpu, t_eval_gpu), ("con2prim cold", t_cold_cpu, t_cold_gpu),
                          ("con2prim warm", t_warm_cpu, t_warm_gpu), ("con2prim_safe", t_safe_cpu, t_safe_gpu))
        @printf("  %-14s %10.3f %10.2f %10.1f %8.0f\n", lbl, rate(tc) / nt, rate(tc), rate(tg), nt * tc / tg)
    end
    println(failures == 0 ? "PASS" : "FAIL ($failures gates)")
    return failures
end

println("EntropyEOS device parity: backend $BACKEND_NAME, $T arithmetic, $S table, Julia $VERSION")
tables = load_tables(get(FLAGS, "tables", ""); only=get(FLAGS, "only", nothing))
nfail = sum(run_table(name, v) for (name, v) in tables)
println("\n", nfail == 0 ? "ALL PASS" : "FAILURES: $nfail")
exit(nfail == 0 ? 0 : 1)

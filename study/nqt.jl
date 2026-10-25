# Not-quite-transcendental (NQT) functions: would they pay here?
# (HISTORY.md, "Not-quite-transcendental functions".)
#
# An NQT log₂ (Miller, Dolence & Holladay, arXiv:2206.08957; Hammond et al.,
# arXiv:2501.05410) is the binary exponent plus a polynomial in the mantissa:
# with n = m·2^p, m ∈ [1/2, 1) and t = m - 1, lg(n) = p + f(t). The identity
# lg(2n) = lg(n) + 1 makes it C^k if and only if f(0) = 0, f(-1/2) = -1 and
# f⁽ʲ⁾(-1/2) = 2ʲ f⁽ʲ⁾(0) for j = 1..k, which fix a polynomial of degree k + 1;
# k = 0 is the papers' NQTo1 and k = 1 their NQTo2. Degrees beyond k + 1 are
# spent on a least-squares fit to log₂.
#
# Four measurements:
#
#   1. accuracy: the package's not-a-knot spline fitted to log₂ U on nodes
#      uniform in log₂ ρ or in an NQT variant of it, for a smooth two-power U(ρ).
#      U, ρ²U_ρ and h·cs² come from the spline's derivatives by the chain rule,
#      as in `eval_at`;
#   2. how those errors converge as the spacing halves;
#   3. the hyperbolic identities an NQT exponential breaks;
#   4. with --tables, the share of `evaluate` and of a warm-started `con2prim`
#      taken by their transcendental calls, on one thread.
#
#   julia --project=study study/nqt.jl [--tables=/path/to/tables] [--n=100000] [--only=LS220]

include("common.jl")
using LinearAlgebra

const FLAGS = parse_flags(ARGS)
const N = parse(Int, get(FLAGS, "n", "100000"))

# ---------------------------------------------------------------------------
# The axes
# ---------------------------------------------------------------------------

"""
    nqt_coeffs(k, d)

Coefficients `c` of `f(t) = Σ c[i+1] tⁱ`, of degree `d ≥ k + 1`, that make the
NQT log₂ C^k. Freedom beyond the `k + 2` conditions goes to a least-squares fit
to `log2(1 + t)`.
"""
function nqt_coeffs(k, d)
    dpow(i, j, t) = i < j ? 0.0 : prod((i - j + 1):i; init=1) * t^(i - j)    # dʲ/dtʲ tⁱ
    C = zeros(k + 2, d + 1)
    r = zeros(k + 2)
    for i in 0:d
        C[1, i + 1] = dpow(i, 0, 0.0)
        C[2, i + 1] = dpow(i, 0, -0.5)
        for j in 1:k
            C[2 + j, i + 1] = dpow(i, j, -0.5) - 2.0^j * dpow(i, j, 0.0)
        end
    end
    r[2] = -1
    d == k + 1 && return C \ r
    ts = range(-0.5, 0; length=2001)
    A = [t^i for t in ts, i in 0:d]
    c0 = C \ r
    Z = nullspace(C)
    return c0 + Z * ((A * Z) \ (log2.(1 .+ ts) - A * c0))
end

struct Log2Axis end

struct NQTAxis
    c::Vector{Float64}      # f
    dc::Vector{Float64}     # f'
    d2c::Vector{Float64}    # f''
end

function NQTAxis(c::Vector{Float64})
    dc = [i * c[i + 1] for i in 1:(length(c) - 1)]
    d2c = [i * (i - 1) * c[i + 1] for i in 2:(length(c) - 1)]
    return NQTAxis(c, isempty(dc) ? [0.0] : dc, isempty(d2c) ? [0.0] : d2c)
end

"""`ξ = lg(ρ)` and its first two ρ-derivatives."""
axis(::Log2Axis, ρ) = (log2(ρ), 1 / (ρ * log(2)), -1 / (ρ^2 * log(2)))
function axis(a::NQTAxis, ρ)
    m, p = frexp(ρ)
    t = m - 1
    return (p + evalpoly(t, a.c), evalpoly(t, a.dc) / exp2(p), evalpoly(t, a.d2c) / exp2(2p))
end

"""The inverse of [`axis`](@ref): an NQT `exp2`. Safeguarded Newton, since `f` is
a polynomial of any degree here."""
inverse(::Log2Axis, ξ) = exp2(ξ)
function inverse(a::NQTAxis, ξ)
    p = floor(Int, ξ) + 1
    r = ξ - p
    lo, hi = -0.5, 0.0
    t = r / 2
    for _ in 1:100
        F = evalpoly(t, a.c) - r
        F > 0 ? (hi = t) : (lo = t)
        t_next = t - F / evalpoly(t, a.dc)
        lo <= t_next <= hi || (t_next = (lo + hi) / 2)
        t_next == t && break
        t = t_next
    end
    return ldexp(1 + t, p)
end

const T_SAMPLES = range(-0.5, 0; length=10_001)
wiggle(::Log2Axis) = 0.0
wiggle(a::NQTAxis) = maximum(abs(evalpoly(t, a.c) - log2(1 + t)) for t in T_SAMPLES)
monotone(::Log2Axis) = true
monotone(a::NQTAxis) = all(evalpoly(t, a.dc) > 0 for t in T_SAMPLES)

# ---------------------------------------------------------------------------
# 1 and 2: a spline on each axis
# ---------------------------------------------------------------------------

# A sum of power laws, as an EOS nearly is piecewise; on a log axis each is a
# straight line.
U(ρ) = ρ^0.5 + 0.1 * ρ^1.2
U_ρ(ρ) = 0.5 * ρ^-0.5 + 0.12 * ρ^0.2
U_ρρ(ρ) = -0.25 * ρ^-1.5 + 0.024 * ρ^-0.8

"""
    spline_errors(a, h; ξ_hi=20.0)

Fit `log2 U` at nodes spaced `h` in `ξ = lg(ρ)` over `[0, ξ_hi]`, and return the
maximum relative errors of U, ρ²U_ρ and `2ρU_ρ + ρ²U_ρρ` (h·cs²'s numerator)
over `log2 ρ ∈ [1, ξ_hi - 1]`.
"""
function spline_errors(a, h; ξ_hi=20.0)
    ξs = range(0.0, ξ_hi; length=round(Int, ξ_hi / h) + 1)
    sp = E.BsplineView1(E.fit_bspline_1d([log2(U(inverse(a, ξ))) for ξ in ξs]), 0.0, step(ξs))
    eU = eP = eC = 0.0
    for lgρ in range(1.0, ξ_hi - 1; length=100_001)
        ρ = exp2(lgρ)
        ξ, ξ_ρ, ξ_ρρ = axis(a, ρ)
        b = E.bspline_eval1(sp, ξ)
        u = exp2(b.f)
        u_ρ = u * log(2) * b.fx * ξ_ρ
        u_ρρ = u * log(2) * (b.fxx * ξ_ρ^2 + b.fx * ξ_ρρ + log(2) * b.fx^2 * ξ_ρ^2)
        eU = max(eU, abs(u / U(ρ) - 1))
        eP = max(eP, abs(u_ρ / U_ρ(ρ) - 1))
        eC = max(eC, abs((2ρ * u_ρ + ρ^2 * u_ρρ) / (2ρ * U_ρ(ρ) + ρ^2 * U_ρρ(ρ)) - 1))
    end
    return eU, eP, eC
end

function accuracy()
    println("\n1. Maximum relative errors at spacing h = 1/8, which puts the powers of two on")
    println("   grid points, and for h·cs² also at h = 0.13, which does not. \"wiggle\" is the")
    println("   largest deviation of the NQT log₂ from log₂.")
    @printf("   %-22s %8s %4s %9s %9s %9s %9s\n", "axis", "wiggle", "mono", "U", "ρ²U_ρ", "h·cs²", "h=0.13")
    variants = Tuple{String,Any}[("log₂", Log2Axis())]
    for (k, d) in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 7), (2, 9), (2, 11))
        label = d == k + 1 ? "C$k, degree $d" : "C$k, degree $d, fitted"
        push!(variants, (label, NQTAxis(nqt_coeffs(k, d))))
    end
    for (label, a) in variants
        e = spline_errors(a, 1 / 8)
        e13 = spline_errors(a, 0.13)
        @printf("   %-22s %8.1e %4s %9.1e %9.1e %9.1e %9.1e\n", label, wiggle(a), monotone(a) ? "yes" : "NO",
                e..., e13[3])
    end
    @printf("   C2, degree 3: f(t) = (%s)·tⁱ / 7, i = 0..3\n", join(round.(7 .* nqt_coeffs(2, 3); digits=12), ", "))

    println("\n2. Convergence: the same errors as h halves; 1/4 to 1/32 put the powers of two")
    println("   on grid points, 0.13 and 0.065 do not.")
    for (label, a) in (("log₂", Log2Axis()), ("C0 (NQTo1)", NQTAxis(nqt_coeffs(0, 1))),
                       ("C1 (NQTo2)", NQTAxis(nqt_coeffs(1, 2))), ("C2, degree 3", NQTAxis(nqt_coeffs(2, 3))))
        for h in (1 / 4, 1 / 8, 1 / 16, 1 / 32, 0.13, 0.065)
            @printf("   %-13s h = %-7.4g %9.2e %9.2e %9.2e\n", label, h, spline_errors(a, h)...)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# 3: the hyperbolic identities
# ---------------------------------------------------------------------------

function hyperbolics()
    println("\n3. Hyperbolic functions from an NQT exp, E(w/ln 2), over w/ln 2 ∈ (0, 8]:")
    xs = range(0, 8; length=80_001)[2:end]
    qs = exp10.(range(0, 8; length=400_001))[2:end]
    for (label, a) in (("C0 (NQTo1)", NQTAxis(nqt_coeffs(0, 1))), ("C1 (NQTo2)", NQTAxis(nqt_coeffs(1, 2))))
        cosh_q(x) = (inverse(a, x) + inverse(a, -x)) / 2
        sinh_q(x) = (inverse(a, x) - inverse(a, -x)) / 2
        # cosh² - sinh² = E(x)E(-x); then W = cosh against 1/√(1 - v²) with v = tanh.
        ident = maximum(abs(cosh_q(x)^2 - sinh_q(x)^2 - 1) for x in xs)
        W_v = maximum(abs(cosh_q(x) / sqrt(cosh_q(x)^2 - sinh_q(x)^2) / cosh_q(x) - 1) for x in xs)
        half = maximum(abs(2sinh_q(x / 2)^2 / (cosh_q(x) - 1) - 1) for x in xs)
        # An NQT acosh paired with the true cosh: how far D/cosh(w) lands from ρ.
        acosh_q(q) = log(2) * axis(a, q + sqrt(q^2 - 1))[1]
        miss = maximum(abs(cosh(acosh_q(q)) / q - 1) for q in qs)
        @printf("   %-11s |cosh² - sinh² - 1| ≤ %.3g, W vs 1/√(1 - v²) ≤ %.2g, 2sinh²(w/2) vs cosh w - 1 ≤ %.2g,\n",
                label, ident, W_v, half)
        @printf("   %-11s acosh then true cosh misses q ∈ [1, 1e8] by ≤ %.2g\n", "", miss)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# 4: what the transcendental calls cost
# ---------------------------------------------------------------------------

const SINK = Ref(0.0)

"""Best of five, in ns per state; the result goes to `SINK` so nothing is elided."""
function best_ns(f, n)
    t = typemax(UInt64)
    for _ in 1:5
        t0 = time_ns()
        SINK[] += f()
        t = min(t, time_ns() - t0)
    end
    return t / n
end

function run_evaluate(v, sts)
    acc = 0.0
    for st in sts
        acc += evaluate(v, st.ρ, st.s, st.yₑ, NaN).p
    end
    return acc
end

# What `locate` and `eval_at` call: log10 ρ, then exp10 of x, of u and of L.
function run_logs(ρs, xs, us, Ls)
    acc = 0.0
    for i in eachindex(ρs)
        acc += log10(ρs[i]) + exp10(xs[i]) + exp10(us[i]) + exp10(Ls[i])
    end
    return acc
end

function run_con2prim(v, cins, sts, opts)
    acc = 0.0
    for i in eachindex(cins)
        st = sts[i]
        acc += con2prim(v, cins[i], opts, st.s * (1 + 1e-3), st.w + 1e-3, st.u + 1e-3).ρ
    end
    return acc
end

# What `residuals` calls, once per evaluation.
function run_hyperbolics(ws)
    acc = 0.0
    for w in ws
        acc += cosh(w) + sinh(w) + tanh(w) + sinh(0.5 * w)
    end
    return acc
end

function timing(name, v)
    rng = StableRNG(4242)
    sts = [sample_valid(v, rng) for _ in 1:N]
    pts = [evaluate(v, st.ρ, st.s, st.yₑ, NaN) for st in sts]
    ρs = [st.ρ for st in sts]
    xs = log10.(ρs)
    us = [pt.u_solved for pt in pts]
    Ls = [E.bspline_eval3(v.L, xs[i], us[i], sts[i].yₑ).f for i in 1:N]
    t_eval = best_ns(() -> run_evaluate(v, sts), N)
    t_logs = best_ns(() -> run_logs(ρs, xs, us, Ls), N)

    opts = E.Con2PrimOptions{Float64}()
    cins = map(sts) do st
        c = E.prim2con(v, st.ρ, st.s, st.yₑ, st.w, st.B², st.cos_vB, st.u)
        return E.Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²)
    end
    outs = [con2prim(v, cins[i], opts, sts[i].s * (1 + 1e-3), sts[i].w + 1e-3, sts[i].u + 1e-3) for i in 1:N]
    # One residual evaluation per Newton step plus the first; the fallback's are not counted.
    ws = reduce(vcat, [fill(sts[i].w, outs[i].iters_newton + 1) for i in 1:N])
    t_c2p = best_ns(() -> run_con2prim(v, cins, sts, opts), N)
    t_hyp = best_ns(() -> run_hyperbolics(ws), N)

    @printf("   %-10s evaluate, cold:   %7.1f ns, %4.1f temperature-solve iterations; log10 + 3 exp10: %5.1f ns = %.1f%%\n",
            name, t_eval, sum(pt.iters for pt in pts) / N, t_logs, 100t_logs / t_eval)
    @printf("   %-10s con2prim, warm:   %7.1f ns, %4.2f residual evaluations; cosh, sinh, tanh, sinh(w/2): %5.1f ns = %.1f%%\n",
            "", t_c2p, length(ws) / N, t_hyp, 100t_hyp / t_c2p)
    return nothing
end

println("EntropyEOS NQT study: Julia $VERSION, $(Sys.CPU_NAME), n = $N")
accuracy()
hyperbolics()
if haskey(FLAGS, "tables")
    println("\n4. Time per state on one thread; states as test/test_real_tables.jl samples them,")
    println("   con2prim warm-started from the truth perturbed by 1e-3.")
    for (name, v) in load_tables(FLAGS["tables"]; only=get(FLAGS, "only", nothing))
        name == "synthetic" && continue
        timing(name, v)
    end
else
    println("\n4. Skipped: no --tables.")
end

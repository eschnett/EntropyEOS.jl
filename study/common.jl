# Shared by the study scripts: the table roster, loading, and statistics.
#
# These scripts are measurements, not tests. They are run by hand (on a GPU
# node for the device parts) and their output is what `docs/src/precision.md`
# reports. Nothing here is loaded by the package or its test suite.

using EntropyEOS
using Printf
using StableRNGs

const E = EntropyEOS

# The baryon-mass convention is per table family and is NOT stored in the file.
const REAL_TABLES = [
    (name="LS220", file="LS220_234r_136t_50y_analmu_20091212_SVNr26.h5", m_B=E.M_AMU_G),
    (name="SRO LS220", file="LS220_3335_rho391_temp163_ye66.h5", m_B=E.M_NEUTRON_G),
    (name="DD2", file="Hempel_DD2EOS_rho234_temp180_ye60_version_1.1_20120817.h5", m_B=E.M_AMU_G),
    (name="DD2 repaired", file="Hempel_DD2EOS_rho234_temp180_ye60_version_1.1_20120817_repaired.h5",
     m_B=E.M_AMU_G),
    (name="SFHo", file="Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817.h5", m_B=E.M_AMU_G),
]

"""
    load_tables(dir; only=nothing)

`(name, v)` pairs of Float64 views: the synthetic ideal gas first, then every
real table found in `dir`. `only` restricts to names containing that substring.
"""
function load_tables(dir; only=nothing)
    out = Tuple{String,EOSTableView{Float64,Array{Float64,3}}}[]
    keep(name) = only === nothing || occursin(lowercase(only), lowercase(name))
    keep("synthetic") && push!(out, ("synthetic", EOSTableView(E.build_eos(make_synthetic_table()))))
    for tab in REAL_TABLES
        keep(tab.name) || continue
        path = joinpath(dir, tab.file)
        isfile(path) || (@warn "skipping $(tab.name): not found" path; continue)
        t0 = time()
        t = read_stellarcollapse(path)
        v = EOSTableView(E.build_eos(t, BuildOptions(; m_B_table_g=tab.m_B)))
        @printf("loaded %-13s %s in %.1f s, κ = %.10f\n", tab.name, string(size(t)), time() - t0, v.κ)
        push!(out, (tab.name, v))
    end
    return out
end

"""
    sample_valid(v, rng; w_max_sample=6.0, sigma_max=1e4, margin=0.05)

A valid primitive state, sampled as `test/test_real_tables.jl` does: inside a
`margin` of the density box and of the entropy window, `w` uniform in
`[0, w_max_sample]`, and magnetized in 90% of cases, log-uniformly in `B²/(ρh)`.
"""
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

converged(r) = r === E.C2PResult.converged_newton || r === E.C2PResult.converged_fallback

"""Quantile by the C++ harness's convention, `v[floor(p*(n-1))]` of the sorted data."""
function pct(v, p)
    isempty(v) && return NaN
    s = sort(v)
    return s[1+floor(Int, p * (length(s) - 1))]
end
maxv(v) = isempty(v) ? NaN : maximum(v)

"""The C++ harness's symmetric relative difference."""
rel_diff(a, b) = abs(a - b) / max(abs(a), abs(b), 1e-300)

"""Round-trip error against ground truth; `floor` guards a vanishing truth (w → 0)."""
rt_err(got, truth, floor=0.0) = abs(got - truth) / max(abs(truth), floor, 1e-300)

"""`p50 / p99 / p999 / max` of a sample, formatted on one line."""
function qline(v)
    isempty(v) && return "(empty)"
    return @sprintf("%.2e / %.2e / %.2e / %.2e", pct(v, 0.5), pct(v, 0.99), pct(v, 0.999), maxv(v))
end

"""Parse `--key=value` flags into a Dict; bare `--flag` maps to "true"."""
function parse_flags(args)
    d = Dict{String,String}()
    for a in args
        m = match(r"^--([^=]+)(?:=(.*))?$", a)
        m === nothing && error("unrecognized argument: $a")
        d[m[1]] = something(m[2], "true")
    end
    return d
end

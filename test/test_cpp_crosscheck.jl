# Cross-check against the C++ original, through the `eos_test` binary that
# EntropyEOS_jll ships. The C++ writes each synthetic table to disk, and both
# implementations then examine the same bits; only reports are compared, since
# the JLL exposes executables and no callable library.
#
# One difference is expected and tolerated: on a logenergy plateau (planted in
# the dirty table) the ε values of a T stencil are equal, so `fd3` returns a pure
# rounding residual, which depends on the last bit of each ε (the platform's
# exp10) and on whether the build contracts the sum into FMAs. With a zero
# residual `delta_T` overflows to Inf and the node is skipped; otherwise it is
# reported as 1e16 to 1e17. Either side may skip such a node, depending on the
# platform, so they are excluded below, and every other node must agree.

# `ENTROPYEOS_EOS_TEST` names an `eos_test` binary to use instead of the JLL's,
# to check against a C++ release the JLL does not ship yet.
const CPP_EOS_TEST = get(ENV, "ENTROPYEOS_EOS_TEST", "")
const CPP_AVAILABLE = !isempty(CPP_EOS_TEST) || EntropyEOS_jll.is_available()

"""
Run `eos_test` with `args`, returning its stdout with Unix line endings (the
Windows build writes CRLF). The exit code is a verdict, not an error.
"""
function _eos_test(args)
    exe = isempty(CPP_EOS_TEST) ? EntropyEOS_jll.eos_test() : CPP_EOS_TEST
    return replace(read(ignorestatus(`$exe $args`), String), "\r\n" => "\n")
end

const CPP_VERSION = CPP_AVAILABLE ? VersionNumber(match(r"(\d+\.\d+\.\d+)", _eos_test(["--version"]))[1]) : v"0"

# From 1.2.0 the C++ repair drops the table's hottest T row
# (`RepairOptions::trim_t_hi_rows`), and `eos_test --level con2prim` repairs in
# memory before it samples. Julia builds with the same trim to see that table.
const CPP_TRIM = CPP_VERSION >= v"1.2.0" ? 1 : 0

"""The `name: count=… max=… rms=…` summary lines of a C++ report, in order."""
function _cpp_summary(out)
    rows = NamedTuple{(:name, :count, :max, :rms),Tuple{String,Int,Float64,Float64}}[]
    for m in eachmatch(r"^(\w+): count=(\d+) max=(\S+) rms=(\S+)$"m, out)
        push!(rows, (name=m[1], count=parse(Int, m[2]), max=parse(Float64, m[3]), rms=parse(Float64, m[4])))
    end
    return rows
end

"""A `--csv` file as `(iρ, jT, kYₑ) => value`, with the indices made 1-based."""
function _cpp_csv(path)
    out = Dict{NTuple{3,Int},Float64}()
    isfile(path) || return out
    for line in Iterators.drop(eachline(path), 1)
        p = split(rstrip(line), ',')   # CRLF on Windows
        out[(parse(Int, p[1]) + 1, parse(Int, p[2]) + 1, parse(Int, p[3]) + 1)] = parse(Float64, p[7])
    end
    return out
end

"""Nodes whose `fd3` stencil in T contains two equal logenergy values; see the file comment."""
function _flat_T_stencils(t)
    le = field(t, "logenergy")
    nρ, nT, nYₑ = size(t)
    flat = Set{NTuple{3,Int}}()
    for k in 1:nYₑ, j in 1:nT, i in 1:nρ
        j0 = clamp(j - 1, 1, nT - 2)
        any(le[i, jj, k] == le[i, jj + 1, k] for jj in j0:(j0 + 1)) && push!(flat, (i, j, k))
    end
    return flat
end

_cpp_scalar(out, name) = parse(Float64, match(Regex("^$name=(\\S+)\$", "m"), out)[1])

# The C++ audit's per-state generator (splitmix64 seeding, then xorshift64*).
# Porting it makes Julia solve exactly the states the C++ solved.
mutable struct _SampleRng
    state::UInt64
end

function _SampleRng(seed::Integer, index::Integer)
    z = UInt64(seed) * 0x9E3779B97F4A7C15 + UInt64(index) + 0x1 + 0x9E3779B97F4A7C15
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    z ⊻= z >> 31
    return _SampleRng(iszero(z) ? 0x9E3779B97F4A7C15 : z)
end

function _next_unit!(r::_SampleRng)
    s = r.state
    s ⊻= s >> 12
    s ⊻= s << 25
    s ⊻= s >> 27
    r.state = s
    return Float64((s * 0x2545F4914F6CDD1D) >> 11) / 2.0^53
end

"""State `k` of `check_con2prim`'s default sampling, with its truth and warm start."""
function _cpp_sample_state(v, k; seed=987654, w_max=6.0, σ_max=1e4, warm_rel=1e-3, margin=0.05)
    rng = _SampleRng(seed, k)
    xspan = v.x_hi - v.x_lo
    xlo, xhi = v.x_lo + margin * xspan, v.x_hi - margin * xspan
    ρ = exp10(xlo + _next_unit!(rng) * (xhi - xlo))
    yₑ = v.y_lo + _next_unit!(rng) * (v.y_hi - v.y_lo)
    sr = srange(v, ρ, yₑ)
    sspan = sr.s_max - sr.s_min
    s = sr.s_min + margin * sspan + _next_unit!(rng) * (sspan - 2margin * sspan)
    w = _next_unit!(rng) * w_max
    b2_zero = _next_unit!(rng) < 0.1
    σ = exp(log(1e-6) + _next_unit!(rng) * (log(σ_max) - log(1e-6)))
    cos_vB = -1 + 2 * _next_unit!(rng)
    u1 = -1 + 2 * _next_unit!(rng)
    u2 = -1 + 2 * _next_unit!(rng)
    u3 = -1 + 2 * _next_unit!(rng)
    pt = evaluate(v, ρ, s, yₑ, NaN)
    B² = b2_zero ? 0.0 : σ * ρ * pt.h
    truth = prim2con(v, ρ, s, yₑ, w, B², cos_vB, pt.u_solved)
    guess = (s * (1 + warm_rel * u1), w + warm_rel * u2, pt.u_solved + warm_rel * u3)
    return (; ρ, s, yₑ, w, B², cos_vB, truth, guess)
end

"""The C++ harness's nearest-rank quantile."""
_nearest_rank(sorted, p) = sorted[clamp(ceil(Int, p * length(sorted)), 1, length(sorted))]

if !CPP_AVAILABLE
    @info "C++ cross-check skipped (EntropyEOS_jll has no build for this platform)"
else
    @testset "C++ cross-check" begin
        E = EntropyEOS
        dir = mktempdir()

        @testset "table: $variant" for variant in ("synthetic", "synthetic-seeded", "synthetic-dirty")
            h5 = joinpath(dir, "$variant.h5")
            gen = _eos_test(["--$variant", "--write-synthetic", h5])
            t = read_stellarcollapse(h5)
            if variant == "synthetic-dirty" && CPP_VERSION >= v"1.2.0"
                # 1.2.0's dirty preset adds a SeamDefect, which the Julia
                # generator does not have yet.
                @test_broken false
                continue
            end

            # Julia's generator makes the same table, to libm rounding.
            seeds = [E.SeededViolation(m[1], parse(Int, m[2]) + 1, parse(Int, m[3]) + 1, parse(Int, m[4]) + 1,
                                       parse(Float64, m[5]))
                     for m in eachmatch(r"^  (\w+) at \(irho=(\d+), jT=(\d+), kYe=(\d+)\) delta=(\S+)$"m, gen)]
            @test length(seeds) == (variant == "synthetic-seeded" ? 4 : 0)
            opts = variant == "synthetic-dirty" ? dirty_synthetic_options() : SyntheticOptions(; seed=seeds)
            tj = make_synthetic_table(opts)
            @test size(tj) == size(t)
            @test sort(field_names(tj)) == sort(field_names(t))
            @test tj.logρ ≈ t.logρ rtol = 1e-14
            @test tj.logT ≈ t.logT rtol = 1e-14
            @test energy_shift(tj) == energy_shift(t)
            for name in field_names(tj)
                a, b = field(t, name), field(tj, name)
                @test isfinite.(a) == isfinite.(b)
                @test all(isequal(a[i], b[i]) || isapprox(a[i], b[i]; rtol=1e-14) for i in eachindex(a))
            end

            flat = _flat_T_stencils(t)
            @test isempty(flat) == (variant != "synthetic-dirty")
            fma_sensitive(name) = name == "delta_T" && !isempty(flat)

            # The summary at the default threshold: every class, in order.
            cpp = _cpp_summary(_eos_test([h5, "--worst", "0"]))
            jl = check_table(t)
            @test jl.status === Status.ok
            @test [r.name for r in cpp] == [c.name for c in jl.classes]
            for (r, c) in zip(cpp, jl.classes)
                fma_sensitive(r.name) && continue
                @test r.count == c.count
                @test isapprox(r.max, c.max; rtol=1e-4) || isnan(c.max)
                @test isapprox(r.rms, c.rms; rtol=1e-4) || isnan(c.rms)
            end

            # Every node, through the CSV dump. A zero threshold makes every
            # evaluated point a finding, so the diagnostic classes are dumped
            # in full too.
            prefix = joinpath(dir, "$variant-all")
            _eos_test([h5, "--tol", "0", "--csv", prefix, "--worst", string(10^9)])
            for c in check_table(t, CheckOptions(; tol_consistency=0.0, worst_n=10^9)).classes
                cc = _cpp_csv("$(prefix)_$(c.name).csv")
                jj = Dict((l.iρ, l.jT, l.kYₑ) => Float64(l.value) for l in c.worst)
                if fma_sensitive(c.name)
                    # Only plateau nodes may be missing, on either side, or disagree.
                    @test symdiff(keys(cc), keys(jj)) ⊆ flat
                    @test all(haskey(cc, k) && isapprox(cc[k], jj[k]; rtol=1e-9) for k in keys(jj) if k ∉ flat)
                else
                    @test keys(cc) == keys(jj)
                    @test all(isequal(cc[k], jj[k]) || isapprox(cc[k], jj[k]; rtol=1e-9) for k in keys(jj))
                end
            end
        end

        # The adapter's κ rescaling. The dirty table is taken unrepaired, as
        # Julia has no repair step.
        @testset "adapter: $variant" for variant in ("synthetic", "synthetic-dirty")
            h5 = joinpath(dir, "$variant.h5")
            out = _eos_test(["--level", "adapter", h5, "--no-repair", "--soak", "1000", "--node-stride", "8", "--worst", "0"])
            eos = build_eos(read_stellarcollapse(h5))
            @test _cpp_scalar(out, "kappa") ≈ eos.view.κ rtol = 1e-13
            @test _cpp_scalar(out, "m_B_star_g") ≈ eos.m_B_star_g rtol = 1e-13
        end

        # The con2prim audit, on exactly the C++'s states. Round-trip and
        # recovery errors are compared by quantile; at rounding level (below
        # 1e-14) they are noise and only bounded. Solver paths are not
        # compared: Julia's deliberate deviations change the fallback count.
        @testset "con2prim" begin
            h5 = joinpath(dir, "synthetic.h5")
            out = _eos_test(["--level", "con2prim", h5, "--worst", "0"])
            # Both see the same table: the repair changes no value, and drops
            # the rows Julia's build trims.
            if CPP_TRIM == 0
                @test occursin("repair (in-memory): status=ok entries=0", out)
            else
                @test occursin(r"^repair \(in-memory\): status=repaired entries=0 seams=0 t_rows_dropped=1 "m, out)
            end
            @test occursin(r"^warm: .* n_failed_total=0$"m, out)
            @test occursin(r"^cold: .* n_failed_total=0$"m, out)

            v = EOSTableView(build_eos(read_stellarcollapse(h5), BuildOptions(; trim_t_hi_rows=CPP_TRIM)))
            opts = Con2PrimOptions()
            converged(o) = o.result === E.C2PResult.converged_newton || o.result === E.C2PResult.converged_fallback
            rel(a, b) = abs(a - b) / max(abs(b), 1e-30)
            errs = Dict(name => Float64[] for name in ("rt_D", "rt_tau", "rt_S", "prim_rho", "prim_w", "prim_s"))
            nfail = 0
            for k in 0:19999
                st = _cpp_sample_state(v, k)
                c = st.truth
                cin = Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²)
                o = con2prim(v, cin, opts, st.guess...)
                nfail += !converged(o)
                k % 10 == 0 && (nfail += !converged(con2prim(v, cin, opts)))
                V = hypot(o.v_par, o.v_perp)
                b = prim2con(v, o.ρ, o.s, o.ye, o.w, st.B², V > 1e-10 ? o.v_par / V : st.cos_vB, o.eos.u_solved)
                push!(errs["rt_D"], rel(b.D, c.D))
                push!(errs["rt_tau"], rel(b.τ, c.τ))
                push!(errs["rt_S"], max(rel(b.S_par, c.S_par), rel(b.S_perp, c.S_perp)))
                push!(errs["prim_rho"], rel(o.ρ, st.ρ))
                push!(errs["prim_w"], abs(o.w - st.w) / (1 + st.w))
                push!(errs["prim_s"], abs(o.s - st.s) / max(abs(st.s), 1.0))
            end
            @test nfail == 0

            for (name, e) in errs
                m = match(Regex("^$name quantiles: p50=(\\S+) p90=(\\S+) p99=(\\S+) p999=(\\S+) max=(\\S+)\$", "m"), out)
                sort!(e)
                for (i, p) in enumerate((0.5, 0.9, 0.99, 0.999))
                    qc, qj = parse(Float64, m[i]), _nearest_rank(e, p)
                    @test max(qc, qj) < 1e-14 || 1 / 2 < qj / qc < 2
                end
                # The single worst state is a tail, not a statistic.
                @test max(parse(Float64, m[5]), e[end]) < 1e-6
            end
        end
    end
end

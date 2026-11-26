# Tests against real stellarcollapse tables, which are hundreds of megabytes
# and are not in git. Point ENTROPYEOS_TABLE_DIR at a directory holding them;
# the file skips cleanly otherwise, so CI needs nothing.
#
# The state sampling below deliberately mirrors the C++ audit harness
# (`host/con2prim_audit.cpp`): 5%-per-side margins on the density box and on
# the pointwise entropy window, a 10% zero-field coin, log-uniform
# magnetization, and a warm start perturbed by 1e-3 rather than handed the
# exact answer. Matching it is what makes the failure counts and round-trip
# quantiles comparable between the two implementations -- sampling the extreme
# corners instead raises the cold failure rate by an order of magnitude in
# both, which says something about the tables, not about the solver.

const TABLE_DIR = get(ENV, "ENTROPYEOS_TABLE_DIR", "")

# The baryon-mass convention is per table family and is NOT stored in the file.
const REAL_TABLES = [
    (name="LS220", file="LS220_234r_136t_50y_analmu_20091212_SVNr26.h5",
     m_B=EntropyEOS.M_AMU_G, size=(234, 136, 50)),
    (name="SRO LS220", file="LS220_3335_rho391_temp163_ye66.h5",
     m_B=EntropyEOS.M_NEUTRON_G, size=(391, 163, 66)),
    (name="DD2", file="Hempel_DD2EOS_rho234_temp180_ye60_version_1.1_20120817.h5",
     m_B=EntropyEOS.M_AMU_G, size=(234, 180, 60)),
    (name="DD2 repaired", file="Hempel_DD2EOS_rho234_temp180_ye60_version_1.1_20120817_repaired.h5",
     m_B=EntropyEOS.M_AMU_G, size=(234, 180, 60)),
    (name="SFHo", file="Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817.h5",
     m_B=EntropyEOS.M_AMU_G, size=(222, 180, 60)),
]

"""Mirror of the C++ audit's `sample_state`."""
function _sample_real_state(v, rng; w_max_sample=6.0, sigma_max=1e4, margin=0.05)
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
    pt = evaluate(v, ρ, s, yₑ, NaN)
    return (ρ=ρ, s=s, yₑ=yₑ, w=w, B²=(b2_zero ? 0.0 : σ * ρ * pt.h),
            cos_vB=-1 + 2rand(rng), u=pt.u_solved)
end

"""μ_ν = mu_e − muhat + c on the coldest row, bilinear in raw `log10 ρ` and `Yₑ`: an independent check of a root."""
function _munu_cold(t, c, x, y)
    me, mh = field(t, "mu_e"), field(t, "muhat")
    μ(i, k) = me[i, 1, k] - mh[i, 1, k] + c
    i = clamp(searchsortedlast(t.logρ, x), 1, length(t.logρ) - 1)
    k = clamp(searchsortedlast(t.Yₑ, y), 1, length(t.Yₑ) - 1)
    a = (x - t.logρ[i]) / (t.logρ[i + 1] - t.logρ[i])
    b = (y - t.Yₑ[k]) / (t.Yₑ[k + 1] - t.Yₑ[k])
    return (1 - a) * (1 - b) * μ(i, k) + a * (1 - b) * μ(i + 1, k) + (1 - a) * b * μ(i, k + 1) + a * b * μ(i + 1, k + 1)
end

# The conventions, clamping and phase transitions were surveyed on these tables
# before the slice was written (CODE.md, "β-equilibrium"); the expectations
# below are that survey's.
function _test_beta_equilibrium(name, t, eos)
    v = EOSTableView(eos)
    lκ = log10(v.κ)
    mₑ = EntropyEOS.M_ELECTRON_MEV
    oconnor_ott = name == "LS220"
    if oconnor_ott
        # Its ε is too noisy above 10^13.5 g/cm³ for the electron-mass check,
        # which refuses it; given explicitly, with the default's value:
        @test_throws ArgumentError cold_beta_slice(t, eos)
        sl = cold_beta_slice(t, eos; energy_offset=mₑ)
        @test sl.munu_offset == 0
        @test 0.55 < sl.munu_mismatch < 0.7    # its munu subtracts a further Q above ~10⁸ g/cm³
    else
        sl = cold_beta_slice(t, eos)
        @test sl.munu_offset == 0
        @test sl.munu_mismatch == 0
        @test sl.energy_offset ≈ mₑ
    end
    @test issorted(sl.ρ★) && issorted(sl.p) && issorted(sl.g)
    # Phase transitions only in SRO LS220, just below nuclear density.
    if name == "SRO LS220"
        @test !isempty(sl.jumps)
        @test all(j -> 13 <= log10(j.ρ★_lo) - lκ && log10(j.ρ★_hi) - lκ <= 14, sl.jumps)
    else
        @test isempty(sl.jumps)
    end
    @test all(j -> j.ρ★_hi > j.ρ★_lo && abs(j.g_mismatch) < 1e-5, sl.jumps)

    # Lookups by pressure across the whole slice.
    ps = clamp.(exp.(range(log(sl.p[1]), log(sl.p[end]); length=300)), sl.p[1], sl.p[end])
    st = [beta_equilibrium(sl; p=q) for q in ps]
    @test all(k -> isapprox(st[k].point.p, ps[k]; rtol=1e-8), eachindex(ps))
    @test issorted([r.ρ★ for r in st]) && issorted([r.g for r in st])
    @test all(r -> r.s == srange(v, r.ρ★, r.yₑ).s_min, st)
    root_ok = clamp_ok = sane_low = sane_sat = true
    for r in st
        x = log10(r.ρ★) - lκ
        clamped = r.flags & FLAG_CLAMP_YE != 0
        clamped || abs(_munu_cold(t, sl.munu_offset, x, r.yₑ)) <= 1e-8 || (root_ok = false)
        # Only O'Connor–Ott and SRO LS220 clamp, and only below nuclear density.
        clamped && !(12.5 <= x <= 14.5) && (clamp_ok = false)
        x <= 9 && !(0.4 <= r.yₑ <= 0.51) && (sane_low = false)
        14.3 <= x <= 14.7 && !(r.yₑ < 0.2) && (sane_sat = false)
    end
    @test root_ok
    @test clamp_ok
    @test sane_low
    @test sane_sat

    ys = map((8.0, 10.0, 12.0, 13.0, 14.0, 14.5, 15.0)) do x
        r★ = exp10(x + lκ)
        sl.ρ★[1] <= r★ <= sl.ρ★[end] && !any(j -> j.ρ★_lo < r★ < j.ρ★_hi, sl.jumps) ?
            round(beta_equilibrium(sl; ρ★=r★).yₑ; digits=4) : NaN
    end
    @info "$name: $(length(sl.ρ★)) vertices, $(length(sl.jumps)) jumps; Yₑ at log10 ρ = 8, 10, 12, 13, 14, 14.5, 15: $(join(ys, ", "))"
    return nothing
end

# The composition (CODE.md, "Composition"): its conventions as surveyed on
# these tables, the nodes, μ_ν = 0 on the cold β-equilibrium, and the edge of
# the heavy nuclei.
function _test_composition(name, t, eos)
    E = EntropyEOS
    Q = E.Q_NP_MEV
    v = EOSTableView(eos)
    ct = CompositionTable(t, eos)          # every table builds without help
    cv = CompositionView(ct)
    @test matches(ct, eos)
    oconnor_ott = name == "LS220"
    @test ct.light_fields == (oconnor_ott ? String[] : name == "SRO LS220" ? ["Xl"] : ["Xd", "Xt", "X3he", "X4li"])
    @test ct.munu_offset == 0                  # every table's muhat includes Q

    # Every node: the datasets, and μ_ν = the table's munu, except where
    # O'Connor–Ott's munu subtracts a further Q (above 10⁸ g/cm³).
    nx, nu, ny = size(t)
    mue, muh, munu = field(t, "mu_e"), field(t, "muhat"), field(t, "munu")
    Xn, Xh, Abar = field(t, "Xn"), field(t, "Xh"), field(t, "Abar")
    worst = 0.0
    nodes_ok = true
    nbad = 0
    for k in 1:ny, j in 1:nu, i in 1:nx
        r = E.composition_index(cv, Float64(i - 1), Float64(j - 1), Float64(k - 1))
        want = munu[i, j, k] + (oconnor_ott && t.logρ[i] > 8 ? Q : 0.0)
        δ = abs(r.μ_ν - want)
        # O'Connor–Ott LS220's munu is inconsistent with its mu_e − muhat at 88
        # nodes in the noisy band where its nuclei dissolve (CODE.md,
        # "Composition"), cold and neutron-rich.
        if oconnor_ott && δ > 1e-5 && 13.9 <= t.logρ[i] <= 14.1 && t.logT[j] <= -0.5 && t.Yₑ[k] <= 0.07
            nbad += 1
            continue
        end
        worst = max(worst, δ)
        nodes_ok &= r.μ_e === mue[i, j, k] && r.μ̂ === muh[i, j, k] - ct.munu_offset &&
                    r.X_n === clamp(Xn[i, j, k], 0.0, 1.0) && r.X_h === clamp(Xh[i, j, k], 0.0, 1.0) &&
                    (r.X_h > 0 ? r.Ā === max(Abar[i, j, k], 0.0) : r.Ā === 1.0)
    end
    @test nodes_ok
    @info "$name: munu_offset = $(ct.munu_offset) MeV, max |μ_ν − munu| at the nodes = $worst MeV" nbad
    @test worst <= (oconnor_ott ? 1e-5 : 1e-9)
    @test nbad == (oconnor_ott ? 88 : 0)

    # μ_ν = 0 on the cold β-equilibrium: at T_min to round-off, and at the
    # temperature the slice's states evaluate to within that difference.
    sl = oconnor_ott ? cold_beta_slice(t, eos; energy_offset=E.M_ELECTRON_MEV) : cold_beta_slice(t, eos)
    Tmin = exp10(t.logT[1])
    worst_min = worst_pt = 0.0
    for (ρ★, y, s, cl) in zip(sl.ρ★, sl.yₑ, sl.s, sl.clamped)
        cl && continue
        worst_min = max(worst_min, abs(composition(cv, ρ★, Tmin, y).μ_ν))
        pt = evaluate(v, ρ★, s, y, NaN)
        r = composition(cv, ρ★, pt.T_MeV, y)
        fi, fk = (log10(ρ★) - cv.x0) / cv.hx, (y - cv.y0) / cv.hy
        slope = abs(E.composition_index(cv, fi, 1.0, fk).μ_ν - E.composition_index(cv, fi, 0.0, fk).μ_ν) / cv.hu
        du = max(log10(pt.T_MeV) - t.logT[1], 0.0)
        worst_pt = max(worst_pt, abs(r.μ_ν) - du * slope)
    end
    @info "$name: |μ_ν| on the cold slice: at T_min $worst_min MeV, at the states' T $worst_pt MeV beyond the T bound"
    @test worst_min <= 1e-6
    @test worst_pt <= 1e-6

    @testset "the edge of the heavy nuclei" begin
        _test_comp_edge(cv, StableRNG(1204))
    end
    return fingerprint(eos)
end

if isempty(TABLE_DIR) || !isdir(TABLE_DIR)
    @info "real-table tests skipped (set ENTROPYEOS_TABLE_DIR to enable)"
else
    @testset "real tables" begin
        E = EntropyEOS
        audits = Dict{String,Int}()
        fingerprints = Dict{String,UInt64}()

        # The small cropped LS220 fixture, if present: cheap, and it carries
        # LS220's three genuine non-finite cs2/gamma points.
        crop = joinpath(TABLE_DIR, "LS220_san_crop.h5")
        if isfile(crop)
            @testset "LS220 crop" begin
                t = read_stellarcollapse(crop)
                @test size(t) == (61, 136, 16)
                @test count(!isfinite, field(t, "cs2")) == 3
                @test count(!isfinite, field(t, "gamma")) == 3
                rep = check_table(t)
                # Non-finite values in columns the pipeline never interprets are
                # reported, not fatal -- which is what lets LS220 be used at all.
                @test rep.status !== Status.fatal
                @test Dict(c.name => c.count for c in rep.classes)["nonfinite_cs2"] == 3
            end
        end

        for tab in REAL_TABLES
            path = joinpath(TABLE_DIR, tab.file)
            if !isfile(path)
                @info "skipping $(tab.name): not found" path
                continue
            end

            @testset "$(tab.name)" begin
                t = read_stellarcollapse(path)
                @test size(t) == tab.size
                @test validate_axes(t) === nothing
                @test energy_shift(t) > 0
                @test all(isfinite, field(t, "entropy"))
                @test all(isfinite, field(t, "logenergy"))

                rep = check_table(t, CheckOptions(; m_B_g=tab.m_B))
                @test rep.status !== Status.fatal

                eos = E.build_eos(t, BuildOptions(; m_B_table_g=tab.m_B))
                v = EOSTableView(eos)
                @test 0 < v.κ <= 1
                @test eos.m_B_star_g ≈ v.κ * tab.m_B
                @test v.x_lo < v.x_hi && v.u_lo < v.u_hi && v.y_lo < v.y_hi
                audits[tab.name] = eos.audit.σ_u.violation_count

                if has_field(t, "mu_e") && has_field(t, "muhat")
                    @testset "β-equilibrium" begin
                        _test_beta_equilibrium(tab.name, t, eos)
                    end
                    @testset "composition" begin
                        fingerprints[tab.name] = _test_composition(tab.name, t, eos)
                    end
                end

                opts = Con2PrimOptions()
                rng = StableRNG(987654)
                n = 5000
                nwarm_fail = ncold_fail = nsplit = 0
                errs = Float64[]
                for k in 1:n
                    st = _sample_real_state(v, rng)
                    c = prim2con(v, st.ρ, st.s, st.yₑ, st.w, st.B², st.cos_vB, st.u)
                    cin = Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²)
                    wr = 1e-3
                    o = con2prim(v, cin, opts, st.s * (1 + wr * (-1 + 2rand(rng))),
                                 st.w + wr * (-1 + 2rand(rng)), st.u + wr * (-1 + 2rand(rng)))
                    converged = o.result === E.C2PResult.converged_newton ||
                                o.result === E.C2PResult.converged_fallback
                    converged || (nwarm_fail += 1)
                    converged && push!(errs, abs(o.ρ - st.ρ) / st.ρ)
                    if k % 10 == 0
                        oc = con2prim(v, cin, opts)
                        (oc.result === E.C2PResult.converged_newton ||
                         oc.result === E.C2PResult.converged_fallback) || (ncold_fail += 1)
                        # The second stage rebuilds the first's iterate exactly.
                        nsplit += con2prim_finish(v, cin, opts, con2prim_start(v, cin, opts)[2]) !== oc
                    end
                end
                sort!(errs)
                p(f) = errs[max(1, ceil(Int, f * length(errs)))]
                @info "$(tab.name): κ=$(v.κ) warm_fail=$nwarm_fail/$n cold_fail=$ncold_fail/$(n÷10)" p50 = p(0.5) p99 = p(0.99) p999 = p(0.999) max = errs[end]

                # The C++ audit measures the same quantities on the same tables.
                # Bounds are set where both implementations sit comfortably;
                # the far tail is deliberately loose because both carry the
                # documented accept-and-guard outlier class (on DD2 the C++'s
                # own worst warm density error is 1.6, larger than this port's).
                @test nwarm_fail <= 5
                @test ncold_fail <= 5
                @test nsplit == 0
                @test p(0.5) < 1e-11
                @test p(0.99) < 1e-7
                @test p(0.999) < 1e-6

                # The projection on real data: invalid states made by pushing
                # valid ones off in D, τ and S. The invariants are promised
                # only where the inner root is unique, which excludes the
                # acausal regions (a) and (b) of CODE.md; there only validity
                # and the fixed point are checked.
                proj = default_projection(v, exp10(v.x_lo))
                nbad = nbad_strict = nfalse = 0
                for _ in 1:2000
                    st = _sample_real_state(v, rng)
                    c = prim2con(v, st.ρ, st.s, st.yₑ, st.w, st.B², st.cos_vB, st.u)
                    r() = exp10(-4 + 8rand(rng))
                    k = rand(rng, 1:4)
                    fD = k in (1, 4) ? r() : 1.0
                    fτ = k in (2, 4) ? r() : 1.0
                    fS = k in (3, 4) ? r() : 1.0
                    cin = Con2PrimIn(fD * c.D, fτ * c.τ, fD * c.D_Y, fS * c.S_par, fS * c.S_perp, c.B²)
                    out = con2prim_safe(v, cin, opts, proj)
                    bad = policy_violations(v, proj, opts, cin, out)
                    strict = filter(in(("non-finite", "invalid", "flags 0 but not the input", "not a fixed point")), bad)
                    nbad_strict += !isempty(strict)
                    # Away from (a), the top of the table, and (b), its hottest
                    # edge at low density.
                    b = out.base
                    sr = srange(v, b.ρ, b.ye)
                    acausal = b.ρ > 1e14 * v.κ || (b.ρ < 1e8 * v.κ && b.s > sr.s_max - 0.05 * (sr.s_max - sr.s_min))
                    acausal || (nbad += !isempty(bad))
                    isempty(bad) || nbad <= 3 && @info "$(tab.name): violated" cin out.policy_flags bad
                    nfalse += check_con_state(v, cin, proj; check_endpoints=true) & ~out.policy_flags != 0
                end
                @info "$(tab.name): projection" nbad nbad_strict nfalse
                @test nbad_strict == 0
                @test nbad == 0
                @test nfalse == 0
            end
        end

        # No two tables are taken for one another, the repaired DD2 included.
        @test allunique(values(fingerprints))

        # Repair is an offline C++ step, but its effect must be visible here:
        # the repaired DD2 has measurably fewer entropy-monotonicity violations
        # than the original.
        if haskey(audits, "DD2") && haskey(audits, "DD2 repaired")
            @info "DD2 σ_u violations: original=$(audits["DD2"]) repaired=$(audits["DD2 repaired"])"
            @test audits["DD2 repaired"] < audits["DD2"]
        end
    end
end

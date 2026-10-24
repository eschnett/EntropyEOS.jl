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

if isempty(TABLE_DIR) || !isdir(TABLE_DIR)
    @info "real-table tests skipped (set ENTROPYEOS_TABLE_DIR to enable)"
else
    @testset "real tables" begin
        E = EntropyEOS
        audits = Dict{String,Int}()

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

        # Repair is an offline C++ step, but its effect must be visible here:
        # the repaired DD2 has measurably fewer entropy-monotonicity violations
        # than the original.
        if haskey(audits, "DD2") && haskey(audits, "DD2 repaired")
            @info "DD2 σ_u violations: original=$(audits["DD2"]) repaired=$(audits["DD2 repaired"])"
            @test audits["DD2 repaired"] < audits["DD2"]
        end
    end
end

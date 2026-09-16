# Tests against real stellarcollapse tables, which are hundreds of megabytes
# and are not in git. Point ENTROPYEOS_TABLE_DIR at a directory holding them;
# the whole file skips cleanly otherwise, so CI needs nothing.
#
# The small cropped LS220 fixture is the useful one here: it carries real
# data's pathologies -- including LS220's three genuine non-finite cs2/gamma
# points -- at 20 MB.

const TABLE_DIR = get(ENV, "ENTROPYEOS_TABLE_DIR", "")

if isempty(TABLE_DIR) || !isdir(TABLE_DIR)
    @info "real-table tests skipped (set ENTROPYEOS_TABLE_DIR to enable)"
else
    @testset "real tables" begin
        E = EntropyEOS
        crop = joinpath(TABLE_DIR, "LS220_san_crop.h5")

        if !isfile(crop)
            @info "LS220_san_crop.h5 not found in ENTROPYEOS_TABLE_DIR; skipping"
        else
            t = read_stellarcollapse(crop)

            @testset "read" begin
                @test size(t) == (61, 136, 16)
                @test has_field(t, "entropy") && has_field(t, "logenergy")
                @test energy_shift(t) > 0
                @test validate_axes(t) === nothing
                @test all(isfinite, field(t, "entropy"))
                @test all(isfinite, field(t, "logenergy"))
                # LS220 genuinely carries a few non-finite points in columns the
                # pipeline never interprets. They must pass through rather than
                # being treated as a broken file.
                @test count(!isfinite, field(t, "cs2")) == 3
                @test count(!isfinite, field(t, "gamma")) == 3
            end

            @testset "check" begin
                rep = check_table(t)
                # Not fatal: the non-finite values are in uninterpreted columns.
                @test rep.status !== Status.fatal
                byname = Dict(c.name => c for c in rep.classes)
                @test haskey(byname, "nonfinite_cs2")
                @test byname["nonfinite_cs2"].count == 3
                @test byname["entropy_negative"].count == 0
                @test sprint(show, MIME"text/plain"(), rep) isa String
            end

            @testset "build and solve" begin
                eos = E.build_eos(t)
                v = EOSTableView(eos)
                @test v.κ <= 1
                @test 0 < v.κ
                # This crop is the ORIGINAL table, not a repaired one, so the
                # audit is expected to find entropy non-monotonicity in T. That
                # the build reports it rather than refusing is the contract.
                @test eos.audit.σ_u.violation_count > 0

                opts = Con2PrimOptions()
                rng = StableRNG(3)
                nfail = 0
                maxρ = 0.0
                for _ in 1:2000
                    ρ = exp10(v.x_lo + (v.x_hi - v.x_lo) * rand(rng))
                    yₑ = v.y_lo + (v.y_hi - v.y_lo) * rand(rng)
                    sr = srange(v, ρ, yₑ)
                    s = sr.s_min + (sr.s_max - sr.s_min) * rand(rng)
                    w = 3rand(rng)
                    B² = rand(rng) < 0.5 ? 0.0 : ρ * 10.0^(-2 + 3rand(rng))
                    pt = evaluate(v, ρ, s, yₑ, NaN)
                    c = prim2con(v, ρ, s, yₑ, w, B², 2rand(rng) - 1, pt.u_solved)
                    cin = Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²)
                    o = con2prim(v, cin, opts, s, w, pt.u_solved)
                    (o.result === E.C2PResult.failed_no_bracket ||
                     o.result === E.C2PResult.failed_max_iter) && (nfail += 1)
                    maxρ = max(maxρ, abs(o.ρ - ρ) / ρ)
                end
                # Measured: every warm state converges in the Newton alone, to
                # 2e-16 in density.
                @test nfail == 0
                @test maxρ < 1e-12

                # And the never-fails layer on the same table.
                pol = default_policy(v, exp10(v.x_lo))
                for bad in (Con2PrimIn(1e14, -1e30, 1e13, 1e30, 0.0, 0.0),
                            Con2PrimIn(NaN, 1.0, 1.0, 0.0, 0.0, 0.0),
                            Con2PrimIn(1e-30, 1.0, 1e-31, 0.0, 0.0, 0.0))
                    sa = con2prim_safe(v, bad, opts, pol)
                    @test sa.policy_flags != 0
                    @test all(isfinite, (sa.base.ρ, sa.base.s, sa.base.w, sa.cons.D, sa.cons.τ))
                    @test check_prim_state(v, PrimState(sa.base.ρ, sa.base.s, sa.base.ye, sa.base.w),
                                           pol) == 0
                end
            end
        end
    end
end

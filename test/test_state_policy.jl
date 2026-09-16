@testset "state policy" begin
    E = EntropyEOS
    tbl = make_synthetic_table()
    eos = E.build_eos(tbl)
    v = EOSTableView(eos)
    opts = Con2PrimOptions()
    pol = default_policy(v, 1e3 * v.κ)

    """
    The acceptance bar for every case below: are the returned primitives valid
    and finite, and are the returned conservatives *exactly solvable* -- does
    the plain solver, re-run on them, converge and reproduce them? The second
    question is the real one, since it is what makes adopting `cons` a safe
    operation rather than a hope.
    """
    function accept(out)
        prim_finite = all(isfinite, (out.base.ρ, out.base.s, out.base.ye, out.base.w, out.base.W))
        cons_finite = all(isfinite, (out.cons.D, out.cons.τ, out.cons.D_Y, out.cons.S_par,
                                     out.cons.S_perp, out.cons.B²))
        ps = PrimState(out.base.ρ, out.base.s, out.base.ye, out.base.w)
        valid = check_prim_state(v, ps, pol) == 0
        cin = Con2PrimIn(out.cons.D, out.cons.τ, out.cons.D_Y, out.cons.S_par, out.cons.S_perp,
                         out.cons.B²)
        re = con2prim(v, cin, opts, out.base.s, out.base.w, out.base.eos.u_solved)
        solved = re.result === E.C2PResult.converged_newton ||
                 re.result === E.C2PResult.converged_fallback
        reproduces = abs(re.ρ - out.base.ρ) / out.base.ρ < 1e-9 &&
                     abs(re.s - out.base.s) / max(abs(out.base.s), 1) < 1e-9
        return prim_finite && cons_finite && valid && solved && reproduces
    end

    @testset "derived bounds" begin
        @test pol.ρ_ceiling ≈ exp10(v.x_hi)
        @test pol.w_cap ≈ acosh(100)
        # The cap must stay below the solver's own rapidity bound, or it is
        # silently inoperative.
        @test pol.w_cap < opts.w_max
        @test pol.D_max ≈ pol.ρ_ceiling * cosh(pol.w_cap)
        @test pol.τ_max > 0 && isfinite(pol.τ_max)
        # NaN sentinels must survive: they mean "derive", and replacing them
        # with a Union type would break the GPU path.
        @test isnan(pol.s_atm)
        @test isnan(pol.ye_atm)
        @test isnan(PolicyOptions{Float64}().s_atm)
    end

    @testset "no false positives on valid states" begin
        rng = StableRNG(5)
        touched = 0
        for _ in 1:500
            ρ = exp10(v.x_lo + (v.x_hi - v.x_lo) * rand(rng))
            yₑ = v.y_lo + (v.y_hi - v.y_lo) * rand(rng)
            sr = srange(v, ρ, yₑ)
            s = sr.s_min + (0.1 + 0.8rand(rng)) * (sr.s_max - sr.s_min)
            c = prim2con(v, ρ, s, yₑ, 2rand(rng), 0.0, 0.0, NaN)
            cin = Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²)
            out = con2prim_safe(v, cin, opts, pol)
            out.policy_flags != 0 && (touched += 1)
            if out.policy_flags == 0
                # An untouched state must hand back the input conservatives
                # bit-identically, not a round trip through prim2con.
                @test out.cons.D === cin.D
                @test out.cons.τ === cin.τ
                @test out.cons.D_Y === cin.D_Y
                @test out.cons.S_par === cin.S_par
                @test out.cons.S_perp === cin.S_perp
                @test out.solved
            end
        end
        @test touched == 0
    end

    # A known-good conserved state to corrupt in various ways.
    ρ0 = exp10(0.5 * (v.x_lo + v.x_hi))
    yₑ0 = 0.35
    sr0 = srange(v, ρ0, yₑ0)
    s0 = 0.5 * (sr0.s_min + sr0.s_max)
    g = let c = prim2con(v, ρ0, s0, yₑ0, 0.8, 0.0, 0.0, NaN)
        Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²)
    end

    @testset "non-finite input is excised" begin
        for cin in (Con2PrimIn(g.D, NaN, g.D_Y, g.S_par, g.S_perp, g.B²),
                    Con2PrimIn(g.D, g.τ, g.D_Y, Inf, g.S_perp, g.B²),
                    Con2PrimIn(NaN, g.τ, g.D_Y, g.S_par, g.S_perp, g.B²),
                    Con2PrimIn(-1.0, g.τ, g.D_Y, g.S_par, g.S_perp, g.B²),
                    Con2PrimIn(g.D, g.τ, g.D_Y, g.S_par, g.S_perp, -1.0))
            out = con2prim_safe(v, cin, opts, pol)
            @test out.policy_flags & FLAG_POL_NONFINITE != 0
            @test out.policy_flags & FLAG_POL_ATMOSPHERE != 0
            @test !out.solved            # no solve was attempted
            @test accept(out)
        end
    end

    @testset "vacuum is excised to the atmosphere" begin
        cin = Con2PrimIn(1e-8, g.τ, g.D_Y * 1e-18, 0.0, 0.0, 0.0)
        out = con2prim_safe(v, cin, opts, pol)
        @test out.policy_flags & FLAG_POL_ATMOSPHERE != 0
        @test out.base.ρ ≈ E.pol_ρ_atm(v, pol)
        @test out.base.w == 0
        @test accept(out)
    end

    @testset "collapse ceilings" begin
        for cin in (Con2PrimIn(1e30, g.τ, 0.35e30, g.S_par, g.S_perp, g.B²),
                    Con2PrimIn(g.D, 1e30, g.D_Y, g.S_par, g.S_perp, g.B²))
            out = con2prim_safe(v, cin, opts, pol)
            @test out.policy_flags & FLAG_POL_CEILING != 0
            @test accept(out)
        end
        # With the option set, a collapse state is excised rather than projected.
        pol_atm = E.policy_derive_bounds(v,
            PolicyOptions(; ρ_atm=pol.ρ_atm, ρ_ceiling=pol.ρ_ceiling, w_cap=pol.w_cap,
                          collapse_to_atmosphere=true))
        out = con2prim_safe(v, Con2PrimIn(1e30, g.τ, 0.35e30, g.S_par, g.S_perp, g.B²), opts, pol_atm)
        @test out.policy_flags & FLAG_POL_ATMOSPHERE != 0
        @test out.policy_flags & FLAG_POL_CEILING != 0
    end

    @testset "energy below the coldest expressible state" begin
        # τ shrunk far below anything the table can represent at this momentum:
        # the entropy is floored onto the physical minimum rather than the
        # state being thrown away.
        cin = Con2PrimIn(g.D, g.τ * 1e-10, g.D_Y, g.S_par, g.S_perp, g.B²)
        out = con2prim_safe(v, cin, opts, pol)
        @test out.policy_flags & FLAG_POL_S_FLOORED != 0
        @test out.base.s ≈ srange(v, out.base.ρ, out.base.ye).s_min rtol = 1e-9
        @test accept(out)
    end

    @testset "superluminal momentum demand" begin
        out = con2prim_safe(v, Con2PrimIn(g.D, g.τ, g.D_Y, 1e30, 0.0, g.B²), opts, pol)
        @test out.policy_flags != 0
        @test accept(out)
    end

    @testset "rapidity cap preserves D" begin
        # A state solved above the cap is capped, and the density is recomputed
        # so that D itself is preserved exactly -- it is the best-conditioned
        # conservative and the only one recoverable in closed form.
        tight = E.policy_derive_bounds(v, PolicyOptions(; ρ_atm=pol.ρ_atm, ρ_ceiling=pol.ρ_ceiling,
                                                        w_cap=0.5))
        c = prim2con(v, ρ0, s0, yₑ0, 2.0, 0.0, 0.0, NaN)
        cin = Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²)
        out = con2prim_safe(v, cin, opts, tight)
        @test out.policy_flags & FLAG_POL_W_CAPPED != 0
        @test out.base.w ≈ 0.5
        @test out.base.ρ ≈ cin.D / cosh(0.5) rtol = 1e-14
        @test out.cons.D ≈ cin.D rtol = 1e-12
    end

    @testset "check and project agree, and projection is idempotent" begin
        rng = StableRNG(31)
        for _ in 1:200
            ρ = exp10(v.x_lo - 1 + (v.x_hi - v.x_lo + 2) * rand(rng))
            yₑ = v.y_lo - 0.05 + (v.y_hi - v.y_lo + 0.1) * rand(rng)
            sr = srange(v, clamp(ρ, exp10(v.x_lo), exp10(v.x_hi)), clamp(yₑ, v.y_lo, v.y_hi))
            s = sr.s_min - 1 + (sr.s_max - sr.s_min + 2) * rand(rng)
            w = -0.2 + 7rand(rng)
            ps = PrimState(ρ, s, yₑ, w)

            flags = check_prim_state(v, ps, pol)
            ps2, flags2 = project_prim_state(v, ps, pol)
            @test flags == flags2                      # checking and projecting agree
            @test check_prim_state(v, ps2, pol) == 0   # the result is valid
            # Idempotent, and bitwise so: a projection that only approximately
            # repeats itself has a bug.
            ps3, flags3 = project_prim_state(v, ps2, pol)
            @test flags3 == 0
            @test ps3.ρ === ps2.ρ
            @test ps3.s === ps2.s
            @test ps3.ye === ps2.ye
            @test ps3.w === ps2.w
        end
    end

    @testset "projection flags each violation exactly" begin
        ρmid = exp10(0.5 * (v.x_lo + v.x_hi))
        srm = srange(v, ρmid, 0.35)
        smid = 0.5 * (srm.s_min + srm.s_max)
        @test check_prim_state(v, PrimState(ρmid, smid, 0.35, 0.5), pol) == 0
        @test check_prim_state(v, PrimState(ρmid * 1e9, smid, 0.35, 0.5), pol) & FLAG_POL_ρ_CLAMPED != 0
        @test check_prim_state(v, PrimState(ρmid, smid, v.y_hi + 0.2, 0.5), pol) & FLAG_POL_YE_CLAMPED != 0
        @test check_prim_state(v, PrimState(ρmid, srm.s_min - 5, 0.35, 0.5), pol) & FLAG_POL_S_FLOORED != 0
        @test check_prim_state(v, PrimState(ρmid, srm.s_max + 5, 0.35, 0.5), pol) & FLAG_POL_S_CEILED != 0
        @test check_prim_state(v, PrimState(ρmid, smid, 0.35, 99.0), pol) & FLAG_POL_W_CAPPED != 0
        @test check_prim_state(v, PrimState(NaN, smid, 0.35, 0.5), pol) & FLAG_POL_NONFINITE != 0
        # Below the atmosphere the whole state is replaced.
        @test check_prim_state(v, PrimState(1e-3, smid, 0.35, 0.5), pol) & FLAG_POL_ATMOSPHERE != 0
        # The atmosphere itself must check clean, which is why the trigger is
        # not applied here.
        atm = policy_atmosphere(v, pol, 0.35)
        @test check_prim_state(v, atm, pol) == 0
    end

    @testset "check_con_state" begin
        # Pure arithmetic by default: it must agree with con2prim_safe's own
        # step-0 and step-1 verdicts without solving anything.
        @test check_con_state(v, g, pol) == 0
        @test check_con_state(v, Con2PrimIn(g.D, NaN, g.D_Y, g.S_par, g.S_perp, g.B²), pol) &
              FLAG_POL_NONFINITE != 0
        @test check_con_state(v, Con2PrimIn(1e-8, g.τ, 1e-9, 0.0, 0.0, 0.0), pol) &
              FLAG_POL_ATMOSPHERE != 0
        @test check_con_state(v, Con2PrimIn(1e30, g.τ, 0.35e30, g.S_par, g.S_perp, g.B²), pol) &
              FLAG_POL_CEILING != 0
        # With endpoint checking it also detects an energy outside the table.
        @test check_con_state(v, Con2PrimIn(g.D, g.τ * 1e-10, g.D_Y, g.S_par, g.S_perp, g.B²), pol;
                              check_endpoints=true) != 0
    end
end

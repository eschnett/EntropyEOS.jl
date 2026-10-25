@testset "state policy" begin
    E = EntropyEOS
    tbl = make_synthetic_table()
    eos = E.build_eos(tbl)
    v = EOSTableView(eos)
    opts = Con2PrimOptions()
    proj = default_projection(v, 1e3 * v.κ)
    pol = PolicyOptions()
    ρ_min, ρ_max, w_cap = E.proj_ρ_min(v, proj), E.proj_ρ_max(v, proj), E.proj_w_cap(proj)
    δ = E.face_slack(Float64)

    cin_of(c) = Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²)
    Smag(c) = hypot(c.S_par, c.S_perp)
    has(f, bits) = f & bits != 0

    violations(cin, out) = policy_violations(v, proj, opts, cin, out)
    # Fed back without a guess, an output is still a fixed point where the
    # re-solve is well conditioned; the slack is sized for that.
    cold_fixed(out) = con2prim_safe(v, cin_of(out.cons), opts, proj).policy_flags == 0

    function valid_state(rng; w_max=2.0, magnetized=true)
        ρ = exp10(v.x_lo + (v.x_hi - v.x_lo) * rand(rng))
        yₑ = v.y_lo + (v.y_hi - v.y_lo) * rand(rng)
        sr = srange(v, ρ, yₑ)
        s = sr.s_min + (0.1 + 0.8rand(rng)) * (sr.s_max - sr.s_min)
        h = E.evaluate(v, ρ, s, yₑ, NaN).h
        B² = magnetized && rand(rng) < 0.5 ? ρ * h * exp10(-3 + 4rand(rng)) : 0.0
        return cin_of(prim2con(v, ρ, s, yₑ, w_max * rand(rng), B², 2rand(rng) - 1, NaN))
    end

    @testset "options" begin
        @test proj.ρ_max ≈ exp10(v.x_hi)
        @test proj.w_cap ≈ acosh(100)
        # The cap must stay below the solver's own rapidity bound, or it is
        # silently inoperative.
        @test proj.w_cap < opts.w_max
        # Unset fields fall back to the table's box.
        p0 = ProjectionOptions()
        @test E.proj_ρ_min(v, p0) ≈ exp10(v.x_lo)
        @test E.proj_ρ_max(v, p0) ≈ exp10(v.x_hi)
        @test E.proj_w_cap(p0) ≈ acosh(100)
        # NaN sentinels must survive: replacing them with a Union type would
        # break the GPU path.
        @test isnan(pol.s_atm)
        @test isnan(pol.ye_atm)
        @test pol.atm_trigger == 1.01
        @test !pol.collapse_to_atmosphere
    end

    @testset "valid states are untouched" begin
        rng = StableRNG(5)
        touched = 0
        reported = 0
        for _ in 1:500
            cin = valid_state(rng)
            out = con2prim_safe(v, cin, opts, proj)
            touched += out.policy_flags != 0
            reported += check_con_state(v, cin, proj; check_endpoints=true) != 0
            if out.policy_flags == 0
                # An untouched state hands back the input conservatives
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
        @test reported == 0

        # States exactly on an edge of the entropy window, as con2prim_safe
        # itself writes them onto an s face. They are valid, to the slack, and
        # check_con_state must not report them. A state at rest is where the
        # energy alone decides, the hardest case for the check.
        touched = 0
        nfalse = 0
        for _ in 1:400
            ρ = exp10(v.x_lo + (v.x_hi - v.x_lo) * (0.05 + 0.9rand(rng)))
            yₑ = v.y_lo + (v.y_hi - v.y_lo) * rand(rng)
            sr = srange(v, ρ, yₑ)
            s = rand(rng, Bool) ? sr.s_min : sr.s_max
            w = rand(rng, Bool) ? 0.0 : 2rand(rng)
            h = E.evaluate(v, ρ, s, yₑ, NaN).h
            B² = rand(rng) < 0.5 ? ρ * h * exp10(-3 + 4rand(rng)) : 0.0
            cin = cin_of(prim2con(v, ρ, s, yₑ, w, B², 2rand(rng) - 1, NaN))
            out = con2prim_safe(v, cin, opts, proj)
            touched += out.policy_flags != 0
            nfalse += check_con_state(v, cin, proj; check_endpoints=true) & ~out.policy_flags != 0
        end
        @test touched == 0
        @test nfalse == 0

        # At Float32 the energy residual is noise well above the solver's
        # tolerance, and the check must still not report what con2prim_safe
        # does not set; con2prim_safe itself may then project.
        v32 = E.narrow(v, Float32)
        o32 = Con2PrimOptions{Float32}()
        p32 = default_projection(v32, Float32(1e3 * v.κ))
        nfalse = 0
        for _ in 1:400
            ρ = Float32(exp10(v.x_lo + (v.x_hi - v.x_lo) * (0.05 + 0.9rand(rng))))
            yₑ = Float32(v.y_lo + (v.y_hi - v.y_lo) * rand(rng))
            sr = srange(v32, ρ, yₑ)
            s = rand(rng, Bool) ? sr.s_min : sr.s_max
            c = prim2con(v32, ρ, s, yₑ, Float32(2rand(rng)), 0.0f0, 0.0f0, NaN32)
            cin = cin_of(c)
            f = con2prim_safe(v32, cin, o32, p32).policy_flags
            nfalse += check_con_state(v32, cin, p32; check_endpoints=true, solver=o32) & ~f != 0
        end
        @test nfalse == 0
    end

    # A known-good conserved state to drive onto each face.
    ρ0 = exp10(0.5 * (v.x_lo + v.x_hi))
    yₑ0 = 0.35
    sr0 = srange(v, ρ0, yₑ0)
    s0 = 0.5 * (sr0.s_min + sr0.s_max)
    g = cin_of(prim2con(v, ρ0, s0, yₑ0, 0.8, 0.0, 0.0, NaN))
    with(c; D=c.D, τ=c.τ, D_Y=c.D_Y, S_par=c.S_par, S_perp=c.S_perp, B²=c.B²) =
        Con2PrimIn(D, τ, D_Y, S_par, S_perp, B²)

    @testset "each face" begin
        # Untouched bit for bit, including the sign of a zero field.
        cin = with(g; B²=-0.0)
        out = con2prim_safe(v, cin, opts, proj)
        @test out.policy_flags == 0
        @test out.cons.B² === -0.0

        # Too hot for the table, in a thin cell: the energy is lowered onto
        # s_max and the cell stays where it is -- EOS-9's magnetized tubes.
        thin = cin_of(prim2con(v, 3ρ_min, srange(v, 3ρ_min, yₑ0).s_max, yₑ0, 0.5, 0.0, 0.0, NaN))
        cin = with(thin; τ=1e3thin.τ)
        out = con2prim_safe(v, cin, opts, proj)
        @test out.policy_flags == FLAG_POL_S_CEILED
        @test out.cons.D ≈ cin.D rtol = 1e-12
        @test isempty(violations(cin, out))
        @test cold_fixed(out)

        # Too cold: the energy is raised onto s_min.
        cin = with(g; τ=1e-10g.τ)
        out = con2prim_safe(v, cin, opts, proj)
        @test out.policy_flags == FLAG_POL_S_FLOORED
        @test isempty(violations(cin, out))
        @test cold_fixed(out)

        # A cold state at W = 1000, capped to W = 100 with τ kept: the kinetic
        # energy taken out of S heats the cell, onto s_max.
        sr = srange(v, ρ0, yₑ0)
        cin = cin_of(prim2con(v, ρ0, sr.s_min + 0.01 * (sr.s_max - sr.s_min), yₑ0, acosh(1000.0), 0.0, 0.0, NaN))
        out = con2prim_safe(v, cin, opts, proj)
        @test has(out.policy_flags, FLAG_POL_W_CEILED)
        @test out.base.w == w_cap
        @test has(out.policy_flags, FLAG_POL_S_CEILED) || isapprox(out.cons.τ, cin.τ; rtol=1e-10)
        @test isempty(violations(cin, out))
        @test cold_fixed(out)

        # A superluminal momentum demand.
        cin = with(g; S_par=1e30)
        out = con2prim_safe(v, cin, opts, proj)
        @test has(out.policy_flags, FLAG_POL_W_CEILED)
        @test isempty(violations(cin, out))
        @test cold_fixed(out)

        # Below ρ_min: at rest at ρ_min, the least mass added.
        c = prim2con(v, 0.5ρ_min, srange(v, 0.5ρ_min, yₑ0).s_min, yₑ0, 0.3, 0.0, 0.0, NaN)
        cin = cin_of(c)
        out = con2prim_safe(v, cin, opts, proj)
        @test has(out.policy_flags, FLAG_POL_ρ_FLOORED)
        @test out.base.w == 0
        @test out.cons.D == ρ_min
        @test isempty(violations(cin, out))
        @test cold_fixed(out)

        # Above ρ_min but too fast for it: slowed down until ρ = ρ_min, with D
        # kept and no mass added.
        sr = srange(v, ρ_min, yₑ0)
        c = prim2con(v, 0.5ρ_min, sr.s_min + 0.5 * (sr.s_max - sr.s_min), yₑ0, 2.0, 0.0, 0.0, NaN)
        cin = cin_of(c)
        out = con2prim_safe(v, cin, opts, proj)
        @test has(out.policy_flags, FLAG_POL_ρ_FLOORED)
        @test out.base.w ≈ acosh(cin.D / ρ_min) rtol = 1e-12
        @test out.cons.D ≈ cin.D rtol = 1e-12
        @test isempty(violations(cin, out))
        @test cold_fixed(out)

        # Above ρ_max: D lowered, S and τ kept where the window allows. Both the
        # path decided by D alone and the one through the diagnosis.
        c = cin_of(prim2con(v, ρ_max, s0, yₑ0, 0.0, 0.0, 0.0, NaN))
        for cin in (with(c; D=3c.D, D_Y=3c.D_Y), with(c; D=1e3 * cosh(w_cap) * c.D, D_Y=1e3 * cosh(w_cap) * c.D_Y),
                    with(g; D=1e30, D_Y=0.35e30))
            out = con2prim_safe(v, cin, opts, proj)
            @test has(out.policy_flags, FLAG_POL_ρ_CEILED)
            @test out.cons.D < cin.D
            @test isempty(violations(cin, out))
            # Not `cold_fixed`: the collapse state lands on s_min at ρ_max,
            # nearly at rest, where the energy barely depends on s and a cold
            # re-solve cannot place s to the slack (study/cold_resolve.jl).
        end

        # Yₑ outside the table: D_Y follows Yₑ·D, nothing else changes.
        for (DY, bit, y) in ((1.5v.y_hi * g.D, FLAG_POL_YE_CEILED, v.y_hi), (0.0, FLAG_POL_YE_FLOORED, v.y_lo))
            cin = with(g; D_Y=DY)
            out = con2prim_safe(v, cin, opts, proj)
            @test out.policy_flags == bit
            @test out.base.ye == y
            @test out.cons.D_Y == y * cin.D
            @test out.cons.τ === cin.τ
            @test isempty(violations(cin, out))
            @test cold_fixed(out)
        end
    end

    @testset "continuity at D = ρ_min" begin
        # A state just above ρ_min is slowed almost to rest; one just below is
        # put at rest. Both meet at a state at rest at ρ_min.
        sr = srange(v, ρ_min, yₑ0)
        c = cin_of(prim2con(v, ρ_min, sr.s_min + 0.5 * (sr.s_max - sr.s_min), yₑ0, 0.5, 0.0, 0.0, NaN))
        lo = con2prim_safe(v, with(c; D=ρ_min * (1 - 1e-10), D_Y=yₑ0 * ρ_min * (1 - 1e-10)), opts, proj)
        hi = con2prim_safe(v, with(c; D=ρ_min * (1 + 1e-10), D_Y=yₑ0 * ρ_min * (1 + 1e-10)), opts, proj)
        @test has(lo.policy_flags, FLAG_POL_ρ_FLOORED)
        @test has(hi.policy_flags, FLAG_POL_ρ_FLOORED)
        @test lo.base.w == 0
        @test hi.base.w < 1e-4
        @test hi.cons.τ ≈ lo.cons.τ rtol = 1e-6
        @test hi.base.s ≈ lo.base.s rtol = 1e-6
        @test Smag(hi.cons) < 1e-4 * Smag(c)
    end

    @testset "random invalid states" begin
        # Valid states pushed off in D, τ, S and Yₑ by up to four decades each.
        rng = StableRNG(23)
        nbad = 0
        nfalse = 0
        nsplit = ndone = 0
        seen = UInt32(0)
        for _ in 1:3000
            c = valid_state(rng; w_max=6.0)
            k = rand(rng, 1:6)
            r() = exp10(-4 + 8rand(rng))
            fD = k in (3, 5, 6) ? r() : 1.0
            fτ = k in (1, 5, 6) ? r() : 1.0
            fS = k in (2, 6) ? r() : 1.0
            fY = k == 4 ? 0.5 + rand(rng) : 1.0
            cin = Con2PrimIn(fD * c.D, fτ * c.τ, fY * fD * c.D_Y, fS * c.S_par, fS * c.S_perp, c.B²)
            out = con2prim_safe(v, cin, opts, proj)
            seen |= out.policy_flags
            # Finishing any first stage, done or not, gives the fused result.
            first, st = con2prim_safe_start(v, cin, opts, proj)
            nsplit += con2prim_safe_finish(v, cin, opts, proj, st) !== out || (is_done(st) && first !== out)
            ndone += is_done(st)
            bad = violations(cin, out)
            isempty(bad) || (nbad += 1; nbad <= 5 && @info "violated" cin out.policy_flags bad)
            # check_con_state never reports a face con2prim_safe does not set.
            chk = check_con_state(v, cin, proj; check_endpoints=true)
            nfalse += chk & ~out.policy_flags != 0
        end
        @test nbad == 0
        @test nfalse == 0
        @test nsplit == 0
        @test 0 < ndone < 3000
        # Every face was exercised.
        for bit in (FLAG_POL_ρ_FLOORED, FLAG_POL_ρ_CEILED, FLAG_POL_YE_FLOORED, FLAG_POL_YE_CEILED,
                    FLAG_POL_S_FLOORED, FLAG_POL_S_CEILED, FLAG_POL_W_CEILED)
            @test has(seen, bit)
        end
        @test !has(seen, FLAG_POL_ATMOSPHERE)
    end

    @testset "non-finite input" begin
        fx = E.fixed_state(v, proj)
        @test check_prim_state(v, fx, proj) == 0
        @test fx.s == srange(v, fx.ρ, fx.ye).s_min
        for cin in (with(g; τ=NaN), with(g; S_par=Inf), with(g; D=NaN), with(g; D=-1.0), with(g; B²=-1.0))
            out = con2prim_safe(v, cin, opts, proj)
            _, st = con2prim_safe_start(v, cin, opts, proj)
            @test !is_done(st)
            @test con2prim_safe_finish(v, cin, opts, proj, st) === out
            @test out.policy_flags == FLAG_POL_NONFINITE
            @test !out.solved            # no solve was attempted
            @test (out.base.ρ, out.base.s, out.base.ye, out.base.w) == (fx.ρ, fx.s, fx.ye, fx.w)
            @test isempty(violations(cin, out))
        end
    end

    @testset "NOT_CONVERGED" begin
        # A solver starved of iterations fails on a valid state; the diagnosis
        # then finds it inside the range, so there is nothing to project onto.
        # The seed is starved too: with all its passes, it alone solves these
        # states to the tolerance.
        starved = Con2PrimOptions(; max_iter_newton=0, max_iter_1d=1, seed_passes=1)
        rng = StableRNG(41)
        n = 0
        for _ in 1:50
            cin = valid_state(rng)
            base = con2prim(v, cin, starved)
            base.result in (E.C2PResult.converged_newton, E.C2PResult.converged_fallback) && continue
            out = con2prim_safe(v, cin, starved, proj)
            has(out.policy_flags, FLAG_POL_NOT_CONVERGED) || continue
            n += 1
            fx = E.fixed_state(v, proj, cin.D_Y / cin.D)
            @test out.policy_flags == FLAG_POL_NOT_CONVERGED
            @test out.solved
            @test (out.base.ρ, out.base.s, out.base.ye, out.base.w) == (fx.ρ, fx.s, fx.ye, fx.w)
            @test out.base.result === base.result
            # With Yₑ out of range the fixed state is on the flagged Yₑ face.
            hi = with(cin; D_Y=1.5v.y_hi * cin.D)
            out = con2prim_safe(v, hi, starved, proj)
            if has(out.policy_flags, FLAG_POL_NOT_CONVERGED)
                @test out.policy_flags == FLAG_POL_NOT_CONVERGED | FLAG_POL_YE_CEILED
                @test out.base.ye == v.y_hi
            end
        end
        @test n > 0
    end

    @testset "check_con_state" begin
        # Pure arithmetic by default.
        @test check_con_state(v, g, proj) == 0
        @test check_con_state(v, with(g; τ=NaN), proj) == FLAG_POL_NONFINITE
        @test check_con_state(v, with(g; D=0.5ρ_min, D_Y=0.5yₑ0 * ρ_min), proj) == FLAG_POL_ρ_FLOORED
        @test check_con_state(v, with(g; D=1e30, D_Y=0.35e30), proj) == FLAG_POL_ρ_CEILED
        @test check_con_state(v, with(g; D_Y=0.0), proj) == FLAG_POL_YE_FLOORED
        # With endpoint checking it also finds an energy outside the table.
        @test check_con_state(v, with(g; τ=1e-10g.τ), proj) == 0
        @test check_con_state(v, with(g; τ=1e-10g.τ), proj; check_endpoints=true) == FLAG_POL_S_FLOORED
    end

    @testset "apply_policy" begin
        level2(cin, p=pol) = apply_policy(v, cin, con2prim_safe(v, cin, opts, proj), proj, p)

        # A valid state, and a projection that needs no physics, pass through.
        out = con2prim_safe(v, g, opts, proj)
        @test apply_policy(v, g, out, proj, pol) === out
        cin = with(g; τ=1e-10g.τ)
        out = con2prim_safe(v, cin, opts, proj)
        @test apply_policy(v, cin, out, proj, pol) === out

        # The atmosphere, with level 1's bits saying why.
        thin = cin_of(prim2con(v, 0.5ρ_min, srange(v, 0.5ρ_min, yₑ0).s_min, yₑ0, 0.3, 0.0, 0.0, NaN))
        for (cin, why) in ((thin, FLAG_POL_ρ_FLOORED), (with(g; τ=NaN), FLAG_POL_NONFINITE),
                           (with(g; D=1.005ρ_min, D_Y=1.005yₑ0 * ρ_min), UInt32(0)))
            out = level2(cin)
            @test has(out.policy_flags, FLAG_POL_ATMOSPHERE)
            @test out.policy_flags & why == why
            @test out.base.ρ == ρ_min
            @test out.base.w == 0
            @test out.base.s == srange(v, ρ_min, out.base.ye).s_min
            @test check_prim_state(v, PrimState(out.base.ρ, out.base.s, out.base.ye, out.base.w), proj) == 0
        end
        # Yₑ is the incoming state's unless the policy pins it, and s_atm is
        # honoured inside the window.
        @test level2(thin).base.ye ≈ yₑ0
        sr = srange(v, ρ_min, 0.3)
        out = level2(thin, PolicyOptions(; s_atm=0.5 * (sr.s_min + sr.s_max), ye_atm=0.3))
        @test out.base.ye == 0.3
        @test out.base.s == 0.5 * (sr.s_min + sr.s_max)

        # Collapse goes to the atmosphere only when asked to.
        cin = with(g; D=1e30, D_Y=0.35e30)
        @test !has(level2(cin).policy_flags, FLAG_POL_ATMOSPHERE)
        out = level2(cin, PolicyOptions(; collapse_to_atmosphere=true))
        @test has(out.policy_flags, FLAG_POL_ATMOSPHERE)
        @test has(out.policy_flags, FLAG_POL_ρ_CEILED)
    end

    @testset "primitive states" begin
        rng = StableRNG(31)
        for _ in 1:200
            ρ = exp10(v.x_lo - 1 + (v.x_hi - v.x_lo + 2) * rand(rng))
            yₑ = v.y_lo - 0.05 + (v.y_hi - v.y_lo + 0.1) * rand(rng)
            sr = srange(v, clamp(ρ, exp10(v.x_lo), exp10(v.x_hi)), clamp(yₑ, v.y_lo, v.y_hi))
            s = sr.s_min - 1 + (sr.s_max - sr.s_min + 2) * rand(rng)
            w = 7rand(rng)
            ps = PrimState(ρ, s, yₑ, w)

            flags = check_prim_state(v, ps, proj)
            ps2, flags2 = project_prim_state(v, ps, proj)
            @test flags == flags2                       # checking and projecting agree
            @test check_prim_state(v, ps2, proj) == 0   # the result is valid
            @test inside_window(v, ps2.ρ, ps2.s, ps2.ye)
            # Idempotent, and bitwise so: a projection that only approximately
            # repeats itself has a bug.
            ps3, flags3 = project_prim_state(v, ps2, proj)
            @test flags3 == 0
            @test ps3 === ps2
        end

        ρmid = exp10(0.5 * (v.x_lo + v.x_hi))
        srm = srange(v, ρmid, 0.35)
        smid = 0.5 * (srm.s_min + srm.s_max)
        chk(ρ, s, y, w) = check_prim_state(v, PrimState(ρ, s, y, w), proj)
        @test chk(ρmid, smid, 0.35, 0.5) == 0
        @test chk(ρmid * 1e9, smid, 0.35, 0.5) & (FLAG_POL_ρ_FLOORED | FLAG_POL_ρ_CEILED) == FLAG_POL_ρ_CEILED
        @test chk(0.5ρ_min, smid, 0.35, 0.5) & (FLAG_POL_ρ_FLOORED | FLAG_POL_ρ_CEILED) == FLAG_POL_ρ_FLOORED
        # The window moves with Yₑ, so s may leave it too.
        ye_bits = FLAG_POL_YE_FLOORED | FLAG_POL_YE_CEILED
        @test chk(ρmid, smid, v.y_hi + 0.2, 0.5) & ye_bits == FLAG_POL_YE_CEILED
        @test chk(ρmid, smid, v.y_lo - 0.2, 0.5) & ye_bits == FLAG_POL_YE_FLOORED
        @test chk(ρmid, srm.s_min - 5, 0.35, 0.5) == FLAG_POL_S_FLOORED
        @test chk(ρmid, srm.s_max + 5, 0.35, 0.5) == FLAG_POL_S_CEILED
        @test chk(ρmid, smid, 0.35, 99.0) == FLAG_POL_W_CEILED
        # No closest valid state: garbage, a non-positive density, a negative
        # rapidity (w is a magnitude).
        @test chk(NaN, smid, 0.35, 0.5) == FLAG_POL_NONFINITE
        @test chk(0.0, smid, 0.35, 0.5) == FLAG_POL_NONFINITE
        @test chk(ρmid, smid, 0.35, -0.1) == FLAG_POL_NONFINITE
    end

    @testset "inside_window" begin
        ρm = exp10(0.5 * (v.x_lo + v.x_hi))
        srm = srange(v, ρm, 0.35)
        slack = δ * (srm.s_max - srm.s_min)
        smid = 0.5 * (srm.s_min + srm.s_max)

        # It is check_prim_state's Yₑ and s tests, at any density: with the ρ
        # and w faces opened, the two agree, also within a few slacks of an edge.
        open_proj = ProjectionOptions(; ρ_min=floatmin(), ρ_max=floatmax(), w_cap=floatmax())
        rng = StableRNG(37)
        for _ in 1:500
            ρ = exp10(v.x_lo - 1 + (v.x_hi - v.x_lo + 2) * rand(rng))
            yₑ = v.y_lo - 0.05 + (v.y_hi - v.y_lo + 0.1) * rand(rng)
            sr = srange(v, ρ, yₑ)
            width = sr.s_max - sr.s_min
            s = rand(rng) < 0.5 ? sr.s_min - 1 + (width + 2) * rand(rng) :
                (rand(rng) < 0.5 ? sr.s_min : sr.s_max) + (4rand(rng) - 2) * δ * width
            @test inside_window(v, ρ, s, yₑ) == (check_prim_state(v, PrimState(ρ, s, yₑ, 0.0), open_proj) == 0)
        end

        # The band: half a slack past an edge is inside, two are not, and Yₑ
        # may be one ulp out of range, as Yₑ·D/D can come back.
        @test inside_window(v, ρm, smid, 0.35)
        @test inside_window(v, ρm, srm.s_min - slack / 2, 0.35)
        @test !inside_window(v, ρm, srm.s_min - 2slack, 0.35)
        @test inside_window(v, ρm, srm.s_max + slack / 2, 0.35)
        @test !inside_window(v, ρm, srm.s_max + 2slack, 0.35)
        for (y_in, y_out) in ((nextfloat(v.y_hi), nextfloat(v.y_hi, 2)), (prevfloat(v.y_lo), prevfloat(v.y_lo, 2)))
            sr = srange(v, ρm, y_in)     # the window moves with Yₑ
            s = 0.5 * (sr.s_min + sr.s_max)
            @test inside_window(v, ρm, s, y_in)
            @test !inside_window(v, ρm, s, y_out)
        end
        # ρ is not tested against a range ...
        for ρ in (1e-3ρ_min, 1e3ρ_max)
            sr = srange(v, ρ, 0.35)
            @test inside_window(v, ρ, 0.5 * (sr.s_min + sr.s_max), 0.35)
        end
        # ... but there is no window at a non-positive or non-finite density.
        for (ρ, s, yₑ) in ((NaN, smid, 0.35), (Inf, smid, 0.35), (0.0, smid, 0.35), (-ρm, smid, 0.35),
                           (ρm, NaN, 0.35), (ρm, Inf, 0.35), (ρm, smid, NaN))
            @test !inside_window(v, ρ, s, yₑ)
        end

        # What it is for: a state within the slack of an s face comes back from
        # con2prim_safe unflagged and outside the exact window, and passes.
        for (s0, outside) in ((srm.s_min - slack / 2, s -> s < srm.s_min), (srm.s_max + slack / 2, s -> s > srm.s_max))
            out = con2prim_safe(v, cin_of(prim2con(v, ρm, s0, 0.35, 0.4, 0.0, 1.0, NaN)), opts, proj)
            @test out.policy_flags == 0
            @test outside(out.base.s)
            @test inside_window(v, out.base.ρ, out.base.s, out.base.ye)
        end
    end
end

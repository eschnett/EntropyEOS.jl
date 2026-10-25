_c2p_probe(v, cin, opts, s, w, u) = EntropyEOS.con2prim(v, cin, opts, s, w, u).ρ
function _c2p_allocs(v, cin, opts, s, w, u)
    _c2p_probe(v, cin, opts, s, w, u)
    return @allocated _c2p_probe(v, cin, opts, s, w, u)
end
_c2p_start_probe(v, cin, opts) = EntropyEOS.con2prim_start(v, cin, opts)[2].s
function _c2p_start_allocs(v, cin, opts)
    _c2p_start_probe(v, cin, opts)
    return @allocated _c2p_start_probe(v, cin, opts)
end
_c2p_finish_probe(v, cin, opts, st) = EntropyEOS.con2prim_finish(v, cin, opts, st).ρ
function _c2p_finish_allocs(v, cin, opts, st)
    _c2p_finish_probe(v, cin, opts, st)
    return @allocated _c2p_finish_probe(v, cin, opts, st)
end

@testset "con2prim" begin
    E = EntropyEOS
    tbl = make_synthetic_table()
    eos = E.build_eos(tbl)
    v = EOSTableView(eos)
    opts = Con2PrimOptions()

    # Build a conserved state from primitives, so the answer is known exactly.
    function forward(ρ, s, yₑ, w, B², cos_vB)
        pt = evaluate(v, ρ, s, yₑ, NaN)
        c = prim2con(v, ρ, s, yₑ, w, B², cos_vB, pt.u_solved)
        return Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²), pt
    end

    function random_state(rng; wmax=3.0, magnetized=true)
        x = v.x_lo + (v.x_hi - v.x_lo) * rand(rng)
        ρ = exp10(x)
        yₑ = v.y_lo + (v.y_hi - v.y_lo) * rand(rng)
        sr = srange(v, ρ, yₑ)
        s = sr.s_min + (sr.s_max - sr.s_min) * rand(rng)
        w = wmax * rand(rng)
        B² = (magnetized && rand(rng) < 0.5) ? ρ * 10.0^(-2 + 3rand(rng)) : 0.0
        return ρ, s, yₑ, w, B², 2rand(rng) - 1
    end

    @testset "analytic Jacobian matches automatic differentiation" begin
        # A hand-written 2×2 Jacobian is the classic place for a dropped chain
        # rule term, and finite differences at 1e-6 would hide exactly that.
        ρ, yₑ, w = exp10(0.5 * (v.x_lo + v.x_hi)), 0.35, 0.9
        sr = srange(v, ρ, yₑ)
        s = 0.5 * (sr.s_min + sr.s_max)
        for (B², cos_vB) in ((0.0, 0.0), (0.3ρ, 0.4))
            cin, _ = forward(ρ, s, yₑ, w, B², cos_vB)
            r = E.residuals(v, cin.D, cin.τ, yₑ, cin.S_par, cin.S_perp, cin.B², s, w, NaN, opts.τ_floor_rel)
            f1(σ, ω) = E.residuals(v, cin.D, cin.τ, yₑ, cin.S_par, cin.S_perp, cin.B², σ, ω, NaN,
                                   opts.τ_floor_rel).f1
            f2(σ, ω) = E.residuals(v, cin.D, cin.τ, yₑ, cin.S_par, cin.S_perp, cin.B², σ, ω, NaN,
                                   opts.τ_floor_rel).f2
            @test r.df1_ds ≈ ForwardDiff.derivative(σ -> f1(σ, w), s) rtol = 1e-8
            @test r.df1_dw ≈ ForwardDiff.derivative(ω -> f1(s, ω), w) rtol = 1e-8
            @test r.df2_ds ≈ ForwardDiff.derivative(σ -> f2(σ, w), s) rtol = 1e-8
            @test r.df2_dw ≈ ForwardDiff.derivative(ω -> f2(s, ω), w) rtol = 1e-8
        end
    end

    @testset "convergence semantics" begin
        # The momentum residual carries a factor of cosh(w), so the test must be
        # on the normalized residual. An absolute test would be unsatisfiable at
        # high rapidity, where f₁'s own precision floor exceeds the tolerance.
        ρ, yₑ, w = exp10(0.5 * (v.x_lo + v.x_hi)), 0.35, 8.0
        sr = srange(v, ρ, yₑ)
        s = 0.5 * (sr.s_min + sr.s_max)
        cin, _ = forward(ρ, s, yₑ, w, 0.0, 0.0)
        r = E.residuals(v, cin.D, cin.τ, yₑ, cin.S_par, cin.S_perp, cin.B², s, w, NaN, opts.τ_floor_rel)
        @test E.f1_converged(r, 1e-12) == (abs(r.f1) <= 1e-12 * r.coshw)
        @test r.coshw > 1e3          # the regime where the distinction bites
        # The scaled norm is a max of the two normalized residuals, never an
        # L2 norm that the cosh-sized component would dominate.
        @test E.scaled_norm(r) == max(abs(r.f1) / r.coshw, abs(r.f2))
    end

    @testset "warm round trip" begin
        rng = StableRNG(20260916)
        n = 2000
        nfail = 0
        maxρ = maxs = maxw = 0.0
        for _ in 1:n
            ρ, s, yₑ, w, B², cos_vB = random_state(rng)
            cin, pt = forward(ρ, s, yₑ, w, B², cos_vB)
            out = con2prim(v, cin, opts, s, w, pt.u_solved)
            out.result === E.C2PResult.converged_newton ||
                out.result === E.C2PResult.converged_fallback || (nfail += 1)
            maxρ = max(maxρ, abs(out.ρ - ρ) / ρ)
            maxs = max(maxs, abs(out.s - s) / max(abs(s), 1))
            maxw = max(maxw, abs(out.w - w) / max(w, 1e-3))
            @test out.ye ≈ yₑ rtol = 1e-14
        end
        # Measured: every state converges in the Newton alone, and s and w come
        # back bit-identical because the warm start is already the answer.
        @test nfail == 0
        @test maxρ < 1e-12
        @test maxs < 1e-12
        @test maxw < 1e-12
    end

    @testset "cold round trip" begin
        rng = StableRNG(11)
        nfail = 0
        maxρ = 0.0
        for _ in 1:400
            ρ, s, yₑ, w, B², cos_vB = random_state(rng)
            cin, _ = forward(ρ, s, yₑ, w, B², cos_vB)
            out = con2prim(v, cin, opts)          # no guesses at all
            out.result === E.C2PResult.converged_newton ||
                out.result === E.C2PResult.converged_fallback || (nfail += 1)
            maxρ = max(maxρ, abs(out.ρ - ρ) / ρ)
        end
        # Measured: zero failures, worst density error ~1.4e-12.
        @test nfail == 0
        @test maxρ < 1e-9
    end

    @testset "forced fallback" begin
        # With the Newton budget set to zero the nested one-dimensional solve
        # must carry every state on its own.
        optsF = Con2PrimOptions(; max_iter_newton=0)
        rng = StableRNG(7)
        nfail = 0
        maxρ = 0.0
        for _ in 1:200
            ρ, s, yₑ, w, B², cos_vB = random_state(rng; magnetized=false)
            cin, _ = forward(ρ, s, yₑ, w, B², cos_vB)
            out = con2prim(v, cin, optsF)
            out.result === E.C2PResult.failed_no_bracket && (nfail += 1)
            out.result === E.C2PResult.failed_max_iter && (nfail += 1)
            maxρ = max(maxρ, abs(out.ρ - ρ) / ρ)
            @test out.iters_newton == 0
        end
        @test nfail == 0
        @test maxρ < 1e-9
    end

    @testset "two stages" begin
        # The second stage rebuilds the first stage's iterate from (s, w, u)
        # rather than receiving it, so it must reproduce it bit for bit: for any
        # stage, done or not, finishing gives con2prim's own result. Starved
        # options send states through the fallback and to failures as well.
        rng = StableRNG(31)
        ndone = npending = 0
        for o in (opts, Con2PrimOptions(; max_iter_newton=2), Con2PrimOptions(; max_iter_newton=0)),
            _ in 1:150
            ρ, s, yₑ, w, B², cos_vB = random_state(rng)
            cin, pt = forward(ρ, s, yₑ, w, B², cos_vB)
            for g in ((NaN, NaN, NaN), (s * (1 + 1e-3), w + 1e-3, pt.u_solved))
                out, st = con2prim_start(v, cin, o, g...)
                ref = con2prim(v, cin, o, g...)
                @test con2prim_finish(v, cin, o, st) === ref
                if is_done(st)
                    ndone += 1
                    @test out === ref
                else
                    npending += 1
                    @test isnan(out.ρ)
                    @test st.iters_newton == ref.iters_newton
                end
            end
        end
        @test ndone > 0
        @test npending > 0
        @test isbitstype(Con2PrimStage{Float64})
        @test sizeof(Con2PrimStage{Float64}) <= 32
    end

    @testset "limiting cases" begin
        ρ, yₑ = exp10(0.5 * (v.x_lo + v.x_hi)), 0.35
        sr = srange(v, ρ, yₑ)
        s = 0.5 * (sr.s_min + sr.s_max)

        # At rest there is no momentum, and the solver must return exactly that.
        cin, pt = forward(ρ, s, yₑ, 0.0, 0.0, 0.0)
        @test cin.S_par == 0 && cin.S_perp == 0
        out = con2prim(v, cin, opts)
        @test out.w ≈ 0 atol = 1e-9
        @test out.ρ ≈ ρ rtol = 1e-10
        @test out.W ≈ 1 atol = 1e-9

        # Unmagnetized at high rapidity, where cosh(w) is large.
        cin, pt = forward(ρ, s, yₑ, 6.0, 0.0, 0.0)
        out = con2prim(v, cin, opts, s, 6.0, pt.u_solved)
        @test out.w ≈ 6.0 rtol = 1e-10
        @test out.ρ ≈ ρ rtol = 1e-10

        # Strongly magnetized.
        cin, pt = forward(ρ, s, yₑ, 3.0, 1e4 * ρ, 0.25)
        out = con2prim(v, cin, opts, s, 3.0, pt.u_solved)
        @test out.ρ ≈ ρ rtol = 1e-9
        @test out.s ≈ s rtol = 1e-9
    end

    @testset "cold slow states" begin
        # Tiny rapidity is where the cancellation-free τ earns its keep: the
        # naive form would have lost most of its digits before the solver sees it.
        ρ, yₑ = exp10(0.5 * (v.x_lo + v.x_hi)), 0.35
        sr = srange(v, ρ, yₑ)
        for s in (sr.s_min * (1 + 1e-9), 0.5 * (sr.s_min + sr.s_max)), w in (1e-6, 1e-8, 1e-10)
            cin, _ = forward(ρ, s, yₑ, w, 0.0, 0.0)
            out = con2prim(v, cin, opts)
            @test out.result !== E.C2PResult.failed_no_bracket
            @test out.result !== E.C2PResult.failed_max_iter
            @test out.ρ ≈ ρ rtol = 1e-9
            @test out.s ≈ s rtol = 1e-8
            @test isfinite(out.w) && out.w >= 0
        end
    end

    @testset "cold states on the edges of the entropy window" begin
        # What the projection produces: a state put at rest, or nearly, with
        # its entropy exactly on an edge of the physical window. Cold solves of
        # these used to fail as `failed_no_bracket` while returning the state
        # itself, a third of them at ρ_min.
        nfail = 0
        maxρ = maxs = 0.0
        for x in (v.x_lo, 0.5 * (v.x_lo + v.x_hi)), yₑ in range(v.y_lo, v.y_hi; length=9),
            edge in (:s_min, :s_max), w in (0.0, 1e-3)

            ρ = exp10(x)
            s = getfield(srange(v, ρ, yₑ), edge)
            cin, _ = forward(ρ, s, yₑ, w, 0.0, 0.0)
            out = con2prim(v, cin, opts)
            out.result === E.C2PResult.converged_newton ||
                out.result === E.C2PResult.converged_fallback || (nfail += 1)
            maxρ = max(maxρ, abs(out.ρ - ρ) / ρ)
            maxs = max(maxs, abs(out.s - s) / s)
        end
        @test nfail == 0
        @test maxρ < 1e-11
        @test maxs < 1e-11
    end

    @testset "the search stays inside the density box" begin
        # The box's top, widened by the projection's slack so that a state
        # written onto that face solves again.
        ρ_max = exp10(v.x_hi)
        ρ_top = ρ_max * (1 + E.face_slack(Float64))
        @test E.w_floor(v, 0.5ρ_max, opts.w_max) === 0.0
        @test E.w_floor(v, ρ_top, opts.w_max) === 0.0
        @test 3ρ_max / cosh(E.w_floor(v, 3ρ_max, opts.w_max)) ≈ ρ_top rtol = 1e-14
        @test E.w_floor(v, ρ_max * 1e10, opts.w_max) == opts.w_max

        # above_floor, which lets a warm start skip the floor's acosh, may say no
        # above the floor but never yes below it.
        @test E.above_floor(v, ρ_top, 0.0)
        @test E.above_floor(v, 3ρ_max, 1.1 * acosh(3))
        @test !E.above_floor(v, 3ρ_max, 0.9 * acosh(3))
        rng_f = StableRNG(5)
        wrong = count(1:10_000) do _
            D = ρ_max * exp10(3rand(rng_f) - 1)
            w = opts.w_max * rand(rng_f)
            return E.above_floor(v, D, w) && w < E.w_floor(v, D, opts.w_max)
        end
        @test wrong == 0

        # A state above the box: unconfined, the inner solve finds it; confined,
        # f₁ ≥ 0 on the ρ_max face, and the solve returns the face after testing
        # it once instead of bisecting its budget away.
        for (dx, w, frac) in ((0.5, 2.0, 0.5), (0.2, 1.0, 0.2), (1.0, 3.0, 0.9))
            ρ = exp10(v.x_hi + dx)
            sr = srange(v, ρ, 0.3)
            s = sr.s_min + frac * (sr.s_max - sr.s_min)
            cin, _ = forward(ρ, s, 0.3, w, 0.0, 0.0)
            w_lo = E.w_floor(v, cin.D, opts.w_max)
            inner(lo) = E.inner_solve_w(v, cin.D, cin.τ, 0.3, cin.S_par, cin.S_perp, cin.B², s, lo, opts.w_max,
                                        opts.τ_floor_rel, 3.0, NaN, opts.max_iter_1d, opts.tol)
            r, iters = inner(w_lo)
            @test r.w == w_lo && r.f1 > 0
            @test iters <= 3
            @test inner(0.0)[1].ρ ≈ ρ rtol = 1e-10
            # So con2prim does not solve it, and reports it at the face.
            out = con2prim(v, cin, opts)
            @test out.result === E.C2PResult.failed_no_bracket
            @test out.ρ <= ρ_top * (1 + 1e-12)
        end

        # A fast state just inside the box still solves, cold and warm.
        rng = StableRNG(17)
        maxρ = 0.0
        for _ in 1:40
            ρ = exp10(v.x_hi - 0.2rand(rng))
            sr = srange(v, ρ, 0.3)
            s = sr.s_min + (sr.s_max - sr.s_min) * (0.05 + 0.9rand(rng))
            w = 2 + 4rand(rng)
            cin, pt = forward(ρ, s, 0.3, w, rand(rng) < 0.5 ? 0.0 : ρ * 10.0^(-2 + 3rand(rng)), 2rand(rng) - 1)
            @test cin.D > ρ_max
            # The last guess lies above the box, so the floor lifts it.
            for out in (con2prim(v, cin, opts), con2prim(v, cin, opts, s * (1 + 1e-3), w + 1e-3, pt.u_solved),
                        con2prim(v, cin, opts, s * (1 + 1e-3), 0.0, pt.u_solved))
                @test out.result === E.C2PResult.converged_newton || out.result === E.C2PResult.converged_fallback
                maxρ = max(maxρ, abs(out.ρ - ρ) / ρ)
            end
        end
        @test maxρ < 1e-9
    end

    @testset "bracket_scan never returns an empty bracket" begin
        # The scan's candidates include both edges of the physical window
        # twice. With the root on an edge the two copies' solves can disagree
        # in sign, and that zero-width pair used to be the one picked.
        nempty = 0
        for x in range(v.x_lo, v.x_hi; length=4), yₑ in (0.1, 0.3, 0.5), edge in (:s_min, :s_max)
            ρ = exp10(x)
            s = getfield(srange(v, ρ, yₑ), edge)
            cin, pt = forward(ρ, s, yₑ, 0.0, 0.0, 0.0)
            for u in (NaN, pt.u_solved - 1e-11, pt.u_solved + 1e-11)
                scan = E.bracket_scan(v, cin.D, cin.τ, yₑ, cin.S_par, cin.S_perp, cin.B²,
                                      E.w_floor(v, cin.D, opts.w_max), opts.w_max, opts.τ_floor_rel,
                                      opts.bracket_scan, s, 0.0, 0.0, u, opts.max_iter_1d, opts.tol)
                scan.bracketed && !(scan.s_lo < scan.s_hi) && (nempty += 1)
            end
        end
        @test nempty == 0
    end

    @testset "cold seed accuracy" begin
        # The seed exists so the Newton starts in the right neighbourhood; the
        # C++ measures it as landing within 20% on magnetized states.
        rng = StableRNG(99)
        for _ in 1:50
            ρ, s, yₑ, w, B², cos_vB = random_state(rng; wmax=2.0)
            B² = ρ * 10.0^(-1 + 2rand(rng))
            cin, _ = forward(ρ, s, yₑ, w, B², cos_vB)
            seed = E.cold_seed(v, cin.D, cin.τ, yₑ, cin.S_par, cin.S_perp, cin.B², E.w_floor(v, cin.D, opts.w_max),
                               opts.w_max, opts.seed_passes, opts.seed_s_iters)
            @test isfinite(seed.s) && isfinite(seed.w)
            @test abs(seed.s - s) / max(abs(s), 1) < 0.2
            @test seed.w >= 0
        end
    end

    @testset "seed_z_solve recovers the exact hydrodynamic result" begin
        # With no field the residual is linear, so the very first trial point is
        # the root and the answer must be exact rather than bisected towards.
        D, E_tot, p = 1.0e10, 3.0e10, 2.0e9
        @test E.seed_z_solve(D, E_tot, p, 0.0, 0.0, 40) ≈ E_tot + p rtol = 1e-15
        # z ≥ D always, since z = D·h·cosh(w) and h ≥ 1.
        @test E.seed_z_solve(D, 0.5D, 0.0, 0.0, 0.0, 40) == D
    end

    @testset "seed_z_solve does not overflow Float32" begin
        # Conserved quantities reach 1e17..1e21 on real tables, so B²·S⊥² and q³
        # are far beyond floatmax(Float32). Formed naively, the residual became
        # Inf, bisection drove z to its lower bound, and every magnetized
        # Float32 cold start began at w_max -- a tenth of them then failed.
        D, E_tot, p = 1.0e17, 4.0e19, 1.0e18
        for (S_perp, B²) in ((3.9e19, 1.0e13), (2.0e19, 1.0e17), (1.0e19, 5.0e19))
            z64 = E.seed_z_solve(D, E_tot, p, S_perp, B², 40)
            z32 = E.seed_z_solve(Float32.((D, E_tot, p, S_perp, B²))..., 40)
            @test z32 isa Float32
            @test isfinite(z32)
            @test z32 ≈ z64 rtol = 1e-5
        end
    end

    @testset "Float32 round trip" begin
        # The measured Float32 behaviour (docs/src/precision.md), held on the
        # synthetic table where it is cheap. The conservatives are built at
        # Float32 too, so what is measured is the solver rather than the
        # conditioning of rounding a Float64 state. At the old 64-eps tolerance,
        # below the Float32 residual's noise floor, this sample had 7 cold
        # failures.
        v32 = E.narrow(v, Float32)
        o32 = Con2PrimOptions{Float32}()
        @test o32.tol == 512 * eps(Float32)
        rng = StableRNG(32)
        nfail_cold = nfail_warm = 0
        errs = Float64[]
        for _ in 1:2000
            ρ = Float32(exp10(v.x_lo + (v.x_hi - v.x_lo) * rand(rng)))
            yₑ = Float32(v.y_lo + (v.y_hi - v.y_lo) * rand(rng))
            sr = srange(v32, ρ, yₑ)
            s = sr.s_min + (sr.s_max - sr.s_min) * Float32(0.05 + 0.9rand(rng))
            w = Float32(4rand(rng))
            pt = evaluate(v32, ρ, s, yₑ, NaN32)
            B² = rand(rng) < 0.5 ? ρ * pt.h * Float32(exp10(-6 + 7rand(rng))) : 0.0f0
            c = prim2con(v32, ρ, s, yₑ, w, B², Float32(2rand(rng) - 1), pt.u_solved)
            cin = Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²)
            oc = con2prim(v32, cin, o32)
            ow = con2prim(v32, cin, o32, s, w, pt.u_solved)
            ok(o) = o.result === E.C2PResult.converged_newton || o.result === E.C2PResult.converged_fallback
            ok(oc) ? push!(errs, abs(oc.ρ - ρ) / ρ) : (nfail_cold += 1)
            ok(ow) || (nfail_warm += 1)
        end
        @test nfail_cold <= 2
        @test nfail_warm == 0
        sort!(errs)
        @test errs[ceil(Int, 0.99 * length(errs))] < 1e-3
    end

    @testset "allocation-free and type-generic" begin
        ρ, yₑ = exp10(0.5 * (v.x_lo + v.x_hi)), 0.35
        sr = srange(v, ρ, yₑ)
        s = 0.5 * (sr.s_min + sr.s_max)
        cin, pt = forward(ρ, s, yₑ, 0.8, 0.1ρ, 0.3)
        # The warm Newton path is the one that dominates a hydro run.
        @test_noallocs _c2p_allocs(v, cin, opts, s, 0.8, pt.u_solved)
        @test (@inferred con2prim(v, cin, opts, s, 0.8, pt.u_solved)) isa Con2PrimOut{Float64}
        # And the fallback path, which uses the stack scratch for its scan.
        optsF = Con2PrimOptions(; max_iter_newton=0)
        @test_noallocs _c2p_allocs(v, cin, optsF, NaN, NaN, NaN)
        # Each stage on its own.
        @test_noallocs _c2p_start_allocs(v, cin, optsF)
        @test_noallocs _c2p_finish_allocs(v, cin, optsF, con2prim_start(v, cin, optsF)[2])
    end
end

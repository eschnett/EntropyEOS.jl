# Allocation probes must live at top level: a helper defined inside a @testset
# body is a local closure, and @allocated then measures the closure machinery
# rather than the kernel under test.
_tail_probe(v, x, u, y, sp) = EntropyEOS.extended_sample(v, x, u, y, sp).f
function _tail_allocs(v, x, u, y, sp)
    _tail_probe(v, x, u, y, sp)
    return @allocated _tail_probe(v, x, u, y, sp)
end

@testset "adapter tails" begin
    E = EntropyEOS

    # A strictly positive spline with genuine curvature along every axis, so the
    # log-space tails are exercisable and no second derivative is accidentally
    # zero (which would make relative tolerances meaningless).
    nx, nu, ny = 12, 11, 9
    x0, hx, u0, hu, y0, hy = -2.0, 0.3, -1.0, 0.2, 0.05, 0.05
    coeffs = [3.0 + 0.3sin(0.3i) + 0.4cos(0.25j) + 0.2sin(0.2i + 0.3j) + 0.05k^2
              for i in 1:(nx + 2), j in 1:(nu + 2), k in 1:(ny + 2)]
    v = BsplineView3(coeffs, x0, hx, u0, hu, y0, hy)
    x_lo, x_hi = x0, x0 + (nx - 1) * hx
    u_lo, u_hi = u0, u0 + (nu - 1) * hu
    y_lo, y_hi = y0, y0 + (ny - 1) * hy
    x_ext_lo = x_lo - 8hx
    inv_c² = 1 / 2.99792458e10^2
    specσ = E.ExtSpec(x_lo, x_hi, u_lo, u_hi, x_ext_lo, 1.0e-6, true, true, 0.0, 1.5e18 * inv_c², inv_c²)
    specL = E.ExtSpec(x_lo, x_hi, u_lo, u_hi, x_ext_lo, 1.0e-8, false, false, 0.0, 1.5e18 * inv_c², inv_c²)

    @testset "transparent inside the box, bitwise" begin
        # The whole extension design rests on this: an interior query must fall
        # through to a single plain spline evaluation with no tail applied, so
        # adding the extensions cannot perturb any existing interior result.
        # Bitwise, not approximate -- this is a within-language exactness
        # property, and loosening it to ≈ would silently permit a real change.
        n = 0
        for x in range(x_lo, x_hi; length=13), u in range(u_lo, u_hi; length=11), y in range(y_lo, y_hi; length=7)
            plain = bspline_eval3(v, x, u, y)
            for spec in (specσ, specL)
                ext = E.extended_sample(v, x, u, y, spec)
                for f in fieldnames(BsplineEval3)
                    @test getfield(plain, f) === getfield(ext, f)
                    n += 1
                end
            end
        end
        @test n == 13 * 11 * 7 * 2 * 7
    end

    @testset "ramp_track" begin
        t = E.Track1D(2.0, 0.7, -0.4)
        w = 0.3
        # At the seam the ramp is the identity, which is what makes the tail C²
        # there: value, slope and curvature all still match the spline sample.
        @test E.ramp_track(t, 0.0, w) === t
        # Phase 1 and phase 2 must agree at the blend-cell edge, on both sides.
        for sgn in (-1.0, 1.0)
            inside = E.ramp_track(t, sgn * w, w)
            outside = E.ramp_track(t, sgn * w * (1 + 1e-12), w)
            @test inside.f0 ≈ outside.f0 atol = 1e-12
            @test inside.f1 ≈ outside.f1 atol = 1e-12
            @test inside.f2 ≈ outside.f2 atol = 1e-12
            # Curvature ramps linearly to exactly zero at the edge, and stays
            # there: beyond the blend cell the tail is straight.
            @test E.ramp_track(t, sgn * w, w).f2 == 0.0
            @test E.ramp_track(t, sgn * 3w, w).f2 == 0.0
            @test E.ramp_track(t, sgn * 3w, w).f1 == E.phase2_slope(t, sgn, w)
        end
        # Halfway through the blend cell the curvature is halved.
        @test E.ramp_track(t, w / 2, w).f2 ≈ t.f2 / 2
        # The slope never overshoots the interval between its endpoints.
        m = E.phase2_slope(t, 1.0, w)
        for d in range(0, w; length=17)
            f1 = E.ramp_track(t, d, w).f1
            @test min(t.f1, m) - 1e-14 <= f1 <= max(t.f1, m) + 1e-14
        end
    end

    @testset "floor_slope (monotonicity guard)" begin
        w = 0.3
        t = E.Track1D(1.0, 0.2, -5.0)
        # A non-positive floor is a no-op -- every track but the u-direction one.
        @test E.floor_slope(t, 1.0, w, 0.0) === t
        @test E.floor_slope(t, 1.0, w, -1.0) === t
        # The floor raises the slope and then caps the curvature so that the
        # asymptotic slope cannot fall back below the floor.
        for (sgn, floor) in ((1.0, 0.5), (-1.0, 0.5), (1.0, 0.1), (-1.0, 0.1))
            g = E.floor_slope(t, sgn, w, floor)
            @test g.f0 == t.f0                       # the seam value is untouched
            @test g.f1 >= floor - 1e-15
            @test E.phase2_slope(g, sgn, w) >= floor - 1e-14
        end
        # A track already satisfying the floor keeps its slope.
        ok = E.Track1D(1.0, 2.0, 0.0)
        @test E.floor_slope(ok, 1.0, w, 0.5).f1 == 2.0
    end

    @testset "cap_slope (causal cap)" begin
        w = 0.3
        t = E.Track1D(1.0, 0.4, 6.0)        # asymptotic slope 0.4 + 6*0.15 = 1.3
        @test E.phase2_slope(t, 1.0, w) ≈ 1.3
        capped = E.cap_slope(t, w, 0.9)
        # A binding cap sets the asymptotic slope to the cap exactly ...
        @test E.phase2_slope(capped, 1.0, w) ≈ 0.9
        # ... by lowering the curvature alone. The seam value and seam slope are
        # deliberately left alone so the tail stays C¹ there, which is what
        # keeps U and U_s continuous across the seam.
        @test capped.f0 == t.f0
        @test capped.f1 == t.f1
        # A non-binding cap is an exact no-op.
        @test E.cap_slope(t, w, 2.0) === t
    end

    @testset "capped_track: the floor wins lexicographically" begin
        w, d = 0.3, 0.5
        t = E.Track1D(1.0, 0.05, 8.0)
        m_floor, m_cap = 0.5, 0.2          # cap below floor
        got = E.capped_track(t, d, w, m_floor, m_cap)
        # Causality is never enforced at the price of the monotonicity the
        # T-solve depends on, so the effective bound is the floor, not the cap.
        @test got.f1 ≈ m_floor atol = 1e-14
        # With the cap above the floor it binds normally.
        got2 = E.capped_track(t, d, w, 0.1, 0.5)
        @test got2.f1 ≈ 0.5 atol = 1e-14
    end

    @testset "log_sample / exp_sample round trip" begin
        b = bspline_eval3(v, 0.1, 0.3, 0.2)
        r = E.exp_sample(E.log_sample(b))
        for f in fieldnames(BsplineEval3)
            orig, back = getfield(b, f), getfield(r, f)
            # Mixed tolerance: the second-derivative fields are formed by a
            # subtraction that cancels, so a pure relative bound is meaningless
            # wherever the quantity itself is near zero.
            @test back ≈ orig atol = 1e-12 * max(abs(orig), 1) rtol = 1e-12
        end
    end

    @testset "L_slope_cap" begin
        shift_hat = 1.5e18 * inv_c²
        L_b = 19.0
        Eh = exp10(L_b) * inv_c²
        ε = Eh - shift_hat
        b_cap = 3.0
        m = E.L_slope_cap(L_b, b_cap, shift_hat, inv_c²)
        # It inverts b = ln10 · m_L · E / ε̂.
        @test log(10) * m * Eh / ε ≈ b_cap rtol = 1e-13
        # No causal statement is possible where ε̂ ≤ 0, so there is no cap.
        @test E.L_slope_cap(L_b, b_cap, Eh * 2, inv_c²) == 0.0
        @test E.L_slope_cap(-Inf, b_cap, shift_hat, inv_c²) == 0.0
        @test E.L_slope_cap(NaN, b_cap, shift_hat, inv_c²) == 0.0
    end

    @testset "xlow_log_ok" begin
        w, depth = 0.3, 2.4
        gentle = BsplineEval3(1.0, 0.5, 0.0, 0.0, 0.1, 0.0, 0.0)
        @test E.xlow_log_ok(gentle, w, depth)
        # An exploding log slope would overflow across the band, so the guard
        # declines and the caller falls back to the plain linear tail.
        steep = BsplineEval3(1.0, 1.0e6, 0.0, 0.0, 0.0, 0.0, 0.0)
        @test !E.xlow_log_ok(steep, w, depth)
        # NaN must decline too: `<=` on NaN is false, which is the intent.
        @test !E.xlow_log_ok(BsplineEval3(NaN, NaN, 0.0, 0.0, NaN, 0.0, 0.0), w, depth)
    end

    @testset "seam continuity" begin
        # C² at a seam means the one-sided limit of the value and of the first
        # two derivatives, taken from *outside*, converges to the spline's own
        # value at the seam. Asserting convergence rather than a hand-picked
        # bound keeps this test honest: the residual must shrink in proportion
        # to the step, which a genuine discontinuity would not do.
        for (spec, fld) in ((specσ, :u), (specL, :u), (specσ, :x), (specL, :x))
            seam_sample = fld === :u ? bspline_eval3(v, 0.1, u_hi, 0.2) : bspline_eval3(v, x_lo, 0.1, 0.2)
            outside(δ) = fld === :u ? E.extended_sample(v, 0.1, u_hi + δ, 0.2, spec) :
                         E.extended_sample(v, x_lo - δ, 0.1, 0.2, spec)
            getters = fld === :u ? (:f, :fu, :fuu) : (:f, :fx, :fxx)
            for g in getters
                target = getfield(seam_sample, g)
                e1 = abs(getfield(outside(1.0e-4), g) - target)
                e2 = abs(getfield(outside(1.0e-5), g) - target)
                e3 = abs(getfield(outside(1.0e-6), g) - target)
                # First order in the step, so each decade cuts the error by ~10.
                @test e2 < 0.2e1
                @test e3 < 0.2e2
                @test e3 < 1.0e-4          # and it is genuinely small
            end
        end
    end

    @testset "log tail is exactly exponential beyond the blend cell" begin
        # This is the point of the log-space construction: past the blend cell
        # σ grows at a constant logarithmic rate, which makes the tail's sound
        # speed the constant b/α - 1 rather than something that crosses one.
        f = [E.extended_sample(v, 0.1, u_hi + hu * k, 0.2, specσ).f for k in 2:7]
        ratios = [f[i + 1] / f[i] for i in 1:(length(f) - 1)]
        @test all(r -> r ≈ ratios[1], ratios)
        @test maximum(abs.(ratios .- ratios[1])) < 1e-14
        # The linear (non-log) tail is instead exactly straight out there.
        g = [E.extended_sample(v, 0.1, u_hi + hu * k, 0.2, specL).f for k in 2:7]
        d2 = [g[i + 1] - 2g[i] + g[i - 1] for i in 2:(length(g) - 1)]
        @test maximum(abs.(d2)) < 1e-13
    end

    @testset "corner composition order" begin
        # The temperature tail is applied first, then the density tail to its
        # result, with the raw sample taken at the seam clamped in BOTH
        # directions. Reversing the order gives a different answer, so this
        # pins the order rather than merely exercising it.
        x, u, y = x_lo - 2hx, u_hi + 2hu, 0.2
        got = E.extended_sample(v, x, u, y, specL)
        b = bspline_eval3(v, x_lo, u_hi, y)
        uthen = E.apply_u_tail(b, u - u_hi, E.TailSpec(hu, specL.u_m_floor, 0.0))
        expected = E.apply_x_tail(uthen, x - x_lo, E.TailSpec(hx, 0.0, 0.0))
        for f in fieldnames(BsplineEval3)
            @test getfield(got, f) === getfield(expected, f)
        end
        xthen = E.apply_x_tail(b, x - x_lo, E.TailSpec(hx, 0.0, 0.0))
        reversed = E.apply_u_tail(xthen, u - u_hi, E.TailSpec(hu, specL.u_m_floor, 0.0))
        @test reversed.f != expected.f
    end

    @testset "finite and allocation-free everywhere in the extended box" begin
        for x in (x_ext_lo, x_lo - hx, x_lo, 0.1, x_hi, x_hi + 3hx),
            u in (u_lo - 8hu, u_lo, 0.1, u_hi, u_hi + 8hu),
            spec in (specσ, specL)

            s = E.extended_sample(v, x, u, 0.2, spec)
            @test all(isfinite, (s.f, s.fx, s.fu, s.fy, s.fxx, s.fxu, s.fuu))
        end
        @test_noallocs _tail_allocs(v, x_lo - 1.0, u_hi + 1.0, 0.2, specσ)
        @test_noallocs _tail_allocs(v, x_lo - 1.0, u_hi + 1.0, 0.2, specL)
        @test_noallocs _tail_allocs(v, 0.1, 0.1, 0.2, specσ)
        @test (@inferred E.extended_sample(v, x_lo - 1.0, u_hi + 1.0, 0.2, specσ)) isa BsplineEval3{Float64}
    end
end

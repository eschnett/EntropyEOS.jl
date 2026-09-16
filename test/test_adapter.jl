# Allocation probes live at top level; see the note in test_adapter_tail.jl.
_eval_probe(v, ρ, s, y, u) = EntropyEOS.evaluate(v, ρ, s, y, u).U
function _eval_allocs(v, ρ, s, y, u)
    _eval_probe(v, ρ, s, y, u)
    return @allocated _eval_probe(v, ρ, s, y, u)
end

@testset "adapter" begin
    E = EntropyEOS
    c² = E.C_LIGHT_CM_S^2

    opts = SyntheticOptions()
    tbl = make_synthetic_table(opts)
    eos = E.build_eos(tbl)
    v = EOSTableView(eos)

    @testset "matches the C++ reference output" begin
        # The C++ README publishes the expected output of its worked example.
        # Reproducing that configuration exercises the whole pipeline at once --
        # synthetic table, spline fit, κ derivation, T-solve, chain rule -- and
        # every digit it prints must come back.
        yₑ = 0.4
        ρ = exp10(0.5 * (v.x_lo + v.x_hi))
        sr = srange(v, ρ, yₑ)
        s = 0.5 * (sr.s_min + sr.s_max)
        pt = evaluate(v, ρ, s, yₑ, NaN)
        @test ρ ≈ 9.999e9 rtol = 1e-4              # C++: rho 9.999e+09
        @test s ≈ 43.394858 rtol = 1e-8            # C++: s 43.394858
        @test pt.p ≈ 2.376396e7 rtol = 1e-6        # C++: p = 2.376396e+07
        @test pt.T_MeV ≈ 1.5811 rtol = 1e-4        # C++: T = 1.5811 MeV
        @test pt.cs² ≈ 0.0039 rtol = 1e-2          # C++: cs2 = 0.0039
    end

    @testset "κ and the build" begin
        # κ ≤ 1 by construction. It dips just below one here because the
        # extended-box scan finds ε̂ slightly negative out in the tails, which
        # is exactly what that scan exists to catch.
        @test v.κ <= 1
        @test v.κ ≈ 1 atol = 1e-4
        @test eos.m_B_star_g ≈ v.κ * E.M_AMU_G
        @test eos.m_B_table_g == E.M_AMU_G
        # A clean synthetic table is monotone in T, so the audit finds nothing.
        @test eos.audit.σ_u.violation_count == 0
        @test eos.audit.L_u.violation_count == 0
        @test eos.audit.σ_u.min_value > 0
        @test eos.audit.L_u.min_value > 0
        # The extended box is wider than the physical one on both axes, and Yₑ
        # has no extension at all.
        @test v.x_ext_lo < v.x_lo && v.x_ext_hi > v.x_hi
        @test v.u_ext_lo < v.u_lo && v.u_ext_hi > v.u_hi
    end

    @testset "threaded scans are deterministic" begin
        # The refined-grid scans that derive κ run one Yₑ slice per thread. κ is
        # part of the EOS identity -- a table swap that changes it changes D, so
        # checkpoints are not interchangeable across it -- which makes a racy or
        # thread-count-dependent reduction a correctness bug, not a performance
        # one. Rebuilding must give bitwise identical results.
        #
        # An earlier version inlined the slice body in the `@threads` loop; the
        # scalar minimum was boxed into the enclosing frame and shared between
        # threads, which shifted κ by ~6e-8 between thread counts. Hoisting the
        # body into a function fixed it. This test is what would catch that
        # coming back, and it only bites when the runner has threads, so CI sets
        # them (see .github/workflows/CI.yml).
        a = E.build_eos(tbl)
        b = E.build_eos(tbl)
        @test EOSTableView(a).κ === EOSTableView(b).κ
        @test a.audit.σ_u.min_value === b.audit.σ_u.min_value
        @test a.audit.L_u.min_value === b.audit.L_u.min_value
        @test a.audit.σ_u.violation_count == b.audit.σ_u.violation_count
        @test a.audit.L_u.violation_count == b.audit.L_u.violation_count
        @test [l.value for l in a.audit.σ_u.worst] == [l.value for l in b.audit.σ_u.worst]
        @test [l.value for l in a.audit.L_u.worst] == [l.value for l in b.audit.L_u.worst]
        # And identical to the view built once at the top of this file.
        @test EOSTableView(a).κ === v.κ
    end

    @testset "build validation" begin
        short = RawTable(collect(1.0:3.0), collect(0.0:0.5:1.5), collect(0.1:0.1:0.4))
        add_field!(short, "entropy", zeros(3, 4, 4))
        add_field!(short, "logenergy", zeros(3, 4, 4))
        add_attribute!(short, "energy_shift", 1.0)
        @test_throws ArgumentError E.build_eos(short)

        nonuniform = RawTable([1.0, 2.0, 3.0, 5.0], collect(0.0:0.5:1.5), collect(0.1:0.1:0.4))
        add_field!(nonuniform, "entropy", zeros(4, 4, 4))
        add_field!(nonuniform, "logenergy", zeros(4, 4, 4))
        add_attribute!(nonuniform, "energy_shift", 1.0)
        @test_throws ArgumentError E.build_eos(nonuniform)

        noshift = RawTable(collect(1.0:4.0), collect(0.0:0.5:1.5), collect(0.1:0.1:0.4))
        add_field!(noshift, "entropy", zeros(4, 4, 4))
        add_field!(noshift, "logenergy", zeros(4, 4, 4))
        @test_throws ArgumentError E.build_eos(noshift)

        nofield = RawTable(collect(1.0:4.0), collect(0.0:0.5:1.5), collect(0.1:0.1:0.4))
        add_field!(nofield, "entropy", zeros(4, 4, 4))
        add_attribute!(nofield, "energy_shift", 1.0)
        @test_throws ArgumentError E.build_eos(nofield)

        nanfield = RawTable(collect(1.0:4.0), collect(0.0:0.5:1.5), collect(0.1:0.1:0.4))
        add_field!(nanfield, "entropy", fill(NaN, 4, 4, 4))
        add_field!(nanfield, "logenergy", zeros(4, 4, 4))
        add_attribute!(nanfield, "energy_shift", 1.0)
        @test_throws ArgumentError E.build_eos(nanfield)
    end

    @testset "node round trip against the closed form" begin
        # The adapter's density is ρ★ = κ·ρ, not the raw table density, and its
        # energy is re-zeroed so that ρ★(1+U) = ρ(1+ε̂) exactly. Feeding it a raw
        # density instead is the single easiest way to misuse this library, so
        # the conversion is spelled out here rather than hidden in a helper.
        maxT = maxU = maxp = maxT̂ = maxid = 0.0
        for i in 3:5:(E.nρ(tbl) - 2), j in 3:5:(E.nT(tbl) - 2), k in 2:3:(E.nYₑ(tbl) - 1)
            ρ, Tj, yₑ = E.density(tbl, i), E.temperature(tbl, j), E.electron_fraction(tbl, k)
            s = synthetic_s(ρ, Tj, yₑ, opts)            # entropy is κ-invariant
            pt = evaluate(v, v.κ * ρ, s, yₑ, NaN)
            ε̂ = synthetic_eps(ρ, Tj, yₑ, opts) / c²
            U_expected = (1 + ε̂) / v.κ - 1
            T̂_expected = Tj * E.MEV_TO_ERG / (eos.m_B_star_g * c²)
            maxT = max(maxT, abs(pt.T_MeV - Tj) / Tj)
            maxU = max(maxU, abs(pt.U - U_expected) / abs(U_expected))
            maxp = max(maxp, abs(pt.p * c² - synthetic_p(ρ, Tj, yₑ, opts)) / synthetic_p(ρ, Tj, yₑ, opts))
            maxT̂ = max(maxT̂, abs(pt.T̂ - T̂_expected) / T̂_expected)
            # The rescaling is exact, not approximate.
            maxid = max(maxid, abs((v.κ * ρ) * (1 + pt.U) - ρ * (1 + ε̂)) / (ρ * (1 + ε̂)))
        end
        # Measured on the 40×30×10 grid: T 1.4e-14, U 4.6e-13, p 8.3e-5,
        # T̂ 8.3e-5. p and T̂ carry spline-derivative truncation; the value path
        # is at roundoff.
        @test maxT < 1e-12
        @test maxU < 1e-10
        @test maxp < 1e-3
        @test maxT̂ < 1e-3
        @test maxid < 1e-12
    end

    @testset "derivatives agree with automatic differentiation" begin
        # Far stronger than the finite differences the C++ has to use: AD is
        # exact up to roundoff, so a dropped chain-rule term cannot hide.
        ρ = v.κ * E.density(tbl, 20)
        yₑ = E.electron_fraction(tbl, 5)
        s = synthetic_s(E.density(tbl, 20), E.temperature(tbl, 15), yₑ, opts)
        pt = evaluate(v, ρ, s, yₑ, NaN)
        Uofρ(r) = evaluate(v, r, s, yₑ, NaN).U
        Uofs(σ) = evaluate(v, ρ, σ, yₑ, NaN).U
        @test pt.U_ρ ≈ ForwardDiff.derivative(Uofρ, ρ) rtol = 1e-11
        @test pt.U_s ≈ ForwardDiff.derivative(Uofs, s) rtol = 1e-11
        @test pt.U_ρρ ≈ ForwardDiff.derivative(r -> ForwardDiff.derivative(Uofρ, r), ρ) rtol = 1e-10
        @test pt.U_ρs ≈
            ForwardDiff.derivative(σ -> ForwardDiff.derivative(r -> evaluate(v, r, σ, yₑ, NaN).U, ρ), s) rtol =
            1e-10
        # The derived quantities follow from the same partials.
        @test pt.p ≈ ρ^2 * pt.U_ρ rtol = 1e-14
        @test pt.h ≈ 1 + pt.U + pt.p / ρ rtol = 1e-14
        @test pt.cs² ≈ (2ρ * pt.U_ρ + ρ^2 * pt.U_ρρ) / pt.h rtol = 1e-14
        @test pt.T̂ == pt.U_s
    end

    @testset "srange matches the closed form" begin
        for i in 4:7:(E.nρ(tbl) - 3), k in 2:3:(E.nYₑ(tbl) - 1)
            ρ, yₑ = E.density(tbl, i), E.electron_fraction(tbl, k)
            sr = srange(v, v.κ * ρ, yₑ)
            @test sr.s_min ≈ synthetic_s(ρ, opts.T_min_MeV, yₑ, opts) rtol = 1e-12
            @test sr.s_max ≈ synthetic_s(ρ, opts.T_max_MeV, yₑ, opts) rtol = 1e-12
            # The extended window is strictly wider and still finite.
            sre = srange_extended(v, v.κ * ρ, yₑ)
            @test sre.s_min < sr.s_min
            @test sre.s_max > sr.s_max
            @test isfinite(sre.s_min) && isfinite(sre.s_max)
        end
    end

    @testset "entropy is strictly increasing in u" begin
        # This is what makes the T-solve globally invertible, and it must hold
        # across the extended bracket, not merely inside the table.
        ρ, yₑ = v.κ * E.density(tbl, 20), 0.3
        us = range(v.u_ext_lo, v.u_ext_hi; length=200)
        σs = [sigma_extended(v, ρ, u, yₑ) for u in us]
        @test all(diff(σs) .> 0)
        @test all(isfinite, σs)
    end

    @testset "out-of-range flags" begin
        yₑ = 0.3
        ρ_in = exp10(0.5 * (v.x_lo + v.x_hi))
        s_in = 0.5 * (srange(v, ρ_in, yₑ).s_min + srange(v, ρ_in, yₑ).s_max)
        @test evaluate(v, ρ_in, s_in, yₑ, NaN).flags == 0

        # Below the table is a supported extension; above it is not, and keeps a
        # hard out-of-bounds flag, because a converged state there is invalid.
        @test evaluate(v, exp10(v.x_lo - 2), s_in, yₑ, NaN).flags & FLAG_EXT_ρ_LOW != 0
        @test evaluate(v, exp10(v.x_hi + 2), s_in, yₑ, NaN).flags & FLAG_OOB_ρ_HIGH != 0
        # Yₑ has no extension: it is clamped and flagged.
        @test evaluate(v, ρ_in, s_in, v.y_lo - 0.1, NaN).flags & FLAG_CLAMP_YE != 0
        @test evaluate(v, ρ_in, s_in, v.y_hi + 0.1, NaN).flags & FLAG_CLAMP_YE != 0
        # Entropy outside the physical window lands in the designed extension.
        @test evaluate(v, ρ_in, srange(v, ρ_in, yₑ).s_min - 5, yₑ, NaN).flags & FLAG_EXT_S_LOW != 0
        @test evaluate(v, ρ_in, srange(v, ρ_in, yₑ).s_max + 5, yₑ, NaN).flags & FLAG_EXT_S_HIGH != 0
    end

    @testset "finite and physical over the whole extended box" begin
        # Whatever the solver asks for, the answer must be finite and usable --
        # that is the property the whole extension design exists to provide.
        for x in range(v.x_ext_lo, v.x_ext_hi; length=11),
            yₑ in range(v.y_lo, v.y_hi; length=4)

            ρ = exp10(x)
            sr = srange_extended(v, ρ, yₑ)
            for s in range(sr.s_min, sr.s_max; length=9)
                pt = evaluate(v, ρ, s, yₑ, NaN)
                @test isfinite(pt.U) && isfinite(pt.p) && isfinite(pt.cs²) && isfinite(pt.T_MeV)
                @test pt.U >= 0            # guaranteed by the κ re-zeroing
                @test pt.T̂ > 0             # U_s > 0 is what makes the solve monotone
                @test pt.T_MeV > 0
            end
        end
    end

    @testset "warm start" begin
        ρ, yₑ = v.κ * E.density(tbl, 20), 0.3
        s = synthetic_s(E.density(tbl, 20), E.temperature(tbl, 15), yₑ, opts)
        cold = evaluate(v, ρ, s, yₑ, NaN)
        warm = evaluate(v, ρ * 1.001, s, yₑ, cold.u_solved)
        @test warm.iters <= cold.iters
        @test warm.flags & FLAG_MAXITER == 0
        # The warm start is threaded explicitly, so the two calls are otherwise
        # independent: evaluating twice from the same guess gives the same answer.
        @test evaluate(v, ρ, s, yₑ, cold.u_solved).U == evaluate(v, ρ, s, yₑ, cold.u_solved).U
    end

    @testset "allocation-free and type-generic" begin
        ρ, yₑ = v.κ * E.density(tbl, 20), 0.3
        s = synthetic_s(E.density(tbl, 20), E.temperature(tbl, 15), yₑ, opts)
        @test _eval_allocs(v, ρ, s, yₑ, NaN) == 0
        @test (@inferred evaluate(v, ρ, s, yₑ, NaN)) isa EOSPoint{Float64}

        v32 = E.narrow(v, Float32)
        @test v32 isa EOSTableView{Float32}
        pt32 = evaluate(v32, Float32(ρ), Float32(s), Float32(yₑ), NaN32)
        @test pt32 isa EOSPoint{Float32}
        @test isfinite(pt32.U) && isfinite(pt32.p)
        @test _eval_allocs(v32, Float32(ρ), Float32(s), Float32(yₑ), NaN32) == 0
    end
end

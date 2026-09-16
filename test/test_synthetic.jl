@testset "synthetic" begin
    E = EntropyEOS

    # Block coordinates of `dirty_synthetic_options()`, 1-based, repeated here
    # so the tests fail loudly if the preset ever moves: wiggle on "entropy"
    # over iρ[31,36] × kYₑ[7,9] × jT[11,21]; flatten on "logenergy" over
    # iρ[6,16] × kYₑ[3,5] × jT[4,13]; offset on "entropy" over iρ[1,4] ×
    # kYₑ[1,3] × jT[1,3]; setvalue on "cs2" at (21,16,6) and (22,17,6), and on
    # "gamma" at (21,16,6).

    @testset "stored fields equal the analytic functions exactly" begin
        # `==`, not `≈`: within one language nothing may intervene between the
        # closed form and the stored value -- no reordering, no fused multiply
        # that the generator applies and this test does not.
        opts = E.SyntheticOptions()
        t = E.make_synthetic_table(opts)

        logenergy = field(t, "logenergy")
        entropy = field(t, "entropy")
        logpress = field(t, "logpress")

        rng = StableRNG(12345)
        for _ in 1:200
            iρ = rand(rng, 1:E.nρ(t))
            jT = rand(rng, 1:E.nT(t))
            kYₑ = rand(rng, 1:E.nYₑ(t))
            ρ = density(t, iρ)
            T = temperature(t, jT)
            Yₑ = electron_fraction(t, kYₑ)

            ε = E.synthetic_eps(ρ, T, Yₑ, opts)
            p = E.synthetic_p(ρ, T, Yₑ, opts)
            s = E.synthetic_s(ρ, T, Yₑ, opts)

            @test logenergy[iρ, jT, kYₑ] == log10(ε + opts.energy_shift)
            @test entropy[iρ, jT, kYₑ] == s
            @test logpress[iρ, jT, kYₑ] == log10(p)
        end

        @test energy_shift(t) == opts.energy_shift
    end

    @testset "aux fields equal the analytic functions exactly" begin
        opts = E.SyntheticOptions(; with_aux_fields=true)
        t = E.make_synthetic_table(opts)
        @test field_names(t) == ["logenergy", "entropy", "logpress", "cs2", "gamma", "mu_e"]

        cs2 = field(t, "cs2")
        gamma = field(t, "gamma")
        mu_e = field(t, "mu_e")

        rng = StableRNG(999)
        for _ in 1:200
            iρ = rand(rng, 1:E.nρ(t))
            jT = rand(rng, 1:E.nT(t))
            kYₑ = rand(rng, 1:E.nYₑ(t))
            ρ = density(t, iρ)
            T = temperature(t, jT)
            Yₑ = electron_fraction(t, kYₑ)

            @test cs2[iρ, jT, kYₑ] == E.synthetic_cs2(ρ, T, Yₑ, opts)
            @test gamma[iρ, jT, kYₑ] == 5 / 3
            @test mu_e[iρ, jT, kYₑ] == (T * E.MEV_TO_ERG / E.M_AMU_G) * Yₑ
        end

        # The clean model is comfortably subluminal everywhere; only the
        # stiffened preset below pushes it past 1.
        @test all(0 .< cs2 .< 1)
    end

    @testset "entropy positive, entropy and logenergy strictly increasing in T" begin
        t = E.make_synthetic_table()
        logenergy = field(t, "logenergy")
        entropy = field(t, "entropy")

        all_positive = true
        s_increasing = true
        e_increasing = true
        for kYₑ in 1:E.nYₑ(t), iρ in 1:E.nρ(t)
            all_positive &= all(>(0), @view entropy[iρ, :, kYₑ])
            for jT in 2:E.nT(t)
                s_increasing &= entropy[iρ, jT, kYₑ] > entropy[iρ, jT - 1, kYₑ]
                e_increasing &= logenergy[iρ, jT, kYₑ] > logenergy[iρ, jT - 1, kYₑ]
            end
        end
        @test all_positive
        @test s_increasing
        @test e_increasing
    end

    @testset "thermodynamic identities by finite differences" begin
        # Truncation-limited, not roundoff-limited: the central difference is
        # taken in the uniform log10 coordinate, so its relative error is
        # (Δln x)²/6. On this 120 × 120 grid Δln T = Δln ρ = 3·ln10/119 ≈
        # 0.0580, giving ≈ 5.6e-4 -- and the measured worst case over the
        # sampled interior points is 5.6170e-4, 5.6170e-4 and 5.6138e-4 for the
        # three identities respectively. The 1e-3 bound below therefore leaves
        # a factor ~1.8 of headroom while still catching any genuine
        # inconsistency, which would be O(1) rather than O(Δln x²).
        opts = E.SyntheticOptions(; nρ=120, nT=120)
        t = E.make_synthetic_table(opts)

        logenergy = field(t, "logenergy")
        entropy = field(t, "entropy")
        logpress = field(t, "logpress")

        ln10 = E.ln10(Float64)
        dlog10T = (t.logT[end] - t.logT[begin]) / (E.nT(t) - 1)
        dlog10ρ = (t.logρ[end] - t.logρ[begin]) / (E.nρ(t) - 1)

        # Go through the *stored* representation rather than the analytic
        # shortcuts: that is what the identities are supposed to hold for.
        ε_at(i, j, k) = exp10(logenergy[i, j, k]) - opts.energy_shift
        p_at(i, j, k) = exp10(logpress[i, j, k])

        worst1 = 0.0
        worst2 = 0.0
        worst3 = 0.0
        # Sample interior points only; central differences cannot straddle the
        # boundary rows, and the model has no structure that could hide a bug
        # between the sampled points.
        for kYₑ in 1:3:E.nYₑ(t), jT in 6:11:(E.nT(t) - 5), iρ in 6:11:(E.nρ(t) - 5)
            T_MeV = temperature(t, jT)
            ρ_gcc = density(t, iρ)
            p = p_at(iρ, jT, kYₑ)

            dε_dT = (ε_at(iρ, jT + 1, kYₑ) - ε_at(iρ, jT - 1, kYₑ)) / (2 * dlog10T) / (T_MeV * ln10)
            ds_dT = (entropy[iρ, jT + 1, kYₑ] - entropy[iρ, jT - 1, kYₑ]) / (2 * dlog10T) / (T_MeV * ln10)
            dp_dT = (p_at(iρ, jT + 1, kYₑ) - p_at(iρ, jT - 1, kYₑ)) / (2 * dlog10T) / (T_MeV * ln10)
            dε_dρ = (ε_at(iρ + 1, jT, kYₑ) - ε_at(iρ - 1, jT, kYₑ)) / (2 * dlog10ρ) / (ρ_gcc * ln10)
            ds_dρ = (entropy[iρ + 1, jT, kYₑ] - entropy[iρ - 1, jT, kYₑ]) / (2 * dlog10ρ) / (ρ_gcc * ln10)

            # 1) dε/dT = (kT/m_B) · ds/dT
            rhs1 = (T_MeV * E.MEV_TO_ERG / E.M_AMU_G) * ds_dT
            worst1 = max(worst1, abs(dε_dT - rhs1) / abs(rhs1))

            # 2) ρ² dε/dρ = p - T dp/dT
            worst2 = max(worst2, abs(ρ_gcc^2 * dε_dρ - (p - T_MeV * dp_dT)) / p)

            # 3) Maxwell: ds/dρ = -(dp/dT) · m_B / (ρ² · MeV_to_erg)
            rhs3 = -dp_dT * E.M_AMU_G / (ρ_gcc^2 * E.MEV_TO_ERG)
            worst3 = max(worst3, abs(ds_dρ - rhs3) / abs(rhs3))
        end

        @test worst1 < 1e-3
        @test worst2 < 1e-3
        @test worst3 < 1e-3
    end

    @testset "seeded violation appears exactly at the requested node, nowhere else" begin
        base = E.SyntheticOptions(; nρ=12, nT=10, nYₑ=4)
        clean = E.make_synthetic_table(base)

        iρ, jT, kYₑ = 6, 4, 3
        delta = 7.5
        seeded = E.make_synthetic_table(
            E.SyntheticOptions(; nρ=12, nT=10, nYₑ=4, seed=[E.SeededViolation("entropy", iρ, jT, kYₑ, delta)]),
        )

        expected = copy(field(clean, "entropy"))
        expected[iρ, jT, kYₑ] += delta
        @test field(seeded, "entropy") == expected

        # Untouched fields must be bit-identical.
        @test field(seeded, "logenergy") == field(clean, "logenergy")
        @test field(seeded, "logpress") == field(clean, "logpress")
    end

    @testset "multiple seeded violations are independent and localized" begin
        grid = (nρ=10, nT=8, nYₑ=3)
        clean = E.make_synthetic_table(E.SyntheticOptions(; grid...))
        seeded = E.make_synthetic_table(
            E.SyntheticOptions(;
                grid...,
                seed=[E.SeededViolation("entropy", 2, 2, 1, 3.0), E.SeededViolation("logenergy", 5, 6, 3, -0.25)],
            ),
        )

        expected_s = copy(field(clean, "entropy"))
        expected_s[2, 2, 1] += 3.0
        expected_e = copy(field(clean, "logenergy"))
        expected_e[5, 6, 3] += -0.25

        @test field(seeded, "entropy") == expected_s
        @test field(seeded, "logenergy") == expected_e
        @test field(seeded, "logpress") == field(clean, "logpress")
    end

    @testset "default options are a no-op" begin
        defaults = E.make_synthetic_table()
        # Every defect list spelled out as empty: the point is to run
        # apply_flatten!/apply_wiggle!/apply_offset!/apply_stiffen!/
        # apply_seed!/apply_setvalue! on genuinely empty input and confirm
        # they change nothing.
        explicit_empty = E.make_synthetic_table(
            E.SyntheticOptions(;
                flatten=E.FlattenDefect[],
                wiggle=E.WiggleDefect[],
                offset=E.OffsetDefect[],
                stiffen=E.StiffenDefect[],
                seed=E.SeededViolation[],
                setvalue=E.SetValue[],
            ),
        )

        @test field_names(defaults) == field_names(explicit_empty) == ["logenergy", "entropy", "logpress"]
        for name in field_names(defaults)
            @test field(defaults, name) == field(explicit_empty, name)
        end
        @test !has_field(defaults, "cs2")
        @test !has_field(defaults, "gamma")
        @test !has_field(defaults, "mu_e")
    end

    @testset "determinism" begin
        # No RNG anywhere, so two independent calls must agree bit-for-bit --
        # including for the preset, whose defects are all closed-form.
        for opts in (E.SyntheticOptions(), E.dirty_synthetic_options())
            a = E.make_synthetic_table(opts)
            b = E.make_synthetic_table(opts)
            @test a.logρ == b.logρ && a.logT == b.logT && a.Yₑ == b.Yₑ
            @test field_names(a) == field_names(b)
            for name in field_names(a)
                fa = field(a, name)
                fb = field(b, name)
                # `isequal` rather than `==` so the planted NaNs compare equal.
                @test isequal(fa, fb)
            end
            @test attribute_names(a) == attribute_names(b)
            @test energy_shift(a) == energy_shift(b)
        end
    end

    @testset "dirty preset" begin
        dirty = E.make_synthetic_table(E.dirty_synthetic_options())
        # Same grid and aux fields, no defects, for a fair bit-for-bit
        # comparison.
        clean = E.make_synthetic_table(E.SyntheticOptions(; with_aux_fields=true))

        @testset "default grid, aux fields present" begin
            @test size(dirty) == (40, 30, 10)
            @test has_field(dirty, "cs2")
            @test has_field(dirty, "gamma")
            @test has_field(dirty, "mu_e")
        end

        @testset "planted Inf/NaN land at the exact coordinates" begin
            cs2 = field(dirty, "cs2")
            gamma = field(dirty, "gamma")
            @test cs2[21, 16, 6] == Inf          # +Inf specifically, not -Inf
            @test isnan(cs2[22, 17, 6])
            @test isnan(gamma[21, 16, 6])
            # Nothing else in either field is non-finite.
            @test count(!isfinite, cs2) == 2
            @test count(!isfinite, gamma) == 1
            @test all(isfinite, field(dirty, "mu_e"))
        end

        @testset "offset block entropies are all negative" begin
            @test all(<(0), @view field(dirty, "entropy")[1:4, 1:3, 1:3])
        end

        @testset "wiggle window contains a decreasing adjacent T-pair" begin
            entropy = field(dirty, "entropy")
            found = false
            for kYₑ in 7:9, iρ in 31:36, jT in 11:20
                found |= entropy[iρ, jT + 1, kYₑ] <= entropy[iρ, jT, kYₑ]
            end
            @test found
        end

        @testset "flatten block is constant along its T-range" begin
            logenergy = field(dirty, "logenergy")
            flat = true
            for kYₑ in 3:5, iρ in 6:16
                flat &= all(==(logenergy[iρ, 4, kYₑ]), @view logenergy[iρ, 4:13, kYₑ])
            end
            @test flat
        end

        @testset "no block defect outside its declared block" begin
            # Each point fails at least one of the three range conditions for
            # every block -- a block only bites where iρ, jT and kYₑ are all in
            # range at once -- and none coincides with a setvalue node.
            far_points = [(26, 26, 1), (40, 1, 10), (1, 30, 10), (21, 1, 1), (37, 21, 4), (6, 21, 10)]
            # "logenergy" is excluded: the preset's StiffenDefect is by design
            # not a block defect, and gets its own test below.
            for name in ("entropy", "logpress", "cs2", "gamma", "mu_e"), pt in far_points
                @test field(clean, name)[pt...] == field(dirty, name)[pt...]
            end
        end

        @testset "stiffened corner adds A·(ρ/ρ_c)^α to ε" begin
            ρ_c, α, amp = 1.0e13, 2.0, 1.35e17
            lc = field(clean, "logenergy")
            ld = field(dirty, "logenergy")

            # Away from the other defects (iρ ≥ 21 clears flatten, kYₑ = 1
            # clears wiggle) the dirty logenergy is exactly the clean one with
            # the power-law term added to 10^value -- exactly, because the
            # generator performs precisely these operations.
            matches = true
            for iρ in 21:E.nρ(dirty)
                term = amp * (density(dirty, iρ) / ρ_c)^α
                for jT in 1:E.nT(dirty)
                    matches &= ld[iρ, jT, 1] == log10(exp10(lc[iρ, jT, 1]) + term)
                end
            end
            @test matches

            # Top ρ node: the term is of order c², which is what makes the
            # constructed U superluminal there.
            top_term = amp * (density(dirty, E.nρ(dirty)) / ρ_c)^α
            @test 1e21 < top_term < 2e21

            # Far below ρ_c the ρ^-α falloff makes the term negligible, and at
            # the very first node it vanishes into the last bit entirely.
            @test all(
                abs(ld[iρ, jT, 10] - lc[iρ, jT, 10]) <= 1e-12 * abs(lc[iρ, jT, 10]) for iρ in 1:10,
                jT in 21:E.nT(dirty)
            )
            @test ld[1, :, 10] == lc[1, :, 10]
        end
    end

    @testset "defect index validation" begin
        grid = (nρ=8, nT=6, nYₑ=4)
        @test_throws ArgumentError E.make_synthetic_table(
            E.SyntheticOptions(; grid..., flatten=[E.FlattenDefect("entropy", 1, 9, 1, 2, 1, 2)]),
        )
        @test_throws ArgumentError E.make_synthetic_table(
            E.SyntheticOptions(; grid..., wiggle=[E.WiggleDefect("entropy", 1, 2, 1, 5, 1, 2, 1.0, 4.0)]),
        )
        @test_throws ArgumentError E.make_synthetic_table(
            E.SyntheticOptions(; grid..., offset=[E.OffsetDefect("entropy", 2, 1, 1, 2, 1, 2, 1.0)]),  # iρ0 > iρ1
        )
        @test_throws ArgumentError E.make_synthetic_table(
            E.SyntheticOptions(; grid..., setvalue=[E.SetValue("entropy", 1, 7, 1, 0.0)]),
        )
        @test_throws ArgumentError E.make_synthetic_table(
            E.SyntheticOptions(; grid..., stiffen=[E.StiffenDefect("logenergy", 0.0, 2.0, 1.0)]),
        )
        # Naming a field the table does not have.
        @test_throws KeyError E.make_synthetic_table(
            E.SyntheticOptions(; grid..., seed=[E.SeededViolation("cs2", 1, 1, 1, 1.0)]),
        )
    end
end

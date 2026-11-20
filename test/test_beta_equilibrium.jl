# Cold β-equilibrium on the synthetic table, with an analytic `munu` added so
# that every root is known exactly. The real tables are covered in
# test_real_tables.jl.

@testset "beta_equilibrium" begin
    E = EntropyEOS
    tbl = make_synthetic_table(SyntheticOptions())
    eos = E.build_eos(tbl)
    v = EOSTableView(eos)
    lκ = log10(v.κ)
    nρ, nT, nY = size(tbl)
    Yₑ = tbl.Yₑ

    set_munu!(f) = add_field!(tbl, "munu", [f(tbl.logρ[i], Yₑ[k]) for i in 1:nρ, j in 1:nT, k in 1:nY])
    # ρ★ at a raw-table log-density.
    ρ★_at(x) = exp10(x + lκ)

    @testset "linear munu: exact roots" begin
        # Bilinear interpolation reproduces a function linear in both log ρ and
        # Yₑ exactly, so the root is Y₀(x) to rounding, on and between nodes.
        Y₀(x) = 0.1 + 0.03 * (x - 5)
        set_munu!((x, y) -> 100 * (y - Y₀(x)))
        for x in (tbl.logρ[1], tbl.logρ[7], 0.5 * (tbl.logρ[7] + tbl.logρ[8]), 11.3, tbl.logρ[end])
            r = beta_equilibrium(tbl, eos, ρ★_at(x))
            @test r isa BetaEquilibrium{Float64}
            @test r.yₑ ≈ Y₀(x) rtol = 1e-12
            @test r.nroots == 1
            @test r.flags & FLAG_CLAMP_YE == 0
        end
    end

    @testset "the state is the fitted EOS's cold edge" begin
        ρ★ = ρ★_at(10.0)
        r = beta_equilibrium(tbl, eos, ρ★)
        @test r.s == srange(v, ρ★, r.yₑ).s_min
        @test r.point == evaluate(v, ρ★, r.s, r.yₑ, v.u_lo)
        @test r.point.T_MeV ≈ exp10(tbl.logT[1]) rtol = 1e-6
        @test r.flags == r.point.flags
    end

    @testset "roots outside the Yₑ range are clamped" begin
        set_munu!((x, y) -> 100 * (y - 0.01))
        r = beta_equilibrium(tbl, eos, ρ★_at(10.0))
        @test r.yₑ == Yₑ[1]
        @test r.flags & FLAG_CLAMP_YE != 0
        @test r.nroots == 0

        set_munu!((x, y) -> 100 * (y - 0.7))
        r = beta_equilibrium(tbl, eos, ρ★_at(10.0))
        @test r.yₑ == Yₑ[end]
        @test r.flags & FLAG_CLAMP_YE != 0
        @test r.nroots == 0

        # A root exactly on the upper edge is not clamped.
        set_munu!((x, y) -> 100 * (y - Yₑ[end]))
        r = beta_equilibrium(tbl, eos, ρ★_at(10.0))
        @test r.yₑ == Yₑ[end]
        @test r.flags & FLAG_CLAMP_YE == 0
    end

    @testset "several roots: the global minimum of ∫ munu dYₑ" begin
        # Zeros on Yₑ nodes, so the piecewise-linear interpolant has them
        # exactly. munu rises through zero at a and c, falls at b.
        Φ_ref(f, k) = sum(0.5 * (Yₑ[m + 1] - Yₑ[m]) * (f(Yₑ[m]) + f(Yₑ[m + 1])) for m in 1:(k - 1); init=0.0)
        for (ia, ib, ic) in ((2, 4, 9), (2, 7, 9))
            a, b, c = Yₑ[ia], Yₑ[ib], Yₑ[ic]
            f(y) = 1000 * (y - a) * (y - b) * (y - c)
            set_munu!((x, y) -> f(y))
            expected = Φ_ref(f, ic) < Φ_ref(f, ia) ? c : a
            r = beta_equilibrium(tbl, eos, ρ★_at(10.0))
            @test r.nroots == 2
            @test r.yₑ == expected
            @test r.flags & FLAG_CLAMP_YE == 0
        end
        # Both orders are exercised: b near a favours c, b near c favours a.
        f1(y) = (y - Yₑ[2]) * (y - Yₑ[4]) * (y - Yₑ[9])
        f2(y) = (y - Yₑ[2]) * (y - Yₑ[7]) * (y - Yₑ[9])
        @test Φ_ref(f1, 9) < Φ_ref(f1, 2)
        @test Φ_ref(f2, 9) > Φ_ref(f2, 2)
    end

    @testset "densities outside the table" begin
        Y₀(x) = 0.1 + 0.03 * (x - 5)
        set_munu!((x, y) -> 100 * (y - Y₀(x)))
        lo = beta_equilibrium(tbl, eos, exp10(v.x_lo - 1))
        @test lo.yₑ ≈ Y₀(tbl.logρ[1]) rtol = 1e-12
        @test lo.flags & FLAG_EXT_ρ_LOW != 0
        hi = beta_equilibrium(tbl, eos, exp10(v.x_hi + 0.01))
        @test hi.yₑ ≈ Y₀(tbl.logρ[end]) rtol = 1e-12
        @test hi.flags & FLAG_OOB_ρ_HIGH != 0
    end

    @testset "bad inputs" begin
        plain = make_synthetic_table(SyntheticOptions())
        @test_throws ArgumentError beta_equilibrium(plain, eos, 1e10)

        set_munu!((x, y) -> y - 0.3)
        cropped = crop_table(tbl, 2:nρ, 1:nT, 1:nY)
        @test_throws ArgumentError beta_equilibrium(cropped, eos, 1e10)
        cropped = crop_table(tbl, 1:nρ, 1:nT, 1:(nY - 1))
        @test_throws ArgumentError beta_equilibrium(cropped, eos, 1e10)

        set_munu!((x, y) -> x > 9 ? NaN : y - 0.3)
        @test_throws ArgumentError beta_equilibrium(tbl, eos, ρ★_at(12.0))
    end
end

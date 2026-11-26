# The cold β-equilibrium slice on the synthetic table, with analytic chemical
# potentials added so that every root, and one phase transition, is known
# exactly. The real tables are covered in test_real_tables.jl.

@testset "beta_equilibrium" begin
    E = EntropyEOS
    tbl = make_synthetic_table(SyntheticOptions())
    eos = E.build_eos(tbl)
    v = EOSTableView(eos)
    lκ = log10(v.κ)
    nρ, nT, nY = size(tbl)
    Yₑ = tbl.Yₑ
    cF = v.conv_t / v.κ          # MeV per baryon in U units

    # μ_ν = mu_e − muhat (+ the offset); muhat = 0 throughout.
    function set_mu!(f; munu=nothing)
        add_field!(tbl, "mu_e", [f(tbl.logρ[i], Yₑ[k]) for i in 1:nρ, j in 1:nT, k in 1:nY])
        add_field!(tbl, "muhat", zeros(nρ, nT, nY))
        mn = munu === nothing ? f : munu
        add_field!(tbl, "munu", [mn(tbl.logρ[i], Yₑ[k]) for i in 1:nρ, j in 1:nT, k in 1:nY])
        return nothing
    end
    monotone(sl) = issorted(sl.ρ★) && issorted(sl.p) && issorted(sl.g)

    @testset "one branch: exact roots, lookups" begin
        # Bilinear interpolation reproduces a function linear in log ρ and Yₑ,
        # so the root is Y₀(x) to rounding at every density.
        Y₀(x) = 0.1 + 0.03 * (x - 5)
        set_mu!((x, y) -> 100 * (y - Y₀(x)))
        sl = cold_beta_slice(tbl, eos; energy_offset=0.0)
        @test sl.munu_offset == 0
        @test sl.munu_mismatch == 0
        @test isempty(sl.jumps)
        @test monotone(sl)
        @test sl.ρ★[1] ≈ exp10(v.x_lo) && sl.ρ★[end] ≈ exp10(v.x_hi)
        @test all(i -> isapprox(sl.yₑ[i], Y₀(log10(sl.ρ★[i]) - lκ); rtol=1e-12), eachindex(sl.ρ★))
        @test !any(sl.clamped)

        for x in (6.0, 9.37, 12.5, 14.9)
            r = beta_equilibrium(sl; ρ★=exp10(x + lκ))
            @test r isa BetaEquilibrium{Float64}
            @test r.ρ★ ≈ exp10(x + lκ) rtol = 1e-14
            @test r.yₑ ≈ Y₀(x) rtol = 1e-12
            @test r.s == srange(v, r.ρ★, r.yₑ).s_min
            @test r.point == evaluate(v, r.ρ★, r.s, r.yₑ, v.u_lo)
            @test r.point.T_MeV ≈ exp10(tbl.logT[1]) rtol = 1e-6
            @test r.g ≈ 1 + r.point.U - r.point.T̂ * r.s + r.point.p / r.ρ★ rtol = 1e-14
            @test r.flags & FLAG_CLAMP_YE == 0
            # The same state by pressure and by Gibbs energy.
            rp = beta_equilibrium(sl; p=r.point.p)
            @test rp.ρ★ ≈ r.ρ★ rtol = 1e-10
            @test rp.point.p ≈ r.point.p rtol = 1e-10
            rg = beta_equilibrium(sl; g=r.g)
            @test rg.ρ★ ≈ r.ρ★ rtol = 1e-8
        end
        # At a vertex, the vertex itself.
        r = beta_equilibrium(sl; p=sl.p[17])
        @test r.ρ★ == sl.ρ★[17] && r.yₑ == sl.yₑ[17]

        @test_throws ArgumentError beta_equilibrium(sl)
        @test_throws ArgumentError beta_equilibrium(sl; ρ★=1e10, p=1.0)
        @test_throws ArgumentError beta_equilibrium(sl; p=sl.p[end] * 1.01)
        @test_throws ArgumentError beta_equilibrium(sl; ρ★=sl.ρ★[1] / 2)
    end

    @testset "the energy offset enters g" begin
        set_mu!((x, y) -> 100 * (y - 0.3))
        sl0 = cold_beta_slice(tbl, eos; energy_offset=0.0)
        sl1 = cold_beta_slice(tbl, eos; energy_offset=2.0)
        @test sl1.energy_offset == 2.0
        @test sl1.ρ★ == sl0.ρ★
        @test sl1.g ≈ sl0.g .+ 2.0 * cF * 0.3 rtol = 1e-14
    end

    @testset "roots outside the Yₑ range are clamped" begin
        set_mu!((x, y) -> 100 * (y - 0.01))
        sl = cold_beta_slice(tbl, eos; energy_offset=0.0)
        @test all(sl.clamped) && all(==(Yₑ[1]), sl.yₑ)
        @test beta_equilibrium(sl; ρ★=1e10 * v.κ).flags & FLAG_CLAMP_YE != 0
        set_mu!((x, y) -> 100 * (y - 0.7))
        sl = cold_beta_slice(tbl, eos; energy_offset=0.0)
        @test all(sl.clamped) && all(==(Yₑ[end]), sl.yₑ)
        # A root exactly on the upper edge is not clamped.
        set_mu!((x, y) -> 100 * (y - Yₑ[end]))
        sl = cold_beta_slice(tbl, eos; energy_offset=0.0)
        @test !any(sl.clamped) && all(==(Yₑ[end]), sl.yₑ)
    end

    @testset "munu offset and mismatch" begin
        Q = 1.29333
        # The table's munu subtracts Q above 10^9, and not below: the offset is
        # the majority's, and the rest is reported.
        set_mu!((x, y) -> 100 * (y - 0.3); munu=(x, y) -> 100 * (y - 0.3) - (x >= 9 ? Q : 0.0))
        sl = cold_beta_slice(tbl, eos; energy_offset=0.0)
        frac = count(>=(9), tbl.logρ) / nρ
        @test sl.munu_offset ≈ -Q
        @test sl.munu_mismatch ≈ 1 - frac
        # Followed: mu_e − muhat − Q = 0, so Yₑ = 0.3 + Q/100.
        @test all(≈(0.3 + Q / 100; rtol=1e-12), sl.yₑ)
        @test cold_beta_slice(tbl, eos; munu_offset=0.0, energy_offset=0.0).yₑ[1] ≈ 0.3 rtol = 1e-12
        # Neither 0 nor −Q: refused.
        set_mu!((x, y) -> 100 * (y - 0.3); munu=(x, y) -> 100 * (y - 0.3) + 0.5)
        @test_throws ArgumentError cold_beta_slice(tbl, eos; energy_offset=0.0)
    end

    @testset "muhat's convention from the ideal gas, not munu" begin
        # Free nucleons whose ratio makes muhat = 0 include Q, or lack it: the
        # offset follows them, whatever munu says (O'Connor–Ott LS220's munu
        # subtracts a further Q above 10⁸ g/cm³).
        Q = E.Q_NP_MEV
        mₙ = E.M_NEUTRON_G * E.C_LIGHT_CM_S^2 / E.MEV_TO_ERG
        r = (mₙ / (mₙ - Q))^1.5
        set_mu!((x, y) -> 100 * (y - 0.3); munu=(x, y) -> 100 * (y - 0.3) - Q)
        for (lacks_Q, want) in ((false, 0.0), (true, -Q))
            tg = deepcopy(tbl)
            T = [exp10(tg.logT[j]) for i in 1:nρ, j in 1:nT, k in 1:nY]
            add_field!(tg, "Xp", fill(0.3, nρ, nT, nY))
            add_field!(tg, "Xn", 0.3 .* r .* (lacks_Q ? ones(nρ, nT, nY) : exp.(-Q ./ T)))
            sl = cold_beta_slice(tg, eos; energy_offset=0.0)
            @test sl.munu_offset == want
            @test sl.munu_mismatch ≈ (lacks_Q ? 0 : 1)
            @test all(≈(0.3 - want / 100; rtol=1e-12), sl.yₑ)
        end
        # Muhat that is neither: refused.
        tg = deepcopy(tbl)
        add_field!(tg, "Xp", fill(0.3, nρ, nT, nY))
        add_field!(tg, "Xn", fill(0.3, nρ, nT, nY))
        add_field!(tg, "muhat", fill(0.5, nρ, nT, nY))
        @test_throws ArgumentError cold_beta_slice(tg, eos; energy_offset=0.0)
    end

    @testset "the electron-mass check" begin
        # The synthetic ε = 1.5 (1 + Yₑ) kT / m_B, so m_B ∂ε/∂Yₑ = 1.5 T. A
        # table whose mu_e carries the electron's rest mass on top passes.
        T = exp10(tbl.logT[1])
        mₑ = E.M_ELECTRON_MEV
        set_mu!((x, y) -> 1.5 * T + mₑ)
        sl = cold_beta_slice(tbl, eos)
        @test sl.energy_offset ≈ sl.munu_offset + mₑ
        @test all(sl.clamped)                 # μ_ν > 0 everywhere
        set_mu!((x, y) -> 1.5 * T)
        @test_throws ArgumentError cold_beta_slice(tbl, eos)
        @test cold_beta_slice(tbl, eos; energy_offset=0.0) isa ColdBetaSlice
    end

    @testset "a phase transition where two branches' g cross" begin
        # μ_ν rises through zero at Yₑ = a and c (falls at b), at every density,
        # so there are two branches of constant composition. The energy offset
        # is chosen to make their g equal at the pressure of one state; the
        # transition must be exactly there, the low-density phase the one of
        # larger Yₑ (at given p it is less dense, so dg/dp = 1/ρ★ is larger).
        a, b, c = Yₑ[3], Yₑ[5], Yₑ[8]
        set_mu!((x, y) -> 1000 * (y - a) * (y - b) * (y - c))
        ctx(cE) = E.BranchCtx(v, Float64.(tbl.logρ), Float64.(Yₑ), zeros(1, 1), lκ, cF * cE)
        cand(x★, y, cE) = E.beta_candidate(ctx(cE), x★, y, false)
        x★a = 10.0 + lκ
        A = cand(x★a, a, 0.0)
        lo, hi = x★a - 1, x★a                  # the Yₑ = c state at A's pressure
        for _ in 1:200
            mid = (lo + hi) / 2
            cand(mid, c, 0.0).pt.p < A.pt.p ? (lo = mid) : (hi = mid)
        end
        C = cand((lo + hi) / 2, c, 0.0)
        cE = -(C.g - A.g) / (cF * (c - a))
        sl = cold_beta_slice(tbl, eos; energy_offset=cE)
        @test monotone(sl)
        @test length(sl.jumps) == 1
        j = only(sl.jumps)
        @test j.p ≈ A.pt.p rtol = 1e-8
        @test j.g_mismatch == 0
        @test j.yₑ_lo == c && j.yₑ_hi == a
        @test j.ρ★_lo ≈ C.ρ★ rtol = 1e-8
        @test j.ρ★_hi ≈ A.ρ★ rtol = 1e-8
        k = findfirst(sl.jump_after)
        @test sl.p[k] == sl.p[k + 1] == j.p
        @test sl.g[k] == sl.g[k + 1]
        @test all(==(c), sl.yₑ[1:k]) && all(==(a), sl.yₑ[(k + 1):end])
        # Lookups: below the jump the low-density phase, from it on the other;
        # no state inside the gap.
        @test beta_equilibrium(sl; p=0.999 * j.p).yₑ == c
        @test beta_equilibrium(sl; p=j.p).yₑ == a
        @test beta_equilibrium(sl; p=1.001 * j.p).yₑ == a
        @test_throws ArgumentError beta_equilibrium(sl; ρ★=sqrt(j.ρ★_lo * j.ρ★_hi))
    end

    @testset "bad inputs" begin
        set_mu!((x, y) -> y - 0.3)
        cropped = crop_table(tbl, 2:nρ, 1:nT, 1:nY)
        @test_throws ArgumentError cold_beta_slice(cropped, eos; energy_offset=0.0)
        cropped = crop_table(tbl, 1:nρ, 1:nT, 1:(nY - 1))
        @test_throws ArgumentError cold_beta_slice(cropped, eos; energy_offset=0.0)
        @test_throws ArgumentError cold_beta_slice(tbl, eos; refine=0, energy_offset=0.0)

        plain = make_synthetic_table(SyntheticOptions())
        @test_throws ArgumentError cold_beta_slice(plain, eos; energy_offset=0.0)
        add_field!(plain, "mu_e", field(tbl, "mu_e"))
        add_field!(plain, "muhat", field(tbl, "muhat"))
        @test_throws ArgumentError cold_beta_slice(plain, eos; energy_offset=0.0)   # no munu
        @test cold_beta_slice(plain, eos; munu_offset=0.0, energy_offset=0.0).munu_mismatch === NaN

        add_field!(plain, "mu_e", fill(NaN, size(plain)))
        @test_throws ArgumentError cold_beta_slice(plain, eos; munu_offset=0.0, energy_offset=0.0)
    end
end

_p2c_probe(v, ρ, s, y, w, B², c, u) = EntropyEOS.prim2con(v, ρ, s, y, w, B², c, u).τ
function _p2c_allocs(v, ρ, s, y, w, B², c, u)
    _p2c_probe(v, ρ, s, y, w, B², c, u)
    return @allocated _p2c_probe(v, ρ, s, y, w, B², c, u)
end

@testset "prim2con" begin
    E = EntropyEOS
    tbl = make_synthetic_table()
    eos = E.build_eos(tbl)
    v = EOSTableView(eos)
    ρ = exp10(0.5 * (v.x_lo + v.x_hi))
    yₑ = 0.4
    sr = srange(v, ρ, yₑ)
    s = 0.5 * (sr.s_min + sr.s_max)
    u0 = evaluate(v, ρ, s, yₑ, NaN).u_solved

    @testset "τ equals E - D" begin
        # The cancellation-free form is an algebraic identity, not an
        # approximation, so at moderate rapidity it must agree with the naive
        # difference to roundoff.
        for w in (0.1, 0.8, 2.0), (B², cos_vB) in ((0.0, 0.0), (0.1ρ, 0.3), (2.0ρ, -0.7))
            pt = evaluate(v, ρ, s, yₑ, u0)
            c = prim2con(v, ρ, s, yₑ, w, B², cos_vB, u0)
            W, vel = cosh(w), tanh(w)
            v_par = vel * cos_vB
            z = ρ * pt.h * W^2
            E_naive = z - pt.p + 0.5B² * (1 + vel^2) - 0.5B² * v_par^2
            @test c.τ ≈ E_naive - c.D rtol = 1e-13
        end
    end

    @testset "components" begin
        w, B², cos_vB = 0.8, 0.1ρ, 0.3
        pt = evaluate(v, ρ, s, yₑ, u0)
        c = prim2con(v, ρ, s, yₑ, w, B², cos_vB, u0)
        W, vel = cosh(w), tanh(w)
        z = ρ * pt.h * W^2
        @test c.D ≈ ρ * W
        @test c.D_Y ≈ c.D * yₑ
        @test c.B² == B²
        # The field's inertia cancels along B and survives only across it.
        @test c.S_par ≈ z * vel * cos_vB
        @test c.S_perp ≈ (z + B²) * vel * sqrt(1 - cos_vB^2)
        @test all(isfinite, (c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²))
    end

    @testset "degenerate cases" begin
        # At rest there is no momentum at all, and the field direction cannot
        # matter.
        for cos_vB in (-1.0, 0.0, 0.5, 1.0)
            c = prim2con(v, ρ, s, yₑ, 0.0, 0.1ρ, cos_vB, u0)
            @test c.S_par == 0
            @test c.S_perp == 0
            @test c.D ≈ ρ
        end
        # With no field, cos_vB is likewise irrelevant.
        a = prim2con(v, ρ, s, yₑ, 0.8, 0.0, 0.3, u0)
        b = prim2con(v, ρ, s, yₑ, 0.8, 0.0, -0.9, u0)
        @test a.τ == b.τ
        @test hypot(a.S_par, a.S_perp) ≈ hypot(b.S_par, b.S_perp)
    end

    @testset "cold flow keeps its digits" begin
        # As w → 0 the naive E - D cancels two terms of order D down to a
        # result of order τ. The cancellation-free form must stay accurate and
        # positive there.
        pt = evaluate(v, ρ, s, yₑ, u0)
        for w in (1e-4, 1e-6, 1e-8, 1e-10)
            c = prim2con(v, ρ, s, yₑ, w, 0.0, 0.0, u0)
            @test c.τ > 0
            @test isfinite(c.τ)
            # τ → ρU as w → 0, which is the rest-frame internal energy. The
            # leading correction is the kinetic term 2D·sinh²(w/2) ≈ ρw²/2,
            # i.e. a relative w²/(2U), so the bound has to scale with w.
            @test c.τ ≈ ρ * pt.U rtol = w^2 / pt.U + 1e-12
        end
    end

    @testset "3-vector form" begin
        # With no field all momentum is perpendicular, and |S| must match.
        out, S = prim2con(v, ρ, s, yₑ, 0.8, SVector(1.0, 0.0, 0.0), SVector(0.0, 0.0, 0.0), u0)
        @test out.S_par == 0
        @test norm(S) ≈ out.S_perp
        @test S ≈ SVector(out.S_perp, 0.0, 0.0)

        # Velocity along the field: all momentum parallel, none across.
        out, S = prim2con(v, ρ, s, yₑ, 0.8, SVector(0.0, 0.0, 1.0), SVector(0.0, 0.0, 3.0), u0)
        @test out.S_perp ≈ 0 atol = 1e-8
        @test norm(S) ≈ abs(out.S_par) rtol = 1e-12

        # Oblique: the reassembled vector must reproduce both projections and
        # the scalar form's B².
        vd = SVector(0.6, 0.8, 0.0)
        B = SVector(0.0, 2.0, 0.0)
        out, S = prim2con(v, ρ, s, yₑ, 0.8, vd, B, u0)
        b = B / norm(B)
        @test out.B² ≈ 4.0
        @test dot(S, b) ≈ out.S_par rtol = 1e-12
        @test norm(S - dot(S, b) * b) ≈ out.S_perp rtol = 1e-12
        scalar = prim2con(v, ρ, s, yₑ, 0.8, 4.0, dot(vd, b), u0)
        @test scalar.τ ≈ out.τ
        @test scalar.S_par ≈ out.S_par
    end

    @testset "allocation-free and type-generic" begin
        @test _p2c_allocs(v, ρ, s, yₑ, 0.8, 0.1ρ, 0.3, u0) == 0
        @test (@inferred prim2con(v, ρ, s, yₑ, 0.8, 0.1ρ, 0.3, u0)) isa Prim2ConOut{Float64}
        v32 = E.narrow(v, Float32)
        c32 = prim2con(v32, Float32(ρ), Float32(s), Float32(yₑ), 0.8f0, Float32(0.1ρ), 0.3f0, NaN32)
        @test c32 isa Prim2ConOut{Float32}
        @test all(isfinite, (c32.D, c32.τ, c32.S_par, c32.S_perp))
    end
end

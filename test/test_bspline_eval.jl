# Allocation probes are top-level functions taking concrete arguments. A closure
# defined inside a @testset, or a varargs helper that splats, boxes its
# arguments on x86-64 and the measurement then reports the harness rather than
# the kernel -- 16 bytes, reproducible under Rosetta and invisible on aarch64.
_bspline_probe(v, x, u, y) = EntropyEOS.bspline_eval3(v, x, u, y).f
function _bspline_allocs(v, x, u, y)
    _bspline_probe(v, x, u, y)
    return @allocated _bspline_probe(v, x, u, y)
end

@testset "bspline_eval" begin
    E = EntropyEOS

    @testset "basis quadruples" begin
        # A B-spline reproduces constants, so the value quadruple sums to one
        # and both derivative quadruples sum to zero, at every local coordinate.
        for t in (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
            b = E.bspline_basis(t)
            d = E.bspline_dbasis(t)
            h = E.bspline_d2basis(t)
            @test b.b0 + b.b1 + b.b2 + b.b3 ≈ 1 atol = 1e-15
            @test d.b0 + d.b1 + d.b2 + d.b3 ≈ 0 atol = 1e-15
            @test h.b0 + h.b1 + h.b2 + h.b3 ≈ 0 atol = 1e-15
        end
    end

    @testset "cell location" begin
        n, x0, h = 9, -3.0, 0.7
        # Interior points land in the expected cell with t in [0,1).
        for i in 0:(n - 2)
            c = E.bspline_cell(x0 + (i + 0.25) * h, x0, h, n)
            @test c.i == i + 1
            @test 0 <= c.t < 1
        end
        # Out of range clamps to the boundary cell and extrapolates t, which is
        # what makes the boundary cubic continue smoothly.
        @test E.bspline_cell(x0 - 100h, x0, h, n).i == 1
        @test E.bspline_cell(x0 - 100h, x0, h, n).t < 0
        @test E.bspline_cell(x0 + 100h, x0, h, n).i == n - 1
        @test E.bspline_cell(x0 + 100h, x0, h, n).t > 1
        # NaN must clamp rather than throw: floor(Int, NaN) would be an
        # InexactError, and an exception path cannot exist in a GPU kernel.
        @test E.bspline_cell(NaN, x0, h, n).i in 1:(n - 1)
        # The top cell must reach exactly the last coefficient and no further.
        @test E.bspline_cell(x0 + (n - 1) * h, x0, h, n).i + 3 == n + 2
    end

    @testset "constant reproduction" begin
        c = ones(Float64, 9, 8, 7)
        v = BsplineView3(c, 0.0, 0.5, 1.0, 0.25, 0.1, 0.05)
        @test E.npoints_x(v) == 7 && E.npoints_u(v) == 6 && E.npoints_y(v) == 5
        e = bspline_eval3(v, 1.3, 1.4, 0.22)
        @test e.f ≈ 1 atol = 1e-15
        @test e.fx ≈ 0 atol = 1e-13
        @test e.fu ≈ 0 atol = 1e-13
        @test e.fy ≈ 0 atol = 1e-13
        @test e.fxx ≈ 0 atol = 1e-12
        @test e.fxu ≈ 0 atol = 1e-12
        @test e.fuu ≈ 0 atol = 1e-12
    end

    @testset "linear reproduction pins the index convention" begin
        # This basis reproduces linear functions exactly when the coefficients
        # are sampled at the Greville abscissae, which are x0 + (i-1)*h for a
        # 0-based coefficient index i -- so x0 + (i-2)*h in 1-based Julia. Any
        # off-by-one in the cell index or the contraction breaks this test, and
        # nothing else in the package pins it as sharply.
        x0, hx, u0, hu, y0, hy = -3.0, 0.7, 1.0, 0.25, 0.1, 0.05
        nx, nu, ny = 9, 8, 7
        α, βx, βu, βy = 2.5, -1.3, 0.8, 4.0
        c = Array{Float64,3}(undef, nx + 2, nu + 2, ny + 2)
        for k in 1:(ny + 2), j in 1:(nu + 2), i in 1:(nx + 2)
            c[i, j, k] = α + βx * (x0 + (i - 2) * hx) + βu * (u0 + (j - 2) * hu) + βy * (y0 + (k - 2) * hy)
        end
        v = BsplineView3(c, x0, hx, u0, hu, y0, hy)
        for (x, u, y) in ((0.0, 1.4, 0.22), (-2.1, 1.9, 0.31), (1.7, 1.05, 0.12))
            e = bspline_eval3(v, x, u, y)
            @test e.f ≈ α + βx * x + βu * u + βy * y atol = 1e-12
            @test e.fx ≈ βx atol = 1e-12
            @test e.fu ≈ βu atol = 1e-12
            @test e.fy ≈ βy atol = 1e-12
            @test e.fxx ≈ 0 atol = 1e-11
            @test e.fxu ≈ 0 atol = 1e-11
            @test e.fuu ≈ 0 atol = 1e-11
        end
    end

    @testset "1D evaluation agrees with the 3D contraction" begin
        # A 3D spline whose coefficients vary only along x must reproduce the
        # corresponding 1D spline exactly, including derivatives.
        x0, h, n = 0.3, 0.4, 10
        c1 = [sin(0.7 * i) for i in 1:(n + 2)]
        v1 = BsplineView1(c1, x0, h)
        c3 = Array{Float64,3}(undef, n + 2, 6, 6)
        for k in axes(c3, 3), j in axes(c3, 2), i in axes(c3, 1)
            c3[i, j, k] = c1[i]
        end
        v3 = BsplineView3(c3, x0, h, 0.0, 1.0, 0.0, 1.0)
        for x in (0.5, 1.1, 2.6, 3.9)
            e1 = bspline_eval1(v1, x)
            e3 = bspline_eval3(v3, x, 1.5, 1.5)
            @test e1.f ≈ e3.f rtol = 1e-14
            @test e1.fx ≈ e3.fx rtol = 1e-13
            @test e1.fxx ≈ e3.fxx rtol = 1e-12
        end
    end

    @testset "derivatives match finite differences of the spline itself" begin
        x0, hx, u0, hu, y0, hy = -1.0, 0.3, 0.5, 0.2, 0.05, 0.04
        nx, nu, ny = 12, 11, 9
        c = [cos(0.3i) * sin(0.4j) + 0.1 * k for i in 1:(nx + 2), j in 1:(nu + 2), k in 1:(ny + 2)]
        v = BsplineView3(c, x0, hx, u0, hu, y0, hy)
        f(x, u, y) = bspline_eval3(v, x, u, y).f
        x, u, y = 0.4, 1.1, 0.19
        δ = 1e-5
        e = bspline_eval3(v, x, u, y)
        @test e.fx ≈ (f(x + δ, u, y) - f(x - δ, u, y)) / 2δ rtol = 1e-6
        @test e.fu ≈ (f(x, u + δ, y) - f(x, u - δ, y)) / 2δ rtol = 1e-6
        @test e.fy ≈ (f(x, u, y + δ) - f(x, u, y - δ)) / 2δ rtol = 1e-6
        @test e.fxx ≈ (f(x + δ, u, y) - 2f(x, u, y) + f(x - δ, u, y)) / δ^2 rtol = 1e-4
        @test e.fuu ≈ (f(x, u + δ, y) - 2f(x, u, y) + f(x, u - δ, y)) / δ^2 rtol = 1e-4
        @test e.fxu ≈
            (f(x + δ, u + δ, y) - f(x + δ, u - δ, y) - f(x - δ, u + δ, y) + f(x - δ, u - δ, y)) / 4δ^2 rtol = 1e-4
    end

    @testset "allocation-free and type-generic" begin
        c = ones(Float64, 9, 8, 7)
        v = BsplineView3(c, 0.0, 0.5, 1.0, 0.25, 0.1, 0.05)
        @test_noallocs _bspline_allocs(v, 1.3, 1.4, 0.22)
        @test (@inferred bspline_eval3(v, 1.3, 1.4, 0.22)) isa BsplineEval3{Float64}

        c32 = ones(Float32, 9, 8, 7)
        v32 = BsplineView3(c32, 0.0f0, 0.5f0, 1.0f0, 0.25f0, 0.1f0, 0.05f0)
        @test (@inferred bspline_eval3(v32, 1.3f0, 1.4f0, 0.22f0)) isa BsplineEval3{Float32}
        @test_noallocs _bspline_allocs(v32, 1.3f0, 1.4f0, 0.22f0)
    end
end

@testset "bspline_fit" begin
    E = EntropyEOS

    # Mixed absolute/relative error, as the C++ tests use throughout: several
    # reference values here legitimately pass through zero, where a pure
    # relative bound measures only how close a random sample landed to a root.
    rel_err(got, ref, floor = 1e-12) = abs(got - ref) / max(abs(ref), floor)

    # LAPACK's band storage is *not* this package's: kl+ku+1 rows plus kl rows
    # of pivot workspace, column-major, A[i,j] at AB[kl+ku+1+i-j, j]. Building
    # it here rather than reusing BandedLU's is the whole point -- the third
    # opinion has to be independent of the code under test.
    function lapack_band(A::Matrix{Float64}, kl::Int, ku::Int)
        n = size(A, 1)
        AB = zeros(Float64, 2kl + ku + 1, n)
        for j in 1:n, i in max(1, j - ku):min(n, j + kl)
            AB[kl + ku + 1 + i - j, j] = A[i, j]
        end
        return AB
    end

    @testset "BandedLU construction and access" begin
        lu = E.BandedLU{Float64}(6)
        @test size(lu) == (6, 6)
        @test eltype(lu) == Float64
        @test lu[3, 3] == 0                       # an unset entry inside the band reads as zero
        lu[3, 3] = 2.5
        @test lu[3, 3] == 2.5
        lu[1, 5] = -1.0                           # the corners of the band are addressable
        lu[5, 1] = -2.0
        @test lu[1, 5] == -1.0 && lu[5, 1] == -2.0

        @test_throws ArgumentError E.BandedLU{Float64}(0)
        @test_throws BoundsError lu[0, 1]
        @test_throws BoundsError lu[1, 7]
        @test_throws ArgumentError lu[1, 6]       # in the matrix, outside the band
        @test_throws ArgumentError lu[6, 1]
        @test_throws ArgumentError E.solve!(lu, zeros(6))   # not factored yet

        for i in 1:6
            lu[i, i] = 10.0
        end
        E.factor!(lu)
        @test_throws ArgumentError E.factor!(lu)
        @test_throws ArgumentError (lu[3, 3] = 1.0)         # storage now holds L and U
        @test_throws ArgumentError E.solve!(lu, zeros(5))   # wrong right-hand side length

        # A structurally singular matrix (an all-zero column) must be rejected
        # rather than produce Infs.
        sing = E.BandedLU{Float64}(5)
        for i in 1:5
            sing[i, i] = i == 3 ? 0.0 : 1.0
        end
        @test_throws ErrorException E.factor!(sing)
    end

    @testset "BandedLU against dense and LAPACK" begin
        rng = StableRNG(20260825)
        kl = ku = E.BANDED_KL
        maxrel_dense = 0.0
        maxrel_lapack = 0.0
        for _ in 1:50
            n = rand(rng, 6:40)
            A = zeros(Float64, n, n)
            lu = E.BandedLU{Float64}(n)
            for row in 1:n, col in max(1, row - kl):min(n, row + kl)
                v = 2 * rand(rng) - 1
                row == col && (v += n)    # diagonal dominance -> well conditioned
                A[row, col] = v
                lu[row, col] = v
            end
            E.factor!(lu)

            b = [2 * rand(rng) - 1 for _ in 1:n]
            x_banded = E.solve!(lu, copy(b))
            x_dense = A \ b

            AB, ipiv = LinearAlgebra.LAPACK.gbtrf!(kl, ku, n, lapack_band(A, kl, ku))
            x_lapack = LinearAlgebra.LAPACK.gbtrs!('N', kl, ku, n, AB, ipiv, copy(b))

            for i in 1:n
                maxrel_dense = max(maxrel_dense, rel_err(x_banded[i], x_dense[i]))
                maxrel_lapack = max(maxrel_lapack, rel_err(x_banded[i], x_lapack[i]))
            end
            # Measured worst case over these 50 trials: 8.9e-15 against the
            # dense solve, 6.2e-15 against LAPACK. The bound is the C++'s.
            @test all(rel_err(x_banded[i], x_dense[i]) <= 1e-11 for i in 1:n)
            @test all(rel_err(x_banded[i], x_lapack[i]) <= 1e-11 for i in 1:n)
        end
        @test maxrel_dense <= 1e-11
        @test maxrel_lapack <= 1e-11
    end

    @testset "not-a-knot reproduces a global cubic exactly" begin
        f(x) = 3x^3 - 2x^2 + x - 5
        fp(x) = 9x^2 - 4x + 1
        fpp(x) = 18x - 4

        rng = StableRNG(1001)
        for n in (4, 7, 20)
            x0 = 10 * rand(rng) - 5
            h = 0.05 + 1.95 * rand(rng)
            data = [f(x0 + i * h) for i in 0:(n - 1)]
            v = BsplineView1(E.fit_bspline_1d(data), x0, h)

            xend = x0 + (n - 1) * h
            for _ in 1:100
                x = x0 + (xend - x0) * rand(rng)
                e = bspline_eval1(v, x)
                # Measured worst case over all three n: 3.7e-15 on f, 1.0e-14
                # on f', 1.8e-13 on f'' (the second derivative loses the most
                # digits, to the 1/h^2 factor). The bound is the C++'s.
                @test rel_err(e.f, f(x)) <= 1e-10
                @test rel_err(e.fx, fp(x)) <= 1e-10
                @test rel_err(e.fxx, fpp(x)) <= 1e-10
            end
        end
    end

    @testset "S(x_i) == f_i at every node" begin
        rng = StableRNG(2002)
        for n in (4, 5, 10, 30)
            data = [200 * rand(rng) - 100 for _ in 1:n]
            c = E.fit_bspline_1d(data)
            @test length(c) == n + 2
            x0, h = 0.0, 1.0
            v = BsplineView1(c, x0, h)
            for i in 1:n
                # Measured worst case: 5.4e-15 relative. The bound is the C++'s.
                @test rel_err(bspline_eval1(v, x0 + (i - 1) * h).f, data[i]) <= 1e-13
            end
        end
    end

    @testset "grid convergence on sin(x)" begin
        x0, xend = 0.0, 3.0
        ns = (20, 40, 80, 160)
        err_f = Float64[]
        err_fp = Float64[]
        err_fpp = Float64[]
        for n in ns
            h = (xend - x0) / (n - 1)
            v = BsplineView1(E.fit_bspline_1d([sin(x0 + i * h) for i in 0:(n - 1)]), x0, h)
            mf = mfp = mfpp = 0.0
            nsamp = 2000
            for k in 0:(nsamp - 1)
                x = x0 + (xend - x0) * (k + 0.5) / nsamp
                e = bspline_eval1(v, x)
                mf = max(mf, abs(e.f - sin(x)))
                mfp = max(mfp, abs(e.fx - cos(x)))
                mfpp = max(mfpp, abs(e.fxx + sin(x)))
            end
            push!(err_f, mf)
            push!(err_fp, mfp)
            push!(err_fpp, mfpp)
        end

        # A cubic spline is 4th-order accurate in the value and loses one order
        # per derivative, so halving h should divide the errors by ~16, ~8 and
        # ~4. Measured ratios (20->40, 40->80, 80->160): f 24.3, 20.5, 18.4;
        # f' 11.9, 10.5, 9.9; f'' 5.7, 5.0, 4.6 -- decaying towards 16/8/4 from
        # above, because the boundary cells (where the not-a-knot conditions
        # act and the error is largest at coarse n) converge faster than the
        # interior. The lower bounds are the C++'s, stated there as orders; the
        # upper one only pins that the ratios really are settling on 16.
        for i in 2:length(ns)
            @test err_f[i - 1] / err_f[i] >= 2^3.7
            @test err_fp[i - 1] / err_fp[i] >= 2^2.7
            @test err_fpp[i - 1] / err_fpp[i] >= 2^1.7
        end
        @test err_f[end - 1] / err_f[end] <= 32
    end

    @testset "f'' is continuous across interior knots" begin
        rng = StableRNG(3003)
        n = 30
        data = [100 * rand(rng) - 50 for _ in 1:n]
        x0, h = 0.0, 0.37
        v = BsplineView1(E.fit_bspline_1d(data), x0, h)

        jumps = Float64[]
        scale = 0.0
        for i in 1:(n - 2)
            xk = x0 + i * h
            el = bspline_eval1(v, xk - 1e-9)
            er = bspline_eval1(v, xk + 1e-9)
            push!(jumps, abs(el.fxx - er.fxx))
            scale = max(scale, abs(el.fxx), abs(er.fxx))
        end
        # The jump is not exactly zero only because the two samples sit 2e-9
        # apart, so f''' * 2e-9 leaks in; measured worst case 7.6e-6 against a
        # tolerance of 2.5e-3 (1e-6 * max|f''| = 1e-6 * 2.5e3).
        tol = 1e-6 * max(scale, 1e-12)
        @test maximum(jumps) <= tol
    end

    @testset "fit_bspline_3d on a separable product" begin
        A(x) = x^3 - x
        Ap(x) = 3x^2 - 1
        App(x) = 6x
        B(u) = 2u^2 + u
        Bp(u) = 4u + 1
        Bpp(_) = 4.0
        C(y) = y^3 + 1
        Cp(y) = 3y^2

        nx, nu, ny = 6, 5, 4
        x0, hx = -1.0, 0.4
        u0, hu = 0.0, 0.3
        y0, hy = -0.5, 0.3
        xend, uend, yend = x0 + (nx - 1) * hx, u0 + (nu - 1) * hu, y0 + (ny - 1) * hy

        data = [A(x0 + (i - 1) * hx) * B(u0 + (j - 1) * hu) * C(y0 + (k - 1) * hy)
                for i in 1:nx, j in 1:nu, k in 1:ny]

        v = E.fit_bspline_3d(data, x0, hx, u0, hu, y0, hy)
        @test v isa BsplineView3{Float64,Array{Float64,3}}
        @test size(v.c) == (nx + 2, nu + 2, ny + 2)
        @test (v.x0, v.hx, v.u0, v.hu, v.y0, v.hy) == (x0, hx, u0, hu, y0, hy)

        # Every factor of every reference product vanishes somewhere inside the
        # sampled box (A at x = -1, 0, 1; Ap at ±1/√3; App at 0; Cp at 0), so
        # the bound is relative to max(|ref|, 1e-2): near a root the absolute
        # error stays at roundoff while the relative one grows like 1/|ref|.
        # Measured worst case with that floor: 2.4e-12. Bound is the C++'s.
        rng = StableRNG(4004)
        fl = 1e-2
        for _ in 1:200
            x = x0 + (xend - x0) * rand(rng)
            u = u0 + (uend - u0) * rand(rng)
            y = y0 + (yend - y0) * rand(rng)
            e = bspline_eval3(v, x, u, y)
            @test rel_err(e.f, A(x) * B(u) * C(y), fl) <= 1e-9
            @test rel_err(e.fx, Ap(x) * B(u) * C(y), fl) <= 1e-9
            @test rel_err(e.fu, A(x) * Bp(u) * C(y), fl) <= 1e-9
            @test rel_err(e.fy, A(x) * B(u) * Cp(y), fl) <= 1e-9
            @test rel_err(e.fxx, App(x) * B(u) * C(y), fl) <= 1e-9
            @test rel_err(e.fxu, Ap(x) * Bp(u) * C(y), fl) <= 1e-9
            @test rel_err(e.fuu, A(x) * Bpp(u) * C(y), fl) <= 1e-9
        end
    end

    @testset "the three axis passes commute" begin
        # A test-local re-implementation of one axis pass, written against the
        # public fit_bspline_1d only: it shares nothing with fit_bspline_3d's
        # internals, so agreement is a cross-check rather than a tautology.
        function fit_along_axis(data::AbstractArray{Float64,3}, axis::Int)
            dims = size(data)
            out = Array{Float64,3}(undef, ntuple(a -> a == axis ? dims[a] + 2 : dims[a], 3))
            a1, a2 = Tuple(a for a in 1:3 if a != axis)
            line(i1, i2) = ntuple(a -> a == axis ? Colon() : (a == a1 ? i1 : i2), 3)
            for i2 in 1:dims[a2], i1 in 1:dims[a1]
                out[line(i1, i2)...] = E.fit_bspline_1d(collect(view(data, line(i1, i2)...)))
            end
            return out
        end

        rng = StableRNG(5005)
        nx, nu, ny = 8, 7, 6
        data = [20 * rand(rng) - 10 for _ in 1:nx, _ in 1:nu, _ in 1:ny]

        xuy = E.fit_bspline_3d(data, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0).c
        yux = fit_along_axis(fit_along_axis(fit_along_axis(data, 3), 2), 1)

        @test size(yux) == size(xuy)
        # Not bitwise: the passes commute exactly in exact arithmetic, but
        # solving along u before x rounds each intermediate differently.
        # Measured worst case 8.3e-14 relative (floor 1e-12); bound is the
        # C++'s 1e-12.
        @test maximum(rel_err(yux[i], xuy[i]) for i in eachindex(xuy)) <= 1e-12
    end

    @testset "argument checking of the fits" begin
        @test_throws ArgumentError E.fit_bspline_1d(zeros(3))
        @test_throws ArgumentError E.fit_bspline_3d(zeros(3, 5, 5), 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        @test_throws ArgumentError E.fit_bspline_3d(zeros(5, 3, 5), 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        @test_throws ArgumentError E.fit_bspline_3d(zeros(5, 5, 3), 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        # n = 4 is the smallest system the not-a-knot stencil fits in.
        @test length(E.fit_bspline_1d(Float64[1, 2, 3, 4])) == 6
    end

    @testset "type generic at Float32" begin
        # The fit is only ever exercised at Float64; this just pins that the
        # plumbing does not silently widen.
        data32 = Float32[sin(0.3f0 * i) for i in 1:8]
        c32 = E.fit_bspline_1d(data32)
        @test c32 isa Vector{Float32}
        v32 = BsplineView1(c32, 0.0f0, 1.0f0)
        @test bspline_eval1(v32, 3.0f0).f ≈ data32[4] rtol = 1e-5
        d3 = Float32[i + j + k for i in 1:5, j in 1:5, k in 1:5]
        @test E.fit_bspline_3d(d3, 0.0f0, 1.0f0, 0.0f0, 1.0f0, 0.0f0, 1.0f0) isa
            BsplineView3{Float32,Array{Float32,3}}
    end
end

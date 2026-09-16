# Allocation probes at top level; see the note in test_adapter_tail.jl.
_dev_eval(v, ρ, s, y, u) = EntropyEOS.evaluate(v, ρ, s, y, u).U
_dev_c2p(v, cin, opts) = EntropyEOS.con2prim(v, cin, opts).ρ
_dev_safe(v, cin, opts, pol) = EntropyEOS.con2prim_safe(v, cin, opts, pol).base.ρ
function _allocs(f, args...)
    f(args...)
    return @allocated f(args...)
end

@testset "device readiness" begin
    E = EntropyEOS
    tbl = make_synthetic_table()
    eos = E.build_eos(tbl)
    v = EOSTableView(eos)
    opts = Con2PrimOptions()
    pol = default_policy(v, 1e3 * v.κ)

    ρ = exp10(0.5 * (v.x_lo + v.x_hi))
    yₑ = 0.35
    sr = srange(v, ρ, yₑ)
    s = 0.5 * (sr.s_min + sr.s_max)
    pt = evaluate(v, ρ, s, yₑ, NaN)
    c = prim2con(v, ρ, s, yₑ, 0.8, 0.1ρ, 0.3, pt.u_solved)
    cin = Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²)

    @testset "kernel-side types are isbits once adapted" begin
        # A view over a device array type must be isbits, or it cannot be a
        # kernel argument at all. This is the single most likely regression:
        # someone adds a Vector or a String field to a core struct.
        @test isbitstype(EOSPoint{Float64})
        @test isbitstype(EOSPoint{Float32})
        @test isbitstype(SRange{Float64})
        @test isbitstype(Prim2ConOut{Float64})
        @test isbitstype(Con2PrimIn{Float64})
        @test isbitstype(Con2PrimOptions{Float64})
        @test isbitstype(Con2PrimOut{Float64})
        @test isbitstype(PolicyOptions{Float64})
        @test isbitstype(PrimState{Float64})
        @test isbitstype(Con2PrimSafeOut{Float64})
        @test isbitstype(BsplineEval3{Float64})
        # The views hold arrays, so they are isbits exactly when the array type
        # is -- which is what Adapt arranges on the device side.
        @test isbitstype(BsplineView3{Float64,SArray{Tuple{6,6,6},Float64,3,216}})
        @test !isbitstype(BsplineView3{Float64,Array{Float64,3}})   # host side, as expected
    end

    @testset "adapt round trip is bitwise" begin
        # The host-side analogue of a device mirror: rebind the view onto copies
        # of the coefficient arrays and assert nothing changes in use.
        #
        # Bitwise is the right bar *here*. On a real device the vendor maths
        # differs by a few units in the last place, so a GPU test has to compare
        # against ground truth instead -- but the rebinding itself must be exact.
        copyspline(b) = BsplineView3(copy(b.c), b.x0, b.hx, b.u0, b.hu, b.y0, b.hy)
        v2 = Adapt.adapt(Array, E.EOSTableView(copyspline(v.σ), copyspline(v.L), v.κ, v.shift_hat,
                                               v.conv_t, v.inv_c², v.x_lo, v.x_hi, v.u_lo, v.u_hi,
                                               v.y_lo, v.y_hi, v.x_ext_lo, v.x_ext_hi, v.u_ext_lo,
                                               v.u_ext_hi, v.ext_slope_floor_σ, v.ext_slope_floor_L,
                                               v.cs²_ext_cap, v.max_iter))
        @test v2.σ.c !== v.σ.c        # genuinely a different array
        @test v2.σ.c == v.σ.c
        n = 0
        for x in range(v.x_lo, v.x_hi; length=9), y in range(v.y_lo, v.y_hi; length=5)
            ρi = exp10(x)
            sri = srange(v, ρi, y)
            for si in range(sri.s_min, sri.s_max; length=7)
                a = evaluate(v, ρi, si, y, NaN)
                b = evaluate(v2, ρi, si, y, NaN)
                for f in (:U, :U_ρ, :U_s, :U_ρρ, :U_ρs, :p, :h, :cs², :T_MeV, :u_solved)
                    @test getfield(a, f) === getfield(b, f)
                    n += 1
                end
                @test a.flags === b.flags
                @test a.iters === b.iters
            end
        end
        @test n == 9 * 5 * 7 * 10
    end

    @testset "allocation-free on every entry point" begin
        # A heap allocation inside a kernel is fatal on a GPU, so these are the
        # gates that keep the port device-ready.
        @test _allocs(_dev_eval, v, ρ, s, yₑ, NaN) == 0
        @test _allocs(_dev_c2p, v, cin, opts) == 0
        @test _allocs(_dev_safe, v, cin, opts, pol) == 0
    end

    @testset "type-stable at Float64 and Float32" begin
        @test (@inferred evaluate(v, ρ, s, yₑ, NaN)) isa EOSPoint{Float64}
        @test (@inferred con2prim(v, cin, opts)) isa Con2PrimOut{Float64}
        @test (@inferred con2prim_safe(v, cin, opts, pol)) isa Con2PrimSafeOut{Float64}

        v32 = E.narrow(v, Float32)
        @test v32 isa EOSTableView{Float32}
        @test (@inferred evaluate(v32, Float32(ρ), Float32(s), Float32(yₑ), NaN32)) isa EOSPoint{Float32}
        opts32 = Con2PrimOptions{Float32}()
        cin32 = Con2PrimIn(Float32(cin.D), Float32(cin.τ), Float32(cin.D_Y), Float32(cin.S_par),
                           Float32(cin.S_perp), Float32(cin.B²))
        @test (@inferred con2prim(v32, cin32, opts32)) isa Con2PrimOut{Float32}
        # Float32 tolerances must at least be representable -- the physics there
        # is explicitly not validated, but the plumbing has to run.
        @test opts32.tol > eps(Float32)
        @test isfinite(con2prim(v32, cin32, opts32).ρ)
    end

    @testset "no hidden mutable state" begin
        # Warm-start state is threaded explicitly, so repeated calls with the
        # same arguments are bit-identical and calls are safe to run in
        # parallel across grid points.
        a = evaluate(v, ρ, s, yₑ, NaN)
        b = evaluate(v, ρ, s, yₑ, NaN)
        @test a.U === b.U && a.u_solved === b.u_solved && a.iters === b.iters
        p = con2prim(v, cin, opts)
        q = con2prim(v, cin, opts)
        @test p.ρ === q.ρ && p.s === q.s && p.w === q.w && p.result === q.result
    end
end

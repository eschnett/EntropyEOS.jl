# Opt-in GPU test. Neither Metal nor CUDA nor KernelAbstractions is a
# dependency of this package or of its test environment -- Adapt is the entire
# GPU contract -- so this file loads a backend only when asked and skips
# cleanly otherwise.
#
# Set ENTROPYEOS_TEST_GPU=metal (or cuda) with the corresponding package
# installed in the active environment.
#
# Note what is and is not asserted. On a device the vendor's own maths differs
# from the host's by units in the last place, so results are compared against
# ground truth and a tolerance, never bitwise against the CPU. Iteration counts
# and the solver's result code are deliberately NOT compared either: a state
# near a decision boundary can legitimately take the Newton path on one device
# and the fallback on the other. Both must converge; which route they take is
# not part of the contract.

const GPU_BACKEND = lowercase(get(ENV, "ENTROPYEOS_TEST_GPU", "none"))

if GPU_BACKEND == "none"
    @info "GPU tests skipped (set ENTROPYEOS_TEST_GPU=metal|cuda to enable)"
else
    gpu_ok = false
    try
        # `global` matters: this runs at top level, where an assignment inside
        # `try` would otherwise introduce a new local and leave gpu_ok false.
        if GPU_BACKEND == "metal"
            @eval using Metal, KernelAbstractions
            global gpu_ok = Metal.functional()
            @eval const BACKEND = MetalBackend()
            @eval const DEVARRAY = MtlArray
            @eval const GPU_ELTYPE = Float32       # Metal has no Float64
        elseif GPU_BACKEND == "cuda"
            @eval using CUDA, KernelAbstractions
            global gpu_ok = CUDA.functional()
            @eval const BACKEND = CUDABackend()
            @eval const DEVARRAY = CuArray
            @eval const GPU_ELTYPE = Float64
        end
    catch err
        @info "GPU backend '$GPU_BACKEND' unavailable" err
    end

    if !gpu_ok
        @info "GPU backend '$GPU_BACKEND' present but not functional; skipping"
    else
        @eval @kernel function _gpu_eval!(out, @Const(ρs), @Const(ss), v, yₑ)
            i = @index(Global)
            pt = EntropyEOS.evaluate(v, ρs[i], ss[i], yₑ, typeof(yₑ)(NaN))
            @inbounds out[i] = pt.p
        end

        @eval @kernel function _gpu_c2p!(outρ, @Const(D), @Const(τ), @Const(DY), @Const(Sp),
                                         @Const(Sq), @Const(B2), v, opts)
            i = @index(Global)
            o = EntropyEOS.con2prim(v, Con2PrimIn(D[i], τ[i], DY[i], Sp[i], Sq[i], B2[i]), opts)
            @inbounds outρ[i] = o.ρ
        end

        @testset "GPU ($GPU_BACKEND)" begin
            E = EntropyEOS
            T = GPU_ELTYPE
            tbl = make_synthetic_table()
            v = E.narrow(EOSTableView(E.build_eos(tbl)), T)
            yₑ = T(0.35)
            n = 1024

            xs = range(T(v.x_lo) + one(T), T(v.x_hi) - one(T); length=n)
            ρs = T[exp10(x) for x in xs]
            mid(ρ) = (sr = srange(v, ρ, yₑ); T(0.5) * (sr.s_min + sr.s_max))
            ss = T[mid(ρ) for ρ in ρs]

            # Adapt is the only thing the package supplies for this hop, and the
            # adapted view must be isbits or it cannot be a kernel argument.
            dv = Adapt.adapt(DEVARRAY, v)
            @test eltype(dv) === T

            @testset "evaluate" begin
                outc = zeros(T, n)
                _gpu_eval!(CPU(), 64)(outc, ρs, ss, v, yₑ; ndrange=n)
                KernelAbstractions.synchronize(CPU())

                dout = DEVARRAY(zeros(T, n))
                _gpu_eval!(BACKEND, 64)(dout, DEVARRAY(ρs), DEVARRAY(ss), dv, yₑ; ndrange=n)
                KernelAbstractions.synchronize(BACKEND)
                outg = Array(dout)

                @test all(isfinite, outg)
                @test all(>(0), outg)
                rel = maximum(abs.(outg .- outc) ./ max.(abs.(outc), eps(T)))
                @info "GPU evaluate: max relative difference from host" rel
                @test rel < sqrt(eps(T)) * 100
            end

            @testset "con2prim" begin
                D = T[]; τ = T[]; DY = T[]; Sp = T[]; Sq = T[]; B2 = T[]
                for (k, ρ) in enumerate(ρs)
                    s = mid(ρ)
                    c = prim2con(v, ρ, s, yₑ, T(0.1 + 1.5k / n), zero(T), zero(T), T(NaN))
                    push!(D, c.D); push!(τ, c.τ); push!(DY, c.D_Y)
                    push!(Sp, c.S_par); push!(Sq, c.S_perp); push!(B2, c.B²)
                end
                opts = Con2PrimOptions{T}()

                outc = zeros(T, n)
                _gpu_c2p!(CPU(), 64)(outc, D, τ, DY, Sp, Sq, B2, v, opts; ndrange=n)
                KernelAbstractions.synchronize(CPU())

                dout = DEVARRAY(zeros(T, n))
                _gpu_c2p!(BACKEND, 64)(dout, DEVARRAY(D), DEVARRAY(τ), DEVARRAY(DY), DEVARRAY(Sp),
                                       DEVARRAY(Sq), DEVARRAY(B2), dv, opts; ndrange=n)
                KernelAbstractions.synchronize(BACKEND)
                outg = Array(dout)

                # That this compiles at all is itself a result: it proves no
                # allocation, no exception path and no host-only call survived
                # into the kernel -- including the bracket scan's stack scratch.
                @test all(isfinite, outg)
                rel = maximum(abs.(outg .- outc) ./ abs.(outc))
                @info "GPU con2prim: max relative difference from host" rel
                @test rel < sqrt(eps(T)) * 100
            end
        end
    end
end

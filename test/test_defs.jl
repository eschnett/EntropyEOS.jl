@testset "defs" begin
    E = EntropyEOS

    @testset "flag bits" begin
        # Evaluation bits occupy 0-5, policy bits 8-15, with 6-7 reserved so the
        # policy group stays contiguous and maskable.
        eval_flags = (FLAG_CLAMP_YE, FLAG_EXT_S_LOW, FLAG_EXT_S_HIGH,
                      FLAG_EXT_ρ_LOW, FLAG_OOB_ρ_HIGH, FLAG_MAXITER)
        pol_flags = (FLAG_POL_ATMOSPHERE, FLAG_POL_CEILING, FLAG_POL_S_FLOORED,
                     FLAG_POL_S_CEILED, FLAG_POL_W_CAPPED, FLAG_POL_ρ_CLAMPED,
                     FLAG_POL_YE_CLAMPED, FLAG_POL_NONFINITE)
        @test all(f -> f isa UInt32, eval_flags)
        @test all(f -> f isa UInt32, pol_flags)
        @test reduce(|, eval_flags) == 0x003f
        @test reduce(|, pol_flags) == FLAG_POL_ANY
        @test FLAG_POL_ANY == 0xff00
        @test reduce(|, eval_flags) & FLAG_POL_ANY == 0    # the groups do not overlap
        @test length(unique(eval_flags)) == 6
        @test length(unique(pol_flags)) == 8
    end

    @testset "outcome enums" begin
        @test Status.ok isa Status.T
        @test length(instances(Status.T)) == 3
        @test length(instances(C2PResult.T)) == 4
        @test isbitstype(Status.T) && isbitstype(C2PResult.T)
    end

    @testset "domain-safe math" begin
        # C semantics: -Inf at zero, NaN below zero, and never a throw. Julia's
        # own log10 raises DomainError below zero, which would be an illegal
        # exception path inside a GPU kernel.
        @test E.safe_log10(100.0) == 2.0
        @test E.safe_log10(0.0) == -Inf
        @test isnan(E.safe_log10(-1.0))
        @test isnan(E.safe_log10(NaN))
        @test E.safe_log10(1.0f2) === 2.0f0

        @test E.safe_sqrt(4.0) == 2.0
        @test isnan(E.safe_sqrt(-1.0))
        @test isnan(E.safe_sqrt(NaN))

        # Total on every input; callers always clamp the result into range, so
        # a non-finite argument only has to be harmless.
        @test E.trunc_floor(2.7) == 2
        @test E.trunc_floor(-2.7) == -3
        @test E.trunc_floor(NaN) isa Int
        @test E.trunc_floor(Inf) isa Int
    end

    @testset "tolerances reproduce the C++ Float64 values" begin
        @test E.con2prim_tol(Float64) == 1.0e-12
        @test E.tau_floor_rel(Float64) == 1.0e-16
        @test E.tsolve_residual_tol(Float64) == 1.0e-12
        @test E.tsolve_step_tol(Float64) == 1.0e-13
        @test E.seed_z_tol(Float64) == 1.0e-14
        @test E.con2prim_tol(Float32) == 512 * eps(Float32)
        @test E.perp_degenerate(Float64) == 1.0e-300
        @test E.ln10(Float64) == log(10.0)
        @test E.BRACKET_SCAN_MAX == 33
        @test E.SEED_SCALAR_ITERS == 40
    end

    @testset "tolerances stay usable at Float32" begin
        # 1e-12 is below eps(Float32), so a literal translation would make the
        # convergence test unsatisfiable and every solve burn its full budget.
        @test E.con2prim_tol(Float32) > eps(Float32)
        @test E.tsolve_residual_tol(Float32) > eps(Float32)
        @test E.tsolve_step_tol(Float32) > eps(Float32)
        @test E.seed_z_tol(Float32) > eps(Float32)
        # 1e-300 underflows to exactly zero in Float32, which would switch the
        # guard off and let Inf*0 produce a NaN.
        @test E.perp_degenerate(Float32) > 0
        @test E.tiny_denom(Float32) > 0
        for T in (Float32, Float64)
            @test E.con2prim_tol(T) isa T
            @test E.ln10(T) isa T
            @test E.scan_delta_min(T) isa T
            @test E.scan_delta_max(T) isa T
            @test E.tiny_w(T) isa T
            @test E.xlow_log_excursion_max(T) isa T
        end
    end

    @testset "source discipline" begin
        # @fastmath folds the NaN and finiteness probes to the wrong answer, so
        # it must not appear in any source file. This is the same prohibition
        # the C++ states for -ffast-math. Comment lines are exempt: the rule is
        # documented in prose in defs.jl.
        offenders = Tuple{String,Int}[]
        for (root, _, files) in walkdir(joinpath(@__DIR__, "..", "src")), f in files
            endswith(f, ".jl") || continue
            for (n, line) in enumerate(eachline(joinpath(root, f)))
                startswith(strip(line), "#") && continue
                occursin("@fastmath", line) && push!(offenders, (f, n))
            end
        end
        @test isempty(offenders)
    end
end

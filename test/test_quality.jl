# Project hygiene and static analysis.
#
# Aqua is cheap and catches a class of defect that unit tests cannot: a
# dependency declared but never used, a missing compat bound, a stale export, a
# method ambiguity. It would have caught `PrecompileTools` sitting unused in
# Project.toml, which is how that was eventually found.
#
# JET asserts the property the GPU path depends on and that `@allocated` can
# only measure indirectly: no runtime dispatch and no type instability anywhere
# in the kernels. Analyse concrete argument types via a wrapper function -- a
# closure over non-const globals reports its own captures as dynamic dispatch
# and buries the real signal.

_jet_bs(v, x, u, y) = EntropyEOS.bspline_eval3(v, x, u, y)
_jet_eval(v, ρ, s, y, u) = EntropyEOS.evaluate(v, ρ, s, y, u)
_jet_p2c(v, ρ, s, y, w, b, c, u) = EntropyEOS.prim2con(v, ρ, s, y, w, b, c, u)
_jet_c2p(v, cin, o) = EntropyEOS.con2prim(v, cin, o)
_jet_safe(v, cin, o, pol) = EntropyEOS.con2prim_safe(v, cin, o, pol)

@testset "quality" begin
    @testset "Aqua" begin
        # `persistent_tasks` bounds how long the probe process may take to
        # *exit* after loading the package; loading itself is unbounded. The
        # default 30 s is ample on a developer machine but has timed out on a
        # loaded two-core CI runner, where the probe must precompile this
        # package and every dependency from scratch — the more so since the
        # `@compile_workload` made precompilation ~70% more expensive, and
        # since a job running under non-default `--check-bounds` cannot reuse
        # the existing cache. The workload runs at precompile time in a
        # separate process, so it cannot leave a task behind at load time;
        # raising the limit is Aqua's own documented remedy rather than
        # switching the check off.
        Aqua.test_all(EntropyEOS; persistent_tasks=(; tmax=180))
    end

    @testset "JET: kernels are free of runtime dispatch" begin
        # JET's analysis tracks the compiler, so results can shift with a Julia
        # release. Pin it to the versions this package is developed against
        # rather than have a new Julia turn a green suite red.
        if VERSION >= v"1.11"
            tbl = make_synthetic_table(SyntheticOptions(; nρ=8, nT=8, nYₑ=8))
            v = EOSTableView(EntropyEOS.build_eos(tbl))
            V, F = typeof(v), Float64
            BV = typeof(v.σ)
            @test isempty(JET.get_reports(JET.report_opt(_jet_bs, (BV, F, F, F))))
            @test isempty(JET.get_reports(JET.report_opt(_jet_eval, (V, F, F, F, F))))
            @test isempty(JET.get_reports(JET.report_opt(_jet_p2c, (V, F, F, F, F, F, F, F))))
            @test isempty(JET.get_reports(JET.report_opt(_jet_c2p, (V, Con2PrimIn{F}, Con2PrimOptions{F}))))
            @test isempty(JET.get_reports(JET.report_opt(_jet_safe,
                (V, Con2PrimIn{F}, Con2PrimOptions{F}, PolicyOptions{F}))))
            # The analytic EOSs run the same solver, and must be as clean.
            for A in (typeof(IdealGasEOS(; Γ=2.0, K_ref=100.0, s_ref=5.0, s_window=(1.0, 20.0),
                                         ρ_bounds=(1e-10, 1e-2), yₑ_bounds=(0.0, 1.0))),
                      typeof(HybridEOS(; ρ_breaks=(1e-4,), K₀=100.0, Γs=(2.0, 2.5), Γ_th=5 / 3, K_th_ref=1.0,
                                       s_ref=0.0, s_window=(1.0, 5.0), ρ_bounds=(1e-10, 1e-3),
                                       yₑ_bounds=(0.0, 1.0))))
                @test isempty(JET.get_reports(JET.report_opt(_jet_eval, (A, F, F, F, F))))
                @test isempty(JET.get_reports(JET.report_opt(_jet_c2p, (A, Con2PrimIn{F}, Con2PrimOptions{F}))))
                @test isempty(JET.get_reports(JET.report_opt(_jet_safe,
                    (A, Con2PrimIn{F}, Con2PrimOptions{F}, PolicyOptions{F}))))
            end
        else
            @info "JET analysis skipped on Julia $VERSION"
        end
    end
end

# Precompilation workload.
#
# `con2prim` inlines a deep tree -- the EOS evaluation, the designed tails and
# the spline contraction all end up inside the residual, which is then inside
# the Newton loop, the inner solve, the bracket scan and the cold seed. Compiling
# that on first call costs seconds, which a hydro code would otherwise pay at
# startup, and a test suite once per session.
#
# The workload runs the whole public `Float64` path over a deliberately tiny
# table: four points per axis is the minimum the spline fit accepts, and the
# refinement and extension widths are turned down because this exists to reach
# the code paths, not to produce a usable adapter. The types and therefore the
# compiled specializations are identical to a real build's.
#
# `Float32` is not precompiled. It would roughly double both the precompilation
# cost and the cache size for a path that is mostly taken on a GPU, where the
# kernel is compiled separately anyway, so a host caller that wants it pays the
# first-call cost.
#
# Of the analytic EOSs only the ideal gas is precompiled: `HybridEOS` is
# specialized on its number of pieces, so no one instance would cover callers.

using PrecompileTools: @compile_workload

@compile_workload begin
    opts = SyntheticOptions(; nρ=5, nT=5, nYₑ=5, with_aux_fields=true)
    tbl = make_synthetic_table(opts)
    check_table(tbl)

    eos = build_eos(tbl, BuildOptions(; refine=1, ext_cells=2))
    v = EOSTableView(eos)

    ρ = exp10(0.5 * (v.x_lo + v.x_hi))
    yₑ = 0.5 * (v.y_lo + v.y_hi)
    sr = srange(v, ρ, yₑ)
    srange_extended(v, ρ, yₑ)
    s = 0.5 * (sr.s_min + sr.s_max)
    sigma_extended(v, ρ, 0.5 * (v.u_lo + v.u_hi), yₑ)

    pt = evaluate(v, ρ, s, yₑ, NaN)
    evaluate(v, ρ, s, yₑ, pt.u_solved)          # the warm path, which dominates

    c = prim2con(v, ρ, s, yₑ, 0.8, 0.1ρ, 0.3, pt.u_solved)
    prim2con(v, ρ, s, yₑ, 0.8, SVector(1.0, 0.0, 0.0), SVector(0.0, 0.0, 1.0), pt.u_solved)

    cin = Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²)
    copts = Con2PrimOptions()
    con2prim(v, cin, copts, s, 0.8, pt.u_solved)   # warm
    con2prim(v, cin, copts)                        # cold: seed and, on this table, the fallback

    pol = default_policy(v, exp10(v.x_lo))
    ps = PrimState(ρ, s, yₑ, 0.8)
    check_prim_state(v, ps, pol)
    project_prim_state(v, ps, pol)
    check_con_state(v, cin, pol)
    con2prim_safe(v, cin, copts, pol)
    # The excision branch compiles separately from the ordinary one.
    con2prim_safe(v, Con2PrimIn(cin.D, NaN, cin.D_Y, cin.S_par, cin.S_perp, cin.B²), copts, pol)

    ig = IdealGasEOS(; Γ=2.0, K_ref=100.0, s_ref=5.0, s_window=(1.0, 20.0), ρ_bounds=(1e-10, 1e-2),
                     yₑ_bounds=(0.0, 1.0))
    ρ = 1e-4
    pt = evaluate(ig, ρ, 5.0, 0.5, NaN)
    c = prim2con(ig, ρ, 5.0, 0.5, 0.8, 0.1ρ, 0.3, NaN)
    prim2con(ig, ρ, 5.0, 0.5, 0.8, SVector(1.0, 0.0, 0.0), SVector(0.0, 0.0, 1.0), NaN)
    cin = Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²)
    con2prim(ig, cin, copts, 5.0, 0.8, NaN)
    con2prim(ig, cin, copts)
    pol = default_policy(ig, 1e-9)
    project_prim_state(ig, PrimState(ρ, 5.0, 0.5, 0.8), pol)
    con2prim_safe(ig, cin, copts, pol)
    con2prim_safe(ig, Con2PrimIn(cin.D, NaN, cin.D_Y, cin.S_par, cin.S_perp, cin.B²), copts, pol)
end

# Shared test helpers.

"""
Whether an allocation measurement means anything in this session.

The kernels are allocation-free only as *optimized* code, and
`--code-coverage` instruments every line and blocks many optimizations. That
does not indicate a bug. `julia-actions/julia-runtest` turns coverage on by
default, which is why the CI workflow runs extra jobs with it off — those jobs
are where these assertions actually execute.

`--check-bounds=yes` is not excluded: the kernels stay allocation-free with
their bounds checks on, and a downstream suite runs that way.
"""
const ALLOCATION_TESTS_MEANINGFUL = Base.JLOptions().code_coverage == 0

"""
    @test_noallocs expr

Assert `expr == 0`, or skip when an allocation measurement would be meaningless
here (see [`ALLOCATION_TESTS_MEANINGFUL`](@ref)). Skipped assertions show up in
the summary's Broken column rather than silently passing.
"""
macro test_noallocs(ex)
    return quote
        if ALLOCATION_TESTS_MEANINGFUL
            @test $(esc(ex)) == 0
        else
            @test_skip $(esc(ex)) == 0
        end
    end
end

"""
    policy_violations(eos, proj, opts, cin, out) -> Vector{String}

Every level-1 invariant of CODE.md ("State policy"), for one input `cin` and
`con2prim_safe`'s output `out`. Returns the names of those violated, so that a
batch can report them together.
"""
function policy_violations(eos, proj, opts, cin, out)
    bad = String[]
    E = EntropyEOS
    f = out.policy_flags
    ρ_min, ρ_max, w_cap = E.proj_ρ_min(eos, proj), E.proj_ρ_max(eos, proj), E.proj_w_cap(proj)
    δ = E.face_slack(Float64)
    has(f, bits) = f & bits != 0
    cin_of(c) = Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²)
    Smag(c) = hypot(c.S_par, c.S_perp)
    b, c = out.base, out.cons
    all(isfinite, (b.ρ, b.s, b.ye, b.w, c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²)) || push!(bad, "non-finite")
    # The output is valid ...
    check_prim_state(eos, PrimState(b.ρ, b.s, b.ye, b.w), proj) == 0 || push!(bad, "invalid")
    inside_window(eos, b.ρ, b.s, b.ye) || push!(bad, "outside the window")
    # ... the input itself where nothing was flagged ...
    if f == 0
        (c.D === cin.D && c.τ === cin.τ && c.D_Y === cin.D_Y && c.S_par === cin.S_par &&
         c.S_perp === cin.S_perp && c.B² === cin.B²) || push!(bad, "flags 0 but not the input")
    end
    # ... and a fixed point, fed back with its own primitives as the guess.
    out2 = con2prim_safe(eos, cin_of(c), opts, proj, b.s, b.w, b.eos.u_solved)
    out2.policy_flags == 0 || push!(bad, "not a fixed point")
    has(f, FLAG_POL_NONFINITE | FLAG_POL_NOT_CONVERGED) && return bad

    # A set face flag means the state lies on that face, to the slack.
    sr = srange(eos, b.ρ, b.ye)
    ds = δ * (sr.s_max - sr.s_min)
    has(f, FLAG_POL_S_FLOORED) && !(abs(b.s - sr.s_min) <= ds) && push!(bad, "off s_min")
    has(f, FLAG_POL_S_CEILED) && !(abs(b.s - sr.s_max) <= ds) && push!(bad, "off s_max")
    has(f, FLAG_POL_W_CEILED) && !(abs(b.w - w_cap) <= δ * w_cap) && push!(bad, "off w_cap")
    has(f, FLAG_POL_ρ_FLOORED) && !(abs(b.ρ - ρ_min) <= δ * ρ_min) && push!(bad, "off ρ_min")
    has(f, FLAG_POL_ρ_CEILED) && !(abs(b.ρ - ρ_max) <= δ * ρ_max) && push!(bad, "off ρ_max")

    # Each face frees one conservative and moves it one way; the others
    # are kept, to the solver's tolerance (bars at 100 times it). S is kept in
    # the solver's sense, V = S/z to tol, so it is measured against z, which on
    # a hot state exceeds D by orders of magnitude. prim2con carries the
    # velocity's direction as cos_vB and recovers sin_vB from it, which costs
    # S_perp a relative eps/sin² where v is nearly along B or B² ≫ z; that is
    # its representation's limit, not the projection's, and widens its bar.
    cos_vB = b.v_par / max(hypot(b.v_par, b.v_perp), 1e-300)
    sin² = max(1 - cos_vB^2, 1e-300)
    along_B = sin² < 1e-8
    z = c.D * b.eos.h * b.W
    if !has(f, FLAG_POL_ρ_FLOORED | FLAG_POL_ρ_CEILED)
        abs(c.D - cin.D) <= 1e-12 * cin.D || push!(bad, "D not kept")
    elseif has(f, FLAG_POL_ρ_FLOORED)
        abs(c.D - max(cin.D, ρ_min)) <= 1e-12 * c.D || push!(bad, "D ≠ max(D, ρ_min)")
    else
        c.D <= cin.D * (1 + 1e-12) || push!(bad, "ρ_max raised D")
    end
    S0, S1 = Smag(cin), Smag(c)
    if !has(f, FLAG_POL_ρ_FLOORED | FLAG_POL_W_CEILED)
        scaleS = max(S0, z)
        (abs(c.S_par - cin.S_par) <= 1e-10 * scaleS &&
         abs(c.S_perp - cin.S_perp) <= (1e-10 + 100 * eps() / sin²) * scaleS) || push!(bad, "S not kept")
    else
        S1 <= S0 * (1 + 1e-10) || push!(bad, "S raised")
        # Lowered by scaling, not turned.
        along_B || abs(c.S_par * cin.S_perp - c.S_perp * cin.S_par) <= 1e-7 * S0 * S1 || push!(bad, "S turned")
    end
    scaleτ = max(cin.τ, opts.τ_floor_rel * cin.D)
    if !has(f, FLAG_POL_S_FLOORED | FLAG_POL_S_CEILED)
        abs(c.τ - cin.τ) <= 1e-10 * scaleτ || push!(bad, "τ not kept")
    else
        has(f, FLAG_POL_S_FLOORED) && !(c.τ >= cin.τ - 1e-10 * scaleτ) && push!(bad, "s_min lowered τ")
        has(f, FLAG_POL_S_CEILED) && !(c.τ <= cin.τ + 1e-10 * scaleτ) && push!(bad, "s_max raised τ")
    end
    if !has(f, FLAG_POL_YE_FLOORED | FLAG_POL_YE_CEILED)
        abs(c.D_Y / c.D - cin.D_Y / cin.D) <= 1e-14 || push!(bad, "Yₑ not kept")
    end
    return bad
end

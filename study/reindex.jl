# Re-indexing study: would a table indexed by (log ρ*, η(s), Yₑ) replace the
# temperature solve without losing accuracy?
#
# Today's adapter stores σ and L = log10(ε̂ + shift) on (x = log10 ρ*, u = log10 T,
# y = Yₑ) and solves σ(x, u, y) = s for u on every evaluation. The candidate
# stores a single field F(x, η, y) = L, with
#
#     φ(s) = asinh(s / s₀)                       (ln-like, but total for s ≤ 0)
#     a(x, y) = φ(σ(x, u_lo, y)),  b(x, y) = φ(σ(x, u_hi, y))
#     η = (φ(s) − a) / (b − a)                   ∈ [0, 1] across the physical window
#
# so an evaluation is two 2-D samples (a, b) and one 3-D sample, with no
# iteration. Derivatives here come from ForwardDiff through that evaluation,
# which is exact for the representation; a kernel would carry the chain rule
# by hand.
#
# Also measured: a seed table G(x, η, y) = u, i.e. a temperature predictor for
# today's solve, how far one or two Newton steps on σ take it, and (third part)
# how many iterations today's solve needs when started from it.
#
# The reference is today's cold `evaluate`, at points uniform in (x, u, y)
# inside the physical box. This measures what re-indexing adds to today's
# interpolation error, not the error against the raw table.
#
# A second part fits both representations to the table with every other point
# dropped on each axis and compares them against the full table, which puts the
# first part's numbers on today's own interpolation-error scale.
#
#     julia -t 8 --project=study study/reindex.jl [/path/to/tables] [--only=DD2] [--s0=1e-2,1]
#           [--neta=1,1.5,2] [--n=20000] [--skip-full] [--skip-held] [--skip-seed]
#
# With no table directory only the synthetic table runs.

using ForwardDiff

include("common.jl")

# ---------------------------------------------------------------------------
# Building the re-indexed fields
# ---------------------------------------------------------------------------

φ(s, s₀) = asinh(s / s₀)
φinv(f, s₀) = s₀ * sinh(f)

"""L = log10(ε̂ + shift) in table units, recovered exactly from an EOSPoint's U."""
L_of_U(v, U) = log10((v.κ * (1 + U) - 1 + v.shift_hat) / v.inv_c²)

struct Reindexed
    s₀::Float64
    nη::Int
    A::BsplineView3{Float64,Array{Float64,3}}   # a(x, y), u axis degenerate
    B::BsplineView3{Float64,Array{Float64,3}}   # b(x, y)
    F::BsplineView3{Float64,Array{Float64,3}}   # L(x, η, y)
    G::BsplineView3{Float64,Array{Float64,3}}   # u(x, η, y): the seed-table variant
    nonmono_cols::Int     # (x, y) columns where the solved u is not increasing in η
    ncols::Int
    neg_s_cols::Int       # columns whose window reaches s ≤ 0
end

"""A 2-D spline on the x and y knots, stored as 3-D with four identical u planes."""
function fit_2d(vals::Matrix{Float64}, σ)
    nx, ny = size(vals)
    data = Array{Float64,3}(undef, nx, 4, ny)
    for j in 1:4
        data[:, j, :] .= vals
    end
    return fit_bspline_3d(data, σ.x0, σ.hx, 0.0, 1.0, σ.y0, σ.hy)
end

function build_reindexed(v, s₀, nη)
    σ = v.σ
    nx, ny = E.npoints_x(σ), E.npoints_y(σ)
    xs = [σ.x0 + (i - 1) * σ.hx for i in 1:nx]
    ys = [σ.y0 + (k - 1) * σ.hy for k in 1:ny]
    a = [φ(E.bspline_eval3(σ, xs[i], v.u_lo, ys[k]).f, s₀) for i in 1:nx, k in 1:ny]
    b = [φ(E.bspline_eval3(σ, xs[i], v.u_hi, ys[k]).f, s₀) for i in 1:nx, k in 1:ny]
    neg = count(k -> E.bspline_eval3(σ, xs[k[1]], v.u_lo, ys[k[2]]).f <= 0, CartesianIndices((nx, ny)))

    ηs = range(0.0, 1.0; length=nη)
    Fd = Array{Float64,3}(undef, nx, nη, ny)
    Gd = Array{Float64,3}(undef, nx, nη, ny)
    nonmono = Threads.Atomic{Int}(0)
    Threads.@threads for k in 1:ny
        for i in 1:nx
            ρ★ = exp10(xs[i])
            mono = true
            for j in 1:nη
                s = φinv(a[i, k] + ηs[j] * (b[i, k] - a[i, k]), s₀)
                pt = E.evaluate(v, ρ★, s, ys[k], NaN)
                Fd[i, j, k] = L_of_U(v, pt.U)
                Gd[i, j, k] = pt.u_solved
                j > 1 && !(Gd[i, j, k] > Gd[i, j - 1, k]) && (mono = false)
            end
            mono || Threads.atomic_add!(nonmono, 1)
        end
    end
    hη = 1.0 / (nη - 1)
    F = fit_bspline_3d(Fd, σ.x0, σ.hx, 0.0, hη, σ.y0, σ.hy)
    G = fit_bspline_3d(Gd, σ.x0, σ.hx, 0.0, hη, σ.y0, σ.hy)
    return Reindexed(s₀, nη, fit_2d(a, σ), fit_2d(b, σ), F, G, nonmono[], nx * ny, neg)
end

# ---------------------------------------------------------------------------
# Evaluating the re-indexed table
# ---------------------------------------------------------------------------

sample(sp, x, u, y) = (R = promote_type(typeof(x), typeof(u), typeof(y)); E.bspline_eval3(sp, R(x), R(u), R(y)).f)

function η_at(r::Reindexed, x, s, y)
    a = sample(r.A, x, 0.0, y)
    b = sample(r.B, x, 0.0, y)
    return (φ(s, r.s₀) - a) / (b - a)
end

"""U(ρ*, s) at fixed Yₑ through the re-indexed field; generic for ForwardDiff."""
function U_new(v, r::Reindexed, ρ★, s, y)
    x = log10(ρ★)
    L = sample(r.F, x, η_at(r, x, s, y), y)
    ε̂ = exp10(L) * v.inv_c² - v.shift_hat
    return (1 + ε̂) / v.κ - 1
end

"""The same quantities `eval_at` returns, from the re-indexed field."""
function eval_new(v, r::Reindexed, ρ★, s, y)
    f(q) = U_new(v, r, q[1], q[2], y)
    res = ForwardDiff.hessian(f, [ρ★, s])   # exact derivatives of the representation
    U = f([ρ★, s])
    g = ForwardDiff.gradient(f, [ρ★, s])
    U_ρ, U_s, U_ρρ = g[1], g[2], res[1, 1]
    p = ρ★^2 * U_ρ
    h = 1 + U + p / ρ★
    cs² = (2ρ★ * U_ρ + ρ★^2 * U_ρρ) / h
    return (; U, U_s, p, h, cs², η=η_at(r, log10(ρ★), s, y))
end

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

function test_points(v, n, rng)
    pts = NamedTuple{(:ρ★, :s, :y, :u),NTuple{4,Float64}}[]
    for _ in 1:n
        x = v.x_lo + rand(rng) * (v.x_hi - v.x_lo)
        u = v.u_lo + rand(rng) * (v.u_hi - v.u_lo)
        y = v.y_lo + rand(rng) * (v.y_hi - v.y_lo)
        s = E.bspline_eval3(v.σ, x, u, y).f
        push!(pts, (ρ★=exp10(x), s=s, y=y, u=u))
    end
    return pts
end

"""The view without its temperature seed (2.6.0), so that a cold start brackets as it did when this study ran."""
unseeded(v) = typeof(v)((f === :seed ? E.empty_seed(v.σ.c) : getfield(v, f) for f in fieldnames(typeof(v)))...)

reference(v, pts) = (v0 = unseeded(v); [E.evaluate(v0, p.ρ★, p.s, p.y, NaN) for p in pts])

physical(pt) = pt.U_s > 0 && 0 < pt.cs² < 1

function compare(v, r::Reindexed, pts, refs)
    n = length(pts)
    eU, ep, eh, ecs, eT, eη = (zeros(n) for _ in 1:6)
    du0, du1, du2 = zeros(n), zeros(n), zeros(n)
    bad_T = bad_cs = 0
    Threads.@threads for i in 1:n
        p, ref = pts[i], refs[i]
        nw = eval_new(v, r, p.ρ★, p.s, p.y)
        eU[i] = rel_diff(nw.U, ref.U)
        ep[i] = rel_diff(nw.p, ref.p)
        eh[i] = rel_diff(nw.h, ref.h)
        ecs[i] = abs(nw.cs² - ref.cs²)
        eT[i] = rel_diff(nw.U_s, ref.U_s)
        eη[i] = max(-nw.η, nw.η - 1, 0.0)

        # Seed table: u from G, then Newton on today's σ.
        x = log10(p.ρ★)
        u = sample(r.G, x, nw.η, p.y)
        du0[i] = abs(u - ref.u_solved)
        for k in 1:2
            e = E.bspline_eval3(v.σ, x, u, p.y)
            e.fu > 0 && (u -= (e.f - p.s) / e.fu)
            (k == 1 ? du1 : du2)[i] = abs(u - ref.u_solved)
        end
    end
    for i in 1:n
        physical(refs[i]) || continue
        nw = eval_new(v, r, pts[i].ρ★, pts[i].s, pts[i].y)
        nw.U_s > 0 || (bad_T += 1)
        0 < nw.cs² < 1 || (bad_cs += 1)
    end
    return (; eU, ep, eh, ecs, eT, eη, du0, du1, du2, bad_T, bad_cs)
end

mb(bytes) = bytes / 2^20

function study_table(name, v, s₀s, nη_factors, n)
    σ = v.σ
    nx, nu, ny = E.npoints_x(σ), E.npoints_u(σ), E.npoints_y(σ)
    today_bytes = 2 * 8 * length(σ.c)
    @printf("\n## %s  (nx, nu, ny) = (%d, %d, %d); today's σ + L: %.1f MiB\n", name, nx, nu, ny, mb(today_bytes))
    @printf("box: x %.3f..%.3f (knots %.3f..%.3f), u %.3f..%.3f (knots %.3f..%.3f)\n",
            v.x_lo, v.x_hi, σ.x0, σ.x0 + (nx - 1) * σ.hx, v.u_lo, v.u_hi, σ.u0, σ.u0 + (nu - 1) * σ.hu)

    rng = StableRNG(20261127)
    pts = test_points(v, n, rng)
    refs = reference(v, pts)
    amb = count(i -> abs(refs[i].u_solved - pts[i].u) > 1e-6, eachindex(pts))
    nphys = count(physical, refs)
    @printf("test points: %d; today's solve lands on another u (σ non-monotone): %d; today physical (U_s > 0, 0 < cs² < 1): %d\n",
            n, amb, nphys)

    for s₀ in s₀s, fac in nη_factors
        nη = round(Int, fac * nu)
        t0 = time()
        r = build_reindexed(v, s₀, nη)
        tb = time() - t0
        c = compare(v, r, pts, refs)
        F_bytes = 8 * length(r.F.c)
        ab_bytes = 8 * 2 * (nx + 2) * (ny + 2)
        @printf("\n### s₀ = %g, nη = %d (%.2g × nu); build %.0f s\n", s₀, nη, fac, tb)
        @printf("memory: F + a, b %.1f MiB (%.2f × today); with the u field G %.1f MiB (%.2f × today)\n",
                mb(F_bytes + ab_bytes), (F_bytes + ab_bytes) / today_bytes,
                mb(2F_bytes + ab_bytes), (2F_bytes + ab_bytes) / today_bytes)
        @printf("columns non-monotone in η: %d of %d; columns reaching s ≤ 0: %d\n", r.nonmono_cols, r.ncols, r.neg_s_cols)
        println("quantity        p50 / p99 / p999 / max")
        println("U   rel         ", qline(c.eU))
        println("p   rel         ", qline(c.ep))
        println("h   rel         ", qline(c.eh))
        println("cs² abs         ", qline(c.ecs))
        println("U_s rel (T)     ", qline(c.eT))
        println("η outside [0,1] ", qline(c.eη))
        println("seed |Δu|       ", qline(c.du0))
        println("+1 Newton |Δu|  ", qline(c.du1))
        println("+2 Newton |Δu|  ", qline(c.du2))
        @printf("where today is physical: new U_s ≤ 0 at %d, new cs² ∉ (0,1) at %d\n", c.bad_T, c.bad_cs)
    end
end

# ---------------------------------------------------------------------------
# Held-out comparison: both representations against a finer table
# ---------------------------------------------------------------------------

"""Every other point on each axis; the omitted points become the test of both representations."""
function coarsen(t::RawTable{T}) where {T}
    ix, iu, iy = 1:2:length(t.logρ), 1:2:length(t.logT), 1:2:length(t.Yₑ)
    return RawTable{T}(t.logρ[ix], t.logT[iu], t.Yₑ[iy], copy(t.field_names), [d[ix, iu, iy] for d in t.field_data],
                       copy(t.field_index), copy(t.attribute_names), copy(t.attribute_values), copy(t.attribute_index))
end

"""The coarse view relabelled to the fine view's κ, so that ρ* means the same in both."""
function relabel_κ(v, κ_new)
    dx = log10(κ_new) - log10(v.κ)
    sh(b) = BsplineView3(b.c, b.x0 + dx, b.hx, b.u0, b.hu, b.y0, b.hy)
    return EOSTableView(sh(v.σ), sh(v.L), κ_new, v.shift_hat, v.conv_t, v.inv_c², v.x_lo + dx, v.x_hi + dx,
                        v.u_lo, v.u_hi, v.y_lo, v.y_hi, v.x_ext_lo + dx, v.x_ext_hi + dx, v.u_ext_lo, v.u_ext_hi,
                        v.ext_slope_floor_σ, v.ext_slope_floor_L, v.cs²_ext_cap, v.max_iter, v.fingerprint,
                        v.s_width_min)
end

function errline(label, refs, pts, f)
    n = length(refs)
    eU, ep, ecs, eT = zeros(n), zeros(n), zeros(n), zeros(n)
    Threads.@threads for i in 1:n
        q = f(pts[i])
        eU[i] = rel_diff(q.U, refs[i].U)
        ep[i] = rel_diff(q.p, refs[i].p)
        ecs[i] = abs(q.cs² - refs[i].cs²)
        eT[i] = rel_diff(q.U_s, refs[i].U_s)
    end
    @printf("%-22s U %s | p %s | cs² %s | U_s %s\n", label, q2(eU), q2(ep), q2(ecs), q2(eT))
end
q2(v) = @sprintf("%.1e/%.1e/%.1e", pct(v, 0.5), pct(v, 0.99), maxv(v))

function held_out(name, raw, opts, vf, s₀s, nη_factors, n)
    vc = relabel_κ(EOSTableView(E.build_eos(coarsen(raw), opts)), vf.κ)
    nu = E.npoints_u(vc.σ)
    @printf("\n### held out: %s at half resolution (nu = %d) against the full table; p50/p99/max\n", name, nu)
    rng = StableRNG(1127)
    pts = test_points(vc, n, rng)
    pts = [(ρ★=p.ρ★, s=E.bspline_eval3(vf.σ, log10(p.ρ★), p.u, p.y).f, y=p.y, u=p.u) for p in pts]
    refs = reference(vf, pts)
    errline("today, half", refs, pts, p -> E.evaluate(vc, p.ρ★, p.s, p.y, NaN))
    for s₀ in s₀s, fac in nη_factors
        r = build_reindexed(vc, s₀, round(Int, fac * nu))
        errline(@sprintf("η, s₀=%g, nη=%.2g nu", s₀, fac), refs, pts, p -> eval_new(vc, r, p.ρ★, p.s, p.y))
    end
end

# ---------------------------------------------------------------------------
# The seed-table variant: G's u as the warm start of today's solve
# ---------------------------------------------------------------------------

"""Today's solve iterations from a cold start and from G's prediction, at full and half resolution."""
function seed_iterations(name, raw, opts, vf, n)
    vc = relabel_κ(EOSTableView(E.build_eos(coarsen(raw), opts)), vf.κ)
    pts = test_points(vc, n, StableRNG(7))
    pts = [(ρ★=p.ρ★, s=E.bspline_eval3(vf.σ, log10(p.ρ★), p.u, p.y).f, y=p.y, u=p.u) for p in pts]
    vf0 = unseeded(vf)
    cold = [E.evaluate(vf0, p.ρ★, p.s, p.y, NaN) for p in pts]
    it = [c.iters for c in cold]
    @printf("\n### seed table: %s; today's solve iterations, cold: mean %.2f, p99 %d, max %d\n", name,
            sum(it) / n, pct(it, 0.99), maximum(it))
    for (lbl, vv, fac) in (("G full, nη = nu", vf, 1.0), ("G half, nη = nu/2", vc, 1.0), ("G half, nη = nu/4", vc, 0.5))
        r = build_reindexed(vv, 0.01, round(Int, fac * E.npoints_u(vv.σ)))
        seeded = map(pts) do p
            x = log10(p.ρ★)
            u0 = clamp(sample(r.G, x, η_at(r, x, p.s, p.y), p.y), vf.u_ext_lo, vf.u_ext_hi)
            return E.evaluate(vf0, p.ρ★, p.s, p.y, u0)
        end
        it = [c.iters for c in seeded]
        moved = count(i -> abs(seeded[i].u_solved - cold[i].u_solved) > 1e-10, eachindex(pts))
        @printf("%-18s %5.1f MiB; iters mean %.2f p99 %d max %d; 0..6: %s; other root than cold: %d\n", lbl,
                mb(8 * (length(r.G.c) + length(r.A.c) ÷ 2)), sum(it) / n, pct(it, 0.99), maximum(it),
                string([count(==(k), it) for k in 0:6]), moved)
    end
end

function main(args)
    dir = isempty(args) || startswith(args[1], "--") ? "" : args[1]
    flags = parse_flags(filter(a -> startswith(a, "--"), args))
    s₀s = parse.(Float64, split(get(flags, "s0", "1e-2,1"), ","))
    facs = parse.(Float64, split(get(flags, "neta", "1,1.5,2"), ","))
    n = parse(Int, get(flags, "n", "20000"))
    only = get(flags, "only", nothing)
    keep(nm) = only === nothing || occursin(lowercase(only), lowercase(nm))
    tables = Tuple{String,Any,BuildOptions}[]
    keep("synthetic") && push!(tables, ("synthetic", make_synthetic_table(), BuildOptions()))
    if !isempty(dir)
        for tab in REAL_TABLES
            keep(tab.name) || continue
            path = joinpath(dir, tab.file)
            isfile(path) || (@warn "skipping $(tab.name): not found" path; continue)
            push!(tables, (tab.name, read_stellarcollapse(path), BuildOptions(; m_B_table_g=tab.m_B)))
        end
    end
    for (name, raw, opts) in tables
        vf = EOSTableView(E.build_eos(raw, opts))
        get(flags, "skip-full", "false") == "true" || study_table(name, vf, s₀s, facs, n)
        get(flags, "skip-held", "false") == "true" || held_out(name, raw, opts, vf, s₀s, facs, n)
        get(flags, "skip-seed", "false") == "true" || seed_iterations(name, raw, opts, vf, n)
        flush(stdout)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)

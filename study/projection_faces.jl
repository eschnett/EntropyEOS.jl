# Behaviour study for the proposed two-level projection (CODE.md, "Two-level
# projection (proposed)"). That design solves the entropy faces as an
# active-set problem. Its termination and uniqueness rest on three
# assumptions, measured here on the real tables:
#
#   1. Along an entropy face the momentum residual of `inner_solve_w`, taken at
#      s = s_edge(D/cosh w, Yₑ) (the physical window's s_min or s_max at the
#      density actually written), is monotone in w.
#   2. The outer function G(s) = f₂(s, w*(s)), the energy residual at the inner
#      solve's root, is monotone in s at fixed D, S, Yₑ.
#   3. The face residual has a unique root on [0, w_max], and an Illinois root
#      find on that bracket converges in a bounded number of evaluations.
#
# On 1: f₁ = sinh w − cosh w·V = cosh w·φ with φ = tanh w − V. f₁ itself is
# not monotone wherever V > 1, even at fixed s and on an ideal gas, so its
# monotonicity is the wrong test; φ has the same sign and roots, and is what
# `f1_converged` tests. Both are reported. A control evaluates φ at fixed s
# (the edge entropy at the face's root), which is the inner solve's own claim
# (it holds for any EOS with cs² < 1). A violation that the control does not
# share comes from the edge's density dependence.
#
# Neither f₁ nor the monotonicity of f₂ depends on τ (f₂ is affine in τ, with
# an s-independent denominator), so 1–3 are properties of (D, S_par, S_perp,
# B², Yₑ). τ only decides which face is active. The script scales it down
# (aimed cold) and up (aimed hot) as TreeGRRMHD's `bin/endpoint_measure.jl` did
# on the synthetic table. It then reports which face the edge-consistent
# diagnosis selects, and whether G's sign over the window agrees with it.
#
# Sampling: log ρ uniform over the table's box (κ-rescaled), Yₑ uniform,
# w ∈ [0, 4], and either s uniform in the window at ρ ("s", TreeGRRMHD's
# choice, which puts most states in the radiation-dominated corner) or
# log T uniform over the table ("logT", where neutron-star matter lives).
# Hydrodynamic (B² = 0) or MHD (B² = z·10^U(−2,1), random angle between v
# and B). τ is scaled by 10^−U(0.05,3) (cold) or 10^U(0.3,6) (hot).
#
# Grids: the face residual on the union of a grid uniform in w and one uniform
# in log cosh w; G on the union of a grid uniform in s and one uniform in
# log(s − s_lo + 10⁻⁴), spanning the windows the face reaches. A violation is a
# decrease below the running maximum: for φ in its own units (those of the
# convergence test), for G as a fraction of G's span. "Valid" restricts the
# face to the rest of the valid range, acosh(D/ρ_max) ≤ w ≤ min(w_cap,
# acosh(D/ρ_min)), with [ρ_min, ρ_max] the table's density box and w_cap =
# acosh(100); states with D > ρ_max·cosh(w_cap) have no valid part and are left
# out of those columns. G is "valid" where the inner solve converged with
# ρ* ∈ [ρ_min, ρ_max], w* ≤ w_cap, and s inside the window at ρ*. A face whose
# root lies outside its valid part is a corner, where another face is active
# too.
#
#   julia -t 8 --project=study study/projection_faces.jl --tables=/path/to/tables \
#         [--n=2000] [--only=LS220] [--nw=800] [--ns=400] [--synthetic] [--examples]

include("common.jl")

const FLAGS = parse_flags(ARGS)
const N = parse(Int, get(FLAGS, "n", "2000"))
const NW = parse(Int, get(FLAGS, "nw", "800"))
const NS = parse(Int, get(FLAGS, "ns", "400"))
const TOL = E.con2prim_tol(Float64)
const NOISE = 1e-12        # violations below this are rounding, not shape
const W_CAP = acosh(100.0)

"""Union of a grid uniform in `t ∈ [0, 1]` mapped by `f` and a uniform one on `[a, b]`, sorted."""
function mixed_grid(a, b, n, f)
    h = n ÷ 2
    return unique!(sort!(vcat([a + (b - a) * j / h for j in 0:h], [f(j / h) for j in 0:h])))
end

w_grid(w_max, n) = mixed_grid(0.0, w_max, n, t -> acosh(cosh(w_max)^t))

function s_grid(s_lo, s_hi, n)
    δ = 1e-4
    l0, l1 = log(δ), log(s_hi - s_lo + δ)
    return mixed_grid(s_lo, s_hi, n, t -> clamp(s_lo + exp(l0 + (l1 - l0) * t) - δ, s_lo, s_hi))
end

"""Largest decrease below the running maximum over kept points, with its abscissa."""
function worst_drop(x, g, keep; scale=1.0)
    m = -Inf
    worst, at = 0.0, NaN
    for j in eachindex(g)
        keep(j) || continue
        d = (m - g[j]) / scale
        d > worst && (worst = d; at = x[j])
        g[j] > m && (m = g[j])
    end
    return worst, at
end

"""Sign changes of `g` over kept points, and where (between which abscissae) they are."""
function roots_at(x, g, keep)
    at = Float64[]
    prev, xprev = 0, NaN
    for j in eachindex(g)
        keep(j) || continue
        sg = g[j] > 0 ? 1 : g[j] < 0 ? -1 : 0
        sg == 0 && continue
        prev != 0 && sg != prev && push!(at, (xprev + x[j]) / 2)
        prev, xprev = sg, x[j]
    end
    return at
end

s_edge(v, e, ρ, ye) = let sr = srange(v, ρ, ye)
    e === :min ? sr.s_min : sr.s_max
end

"""
Illinois on [0, w_max] for the face residual, in f₁ (`:f1`) or φ = f₁/cosh w
(`:φ`). `evals` counts residual evaluations, the two bracket ends included.
"""
function edge_solve(v, e, c, ye, B², opts, var)
    res(w) = E.residuals(v, c.D, c.τ, ye, c.S_par, c.S_perp, B², s_edge(v, e, c.D / cosh(w), ye), w, NaN,
                         opts.τ_floor_rel)
    val(r) = var === :f1 ? r.f1 : r.f1 / r.coshw
    a, b = 0.0, opts.w_max
    ra, rb = res(a), res(b)
    rb.f1 < 0 && return (; ok=true, capped=true, evals=2, r=rb)
    ra.f1 >= 0 && return (; ok=true, capped=false, evals=2, r=ra)
    fa, fb, side, r = val(ra), val(rb), 0, rb
    for k in 1:200
        w = (a * fb - b * fa) / (fb - fa)
        (a < w < b) || (w = (a + b) / 2)
        r = res(w)
        E.f1_converged(r, TOL) && return (; ok=true, capped=false, evals=k + 2, r)
        if r.f1 < 0
            a, fa = w, val(r)
            side == -1 && (fb /= 2)
            side = -1
        else
            b, fb = w, val(r)
            side == 1 && (fa /= 2)
            side = 1
        end
        b - a <= 4eps(b) && return (; ok=false, capped=false, evals=k + 2, r)
    end
    return (; ok=false, capped=false, evals=202, r)
end

"""One random valid state, conserved, with its two τ scalings."""
function sample_state(v, mode, sampling, rng)
    ρ = exp10(v.x_lo + (v.x_hi - v.x_lo) * rand(rng))
    ye = v.y_lo + (v.y_hi - v.y_lo) * rand(rng)
    s = if sampling === :s
        sr = srange(v, ρ, ye)
        sr.s_min + (sr.s_max - sr.s_min) * rand(rng)
    else
        E.sigma_extended(v, ρ, v.u_lo + (v.u_hi - v.u_lo) * rand(rng), ye)
    end
    w = 4 * rand(rng)
    h = E.evaluate(v, ρ, s, ye, NaN).h
    z = ρ * h * cosh(w)^2
    B² = mode === :hydro ? 0.0 : z * exp10(-2 + 3 * rand(rng))
    cosvB = mode === :hydro ? 0.0 : 2 * rand(rng) - 1
    c = E.prim2con(v, ρ, s, ye, w, B², cosvB, NaN)
    fcold = exp10(-(0.05 + 2.95 * rand(rng)))
    fhot = exp10(0.3 + 5.7 * rand(rng))
    return (; ρ, s, ye, w, c, B², fcold, fhot)
end

function study_face(v, e, c, ye, B², opts, ws, valid)
    D = c.D
    g = similar(ws)
    s_lo, s_hi = Inf, -Inf
    for (j, w) in pairs(ws)
        se = s_edge(v, e, D / cosh(w), ye)
        valid(j) && (s_lo = min(s_lo, se); s_hi = max(s_hi, se))
        g[j] = E.residuals(v, D, c.τ, ye, c.S_par, c.S_perp, B², se, w, NaN, opts.τ_floor_rel).f1
    end
    φ = g ./ cosh.(ws)
    all_(_) = true
    f1drop, _ = worst_drop(ws, g, valid; scale=1.0)
    φdrop_val, φat = worst_drop(ws, φ, valid)
    φdrop_all, _ = worst_drop(ws, φ, all_)
    roots_val = roots_at(ws, φ, valid)
    roots_all = roots_at(ws, φ, all_)
    il_f1 = edge_solve(v, e, c, ye, B², opts, :f1)
    il_φ = edge_solve(v, e, c, ye, B², opts, :φ)

    # Control: φ at a fixed entropy, the edge's at the face's root (or at the
    # end of the valid part when the root lies beyond it).
    jv = findall(valid, eachindex(ws))
    w_ctl = isempty(jv) ? il_φ.r.w : clamp(il_φ.r.w, ws[first(jv)], ws[last(jv)])
    s_fix = s_edge(v, e, D / cosh(w_ctl), ye)
    φf = [E.residuals(v, D, c.τ, ye, c.S_par, c.S_perp, B², s_fix, w, NaN, opts.τ_floor_rel).f1 / cosh(w)
          for w in ws]
    ctl_drop, _ = worst_drop(ws, φf, valid)
    return (; f1drop, φdrop_val, φat, φdrop_all, roots_val, roots_all, il_f1, il_φ, ctl_drop, s_lo, s_hi)
end

function study_state(v, st, opts, ws)
    c, ye, B² = st.c, st.c.D_Y / st.c.D, st.B²
    D = c.D
    ρ_min, ρ_max = exp10(v.x_lo), exp10(v.x_hi)
    w_lo = acosh(max(D / ρ_max, 1.0))
    w_hi = min(W_CAP, acosh(max(D / ρ_min, 1.0)))
    valid(j) = w_lo <= ws[j] <= w_hi
    has_valid = count(valid, eachindex(ws)) > 1
    denom0 = max(c.τ, opts.τ_floor_rel * D)

    faces = map(e -> study_face(v, e, c, ye, B², opts, ws, valid), (:min, :max))
    # The face's τ_model at its edge-consistent root, so each aim can put its own τ back.
    τface = map(f -> f.il_φ.r.f2 * denom0 + c.τ, faces)

    in_valid(f) = !f.il_φ.capped && w_lo <= f.il_φ.r.w <= w_hi
    if !has_valid
        aims = map(_ -> (; class=:none, nroot=0, agree=true), (1, 2))
        return (; faces, has_valid, Gdrop=NaN, nG=0, aims, D, ye, B², st)
    end

    # Claim 2: G(s) over the union of the windows the valid part of the face reaches.
    ss = s_grid(min(faces[1].s_lo, faces[2].s_lo), max(faces[1].s_hi, faces[2].s_hi), NS)
    τm = similar(ss)
    gvalid = falses(length(ss))
    w_seed, u_seed = 0.0, NaN
    for (j, s) in pairs(ss)
        r, _ = E.inner_solve_w(v, D, c.τ, ye, c.S_par, c.S_perp, B², s, E.w_floor(v, D, opts.w_max), opts.w_max,
                               opts.τ_floor_rel, w_seed, u_seed, opts.max_iter_1d, opts.tol)
        τm[j] = r.f2 * denom0 + c.τ
        sr = srange(v, r.ρ, ye)
        gvalid[j] = E.f1_converged(r, opts.tol) && ρ_min <= r.ρ <= ρ_max && r.w <= W_CAP && sr.s_min <= s <= sr.s_max
        w_seed, u_seed = r.w, r.pt.u_solved
    end
    nG = count(gvalid)
    span = nG > 1 ? maximum(τm[gvalid]) - minimum(τm[gvalid]) : NaN
    Gdrop, _ = nG > 1 ? worst_drop(ss, τm, j -> gvalid[j]; scale=span) : (NaN, NaN)

    # The aims: which face the edge-consistent diagnosis selects, and whether
    # G's sign over the valid s agrees (cold: G > 0 throughout; hot: G < 0).
    aims = map((st.fcold, st.fhot)) do f
        τ = c.τ * f
        dn = max(τ, opts.τ_floor_rel * D)
        class = (τface[1] - τ) / dn > 0 ? :cold : (τface[2] - τ) / dn < 0 ? :hot : :interior
        (class === :cold && !in_valid(faces[1]) || class === :hot && !in_valid(faces[2])) && (class = :corner)
        Gτ = (τm .- τ) ./ dn
        nroot = length(roots_at(ss, Gτ, j -> gvalid[j]))
        agree = nG == 0 || (class === :cold ? all(Gτ[gvalid] .> 0) :
                            class === :hot ? all(Gτ[gvalid] .< 0) : true)
        (; class, nroot, agree)
    end
    return (; faces, has_valid, Gdrop, nG, aims, D, ye, B², st)
end

pc(n, N) = @sprintf("%.3g%%", 100 * n / max(N, 1))
q3(x) = isempty(x) ? "–" : @sprintf("%d/%d/%d", pct(x, 0.5), pct(x, 0.99), maxv(x))
hist(x) = join(["$k:$(count(==(k), x))" for k in sort(unique(x))], " ")

function report(name, recs, mode, sampling)
    recs = filter(r -> r.has_valid, recs)
    n = length(recs)
    for (i, e) in enumerate((:min, :max))
        F = [r.faces[i] for r in recs]
        ev_f1 = [f.il_f1.evals for f in F if !f.il_f1.capped]
        ev_φ = [f.il_φ.evals for f in F if !f.il_φ.capped]
        nmulti = count(f -> length(f.roots_val) > 1, F)
        @printf("| %s | %s | %s | s_%s | %d | %s | %s | %s | %.2g | %s | %s | %s | %s | %s | %d |\n",
                name, mode, sampling, e, n, pc(count(f -> f.f1drop > NOISE, F), n),
                pc(count(f -> f.φdrop_val > NOISE, F), n), pc(count(f -> f.ctl_drop > NOISE, F), n),
                maximum(f -> f.φdrop_val, F), pc(nmulti, n), hist([length(f.roots_all) for f in F]),
                q3(ev_f1), q3(ev_φ), pc(count(f -> !f.il_φ.ok || !f.il_f1.ok, F), n),
                count(f -> f.il_φ.capped, F))
    end
end

function report_G(name, recs, mode, sampling)
    n = length(recs)
    G = [r.Gdrop for r in recs if r.nG > 1]
    A = [a for r in recs for a in r.aims if a.class !== :none]
    ncl(c) = count(a -> a.class === c, A)
    @printf("| %s | %s | %s | %d | %s | %.2g | %s | %d/%d/%d/%d | %d | %d |\n", name, mode, sampling, length(G),
            pc(count(>(NOISE), G), length(G)), isempty(G) ? NaN : maximum(G), pc(count(>(1e-3), G), length(G)),
            ncl(:cold), ncl(:hot), ncl(:corner), ncl(:interior), count(a -> !a.agree, A),
            count(a -> a.class === :interior && a.nroot > 1, A))
end

function examples(name, recs)
    k = 0
    for r in recs, (i, e) in enumerate((:min, :max))
        f = r.faces[i]
        length(f.roots_val) > 1 || continue
        @printf("  %s s_%s: D=%.3g Yₑ=%.3f B²/D=%.2g (sampled ρ=%.3g s=%.3g w=%.2f), roots at w ≈ %s, ρ ≈ %s; φ drop %.2g, control %.2g\n",
                name, e, r.D, r.ye, r.B² / r.D, r.st.ρ, r.st.s, r.st.w,
                join([@sprintf("%.3f", w) for w in f.roots_val], ", "),
                join([@sprintf("%.3g", r.D / cosh(w)) for w in f.roots_val], ", "), f.φdrop_val, f.ctl_drop)
        (k += 1) >= 6 && return
    end
end

function main()
    dir = get(FLAGS, "tables", get(ENV, "ENTROPYEOS_TABLE_DIR", ""))
    tabs = load_tables(dir; only=get(FLAGS, "only", nothing))
    haskey(FLAGS, "synthetic") || filter!(t -> t[1] != "synthetic", tabs)
    opts = E.Con2PrimOptions{Float64}()
    ws = w_grid(opts.w_max, NW)
    @printf("n = %d per row, %d w points, %d s points, %d threads\n", N, length(ws), NS, Threads.nthreads())
    all_recs = Dict{Any,Any}()
    for (ti, (name, v)) in enumerate(tabs), (mi, mode) in enumerate((:hydro, :mhd)),
        (si, sampling) in enumerate((:s, :logT))
        recs = Vector{Any}(undef, N)
        Threads.@threads for i in 1:N
            rng = StableRNG(1_000_000 * (100ti + 10mi + si) + i)
            recs[i] = study_state(v, sample_state(v, mode, sampling, rng), opts, ws)
        end
        all_recs[(name, mode, sampling)] = recs
    end
    keys_sorted = [(name, mode, sampling) for (name, _) in tabs for mode in (:hydro, :mhd) for sampling in (:s, :logT)]

    println("\nClaims 1 and 3, the face residual along each entropy face. Percentages of states; \"valid\" is the part")
    println("of the face inside the rest of the valid range. Roots counted on φ. Illinois evaluations p50/p99/max,")
    println("over states whose root lies below w_max.\n")
    println("| table | mode | sampling | face | states | f₁ non-monotone (valid) | φ non-monotone (valid) | φ at fixed s non-monotone (valid) | worst φ drop (valid) | >1 root (valid) | roots on [0, w_max] | Illinois on f₁ | Illinois on φ | Illinois failed | capped |")
    println("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for k in keys_sorted
        report(k[1], all_recs[k], k[2], k[3])
    end

    println("\nClaim 2, G(s) = f₂ at the inner root, over the valid s. Drop as a fraction of G's span. Diagnoses")
    println("over both aims (corner: the selected face's root lies outside its valid part); \"disagrees\" counts")
    println("cold or hot diagnoses where G's sign over the valid s contradicts them (cold but G ≤ 0 somewhere, hot")
    println("but G ≥ 0 somewhere); the last column counts interior diagnoses where G has more than one valid root.\n")
    println("| table | mode | sampling | states with valid s | G non-monotone | worst drop | drop > 1e-3 | cold/hot/corner/interior | disagrees | interior, >1 root |")
    println("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for k in keys_sorted
        report_G(k[1], all_recs[k], k[2], k[3])
    end

    if haskey(FLAGS, "examples")
        println("\nStates with more than one root on the valid part of a face:")
        for k in keys_sorted
            examples(join(string.(k), "/"), all_recs[k])
        end
    end
end

# Run as a script; `include` it to reuse the pieces.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

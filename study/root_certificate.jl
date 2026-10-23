# Behaviour study: a cheap certificate that the inner solve's root is the lowest
# one. It was measured for a proposed rule, "return the lowest root", which
# needs a scan of [0, w_max] only where several roots are possible; the rule was
# rejected (HISTORY.md, "State policy flags", 2026-10-22), and these numbers are
# part of why.
#
# The certificate. At fixed (s, Yₑ), with z = D·h·cosh w, residuals() has
# z_w = z·(1 − cs²)·tanh w. At a root of φ = tanh w − V (V = tanh w there),
#
#   dφ/dw · cosh² w = 1 − tanh² w·(1 − θ(1 − cs²))  ≥  1 − cs²·tanh² w,
#
# with θ = (v∥² + v⊥²·z/(z + B²))/V² ∈ (0, 1]. So every crossing rises, and
# there is at most one, on any interval of w where cs²·tanh² w < 1, for any B².
# φ(0) = −V ≤ 0, so if that holds on [0, w_safe] and φ(w_safe) > 0, the bracket
# [0, w_safe] holds exactly one root and it is the lowest; no scan is needed.
#
# cs² varies along the path only through ρ = D/cosh w, at fixed Yₑ. So a bound
# M(x, Yₑ) on cs² over all temperatures, per cell in x = log10 ρ and Yₑ, gives
# w_safe by walking the cells the path crosses: in a cell with M > 1 the
# condition holds below atanh(M^(-1/2)). The bound is taken by sampling, so it
# is not rigorous; the script counts the states it certifies falsely.
#
# Measured per table, over the states of `projection_faces.jl`'s sampling:
# how often w_true < w_safe (no scan), for several bound layouts; false
# certificates (several roots on [0, w_true] although certified); and, for
# reference, how often the fixed-s residual has several roots at all.
#
#   julia -t 8 --project=study study/root_certificate.jl --tables=/path/to/tables \
#         [--n=2000] [--only=LS220] [--nw=800]

include("projection_faces.jl")

"""
Upper bound on cs² per x cell (width `dx` decades, over the extended box) and Yₑ
bin (`ny` bins), sampled at `nxs` × `nus` × `nys` points per cell over all
temperatures of the extended box. Non-finite cs² counts as unbounded.
"""
function cs2_bound(v; dx, ny, nxs=5, nus=160, nys=3)
    x0, x1 = Float64(v.x_ext_lo), Float64(v.x_ext_hi)
    nx = ceil(Int, (x1 - x0) / dx)
    M = fill(-Inf, nx, ny)
    Threads.@threads for i in 1:nx
        for k in 1:ny
            m = -Inf
            for a in 0:(nxs - 1), c in 0:(nys - 1)
                x = min(x0 + dx * (i - 1 + a / (nxs - 1)), x1)
                yb = v.y_lo + (v.y_hi - v.y_lo) * (k - 1 + c / (nys - 1)) / ny
                ρ = exp10(x)
                for u in u_samples(v, nus)
                    s = E.sigma_extended(v, ρ, u, yb)
                    cs² = E.evaluate(v, ρ, s, yb, u).cs²
                    m = isfinite(cs²) ? max(m, cs²) : Inf
                end
            end
            M[i, k] = m
        end
    end
    return (; M, x0, dx, ny)
end

"""
The same bound with entropy bins as well: `ns` bins uniform in asinh(s) over the
sampled range. A path at fixed s reads one entropy bin; a cell no sample reached
falls back to the cell's maximum over all entropies.
"""
function cs2_bound_s(v; dx, ny, ns, nxs=5, nus=160, nys=3)
    x0, x1 = Float64(v.x_ext_lo), Float64(v.x_ext_hi)
    nx = ceil(Int, (x1 - x0) / dx)
    a_lo, a_hi = Inf, -Inf
    for x in range(x0, x1; length=41), u in range(v.u_ext_lo, v.u_ext_hi; length=41), y in (v.y_lo, v.y_hi)
        a = asinh(E.sigma_extended(v, exp10(x), u, y))
        a_lo, a_hi = min(a_lo, a), max(a_hi, a)
    end
    a_lo -= 0.05 * (a_hi - a_lo)
    a_hi += 0.05 * (a_hi - a_lo)
    M = fill(-Inf, nx, ns, ny)
    Threads.@threads for i in 1:nx
        for k in 1:ny
            for a in 0:(nxs - 1), c in 0:(nys - 1)
                x = min(x0 + dx * (i - 1 + a / (nxs - 1)), x1)
                yb = v.y_lo + (v.y_hi - v.y_lo) * (k - 1 + c / (nys - 1)) / ny
                ρ = exp10(x)
                for u in u_samples(v, nus)
                    s = E.sigma_extended(v, ρ, u, yb)
                    cs² = E.evaluate(v, ρ, s, yb, u).cs²
                    j = clamp(1 + floor(Int, ns * (asinh(s) - a_lo) / (a_hi - a_lo)), 1, ns)
                    M[i, j, k] = isfinite(cs²) ? max(M[i, j, k], cs²) : Inf
                end
            end
            colmax = maximum(M[i, :, k])
            for j in 1:ns
                M[i, j, k] == -Inf && (M[i, j, k] = colmax)
            end
        end
    end
    return (; M, x0, dx, ny, ns, a_lo, a_hi)
end

"""View one entropy bin of an s-binned bound as an (x, Yₑ) bound."""
function slice_s(B, s)
    j = clamp(1 + floor(Int, B.ns * (asinh(s) - B.a_lo) / (B.a_hi - B.a_lo)), 1, B.ns)
    return (; M=view(B.M, :, j, :), B.x0, B.dx, B.ny)
end

"""
Temperatures to sample: uniform over the extended box, plus a dense cluster at
each seam of the physical box, where cs² has narrow spikes (DD2's hot seam
reaches 1.17 within ~0.01 in log T).
"""
function u_samples(v, nus)
    us = [v.u_ext_lo + (v.u_ext_hi - v.u_ext_lo) * b / (nus - 1) for b in 0:(nus - 1)]
    for uc in (v.u_lo, v.u_hi), d in range(-0.1, 0.1; length=81)
        push!(us, clamp(uc + d, v.u_ext_lo, v.u_ext_hi))
    end
    return us
end

"""Collapse a bound to fewer Yₑ bins (1 = one row for all Yₑ)."""
function coarsen_y(B)
    return (; M=maximum(B.M; dims=2), B.x0, B.dx, ny=1)
end

"""
The largest w ≤ w_max up to which cs²·tanh² w < 1 is certified along the path
ρ = D/cosh w at this Yₑ. Densities outside the bound's box use its edge cell.
"""
function w_safe(B, D, ye, w_max)
    nx = size(B.M, 1)
    k = clamp(1 + floor(Int, B.ny * (ye - Y_LO[]) / (Y_HI[] - Y_LO[])), 1, B.ny)
    x = log10(D)
    i = clamp(1 + floor(Int, (x - B.x0) / B.dx), 1, nx)
    w = 0.0
    while true
        x_low = i == 1 ? -Inf : B.x0 + B.dx * (i - 1)   # the cell's low-density edge
        w_exit = x_low == -Inf ? w_max : min(w_max, acosh(max(1.0, D / exp10(x_low))))
        m = B.M[i, k]
        if m > 1
            wM = m == Inf ? 0.0 : atanh(1 / sqrt(m))
            wM <= w && return w
            wM < w_exit && return wM
        end
        (w_exit >= w_max || i == 1) && return w_max
        w, i = w_exit, i - 1
    end
end

const Y_LO = Ref(0.0)
const Y_HI = Ref(1.0)

"""φ at fixed s on the grid, and the roots on [0, w_true] and on [0, w_max]."""
function fixed_s_roots(v, st, opts, ws)
    c, ye = st.c, st.c.D_Y / st.c.D
    φ = [E.residuals(v, c.D, c.τ, ye, c.S_par, c.S_perp, st.B², st.s, w, NaN, opts.τ_floor_rel).f1 / cosh(w)
         for w in ws]
    below = roots_at(ws, φ, j -> ws[j] <= st.w * (1 + 1e-9) + 1e-9)
    every = roots_at(ws, φ, _ -> true)
    return (; nbelow=length(below), nall=length(every))
end

function main_cert()
    dir = get(FLAGS, "tables", get(ENV, "ENTROPYEOS_TABLE_DIR", ""))
    tabs = load_tables(dir; only=get(FLAGS, "only", nothing))
    haskey(FLAGS, "synthetic") || filter!(t -> t[1] != "synthetic", tabs)
    opts = E.Con2PrimOptions{Float64}()
    ws = w_grid(opts.w_max, NW)
    layouts = [("global", 0), ("x 0.25 dec", 1), ("x 0.05 dec", 2), ("x 0.25 × Yₑ 24", 3), ("x 0.05 × Yₑ 24", 4),
               ("x 0.25 × s 32 × Yₑ 24", 5)]
    println("| table | sampling | W bin | states | >1 root on [0, w_max] | not lowest | " *
            join(["no scan, $(l[1])" for l in layouts], " | ") * " | false certificates |")
    println("| --- | --- | --- | --- | --- | --- | " * join(fill("---", length(layouts)), " | ") * " | --- |")
    for (ti, (name, v)) in enumerate(tabs)
        Y_LO[], Y_HI[] = v.y_lo, v.y_hi
        t0 = time()
        fine = cs2_bound(v; dx=0.05, ny=24)
        coarse = cs2_bound(v; dx=0.25, ny=24)
        bounds = [(; M=fill(maximum(fine.M), 1, 1), fine.x0, dx=1e9, ny=1), coarsen_y(coarse), coarsen_y(fine), coarse,
                  fine]
        sbound = cs2_bound_s(v; dx=0.25, ny=24, ns=32)
        @printf(stderr, "%s: bounds in %.1f s, max cs² %.3g\n", name, time() - t0, maximum(fine.M))
        for (si, sampling) in enumerate((:s, :logT)), (mi, mode) in enumerate((:hydro, :mhd))
            recs = Vector{Any}(undef, N)
            Threads.@threads for i in 1:N
                rng = StableRNG(1_000_000 * (100ti + 10mi + si) + i)
                st = sample_state(v, mode, sampling, rng)
                ye = st.c.D_Y / st.c.D
                r = fixed_s_roots(v, st, opts, ws)
                safe = [[w_safe(B, st.c.D, ye, opts.w_max) for B in bounds];
                        w_safe(slice_s(sbound, st.s), st.c.D, ye, opts.w_max)]
                recs[i] = (; st.w, r, safe)
            end
            for (bin, lo, hi) in (("all", 0.0, Inf), ("W < 2", 0.0, acosh(2.0)), ("2 ≤ W < 10", acosh(2.0), acosh(10.0)),
                                  ("W ≥ 10", acosh(10.0), Inf))
                sel = filter(r -> lo <= r.w < hi, recs)
                n = length(sel)
                n == 0 && continue
                multi = count(r -> r.r.nall > 1, sel)
                notlow = count(r -> r.r.nbelow > 1, sel)
                cols = [pc(count(r -> r.w < r.safe[j], sel), n) for j in eachindex(layouts)]
                false_c = join([count(r -> r.w < r.safe[j] && r.r.nbelow > 1, sel) for j in eachindex(layouts)], "/")
                println("| $name | $sampling, $mode | $bin | $n | $(pc(multi, n)) | $(pc(notlow, n)) | " *
                        join(cols, " | ") * " | $false_c |")
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_cert()
end

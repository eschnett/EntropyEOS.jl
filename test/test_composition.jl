# The composition on the device (CODE.md, "Composition"): the synthetic
# table's composition is multilinear, so trilinear interpolation must reproduce
# it to round-off everywhere; the real tables are covered in
# test_real_tables.jl, which reuses the helpers below.

# Allocation probes at top level; see the note in test_device_api.jl.
_comp(cv, ρ, T, y) = EntropyEOS.composition(cv, ρ, T, y).μ_ν
function _comp_allocs(cv, ρ, T, y)
    _comp(cv, ρ, T, y)
    return @allocated _comp(cv, ρ, T, y)
end

# The cell and corner weights of fractional 0-based indices, in Float64,
# written independently of the kernel's.
function _comp_corners(cv, fi, fj, fk)
    _, nx, nu, ny = size(cv.c)
    cell(f, n) = (fc = clamp(Float64(f), 0.0, n - 1.0); i = min(floor(Int, fc), n - 2); (i, fc - i))
    (i, a), (j, b), (k, c) = cell(fi, nx), cell(fj, nu), cell(fk, ny)
    return [(i + 1 + di, j + 1 + dj, k + 1 + dk, (di == 0 ? 1 - a : a) * (dj == 0 ? 1 - b : b) * (dk == 0 ? 1 - c : c))
            for di in 0:1, dj in 0:1, dk in 0:1]
end

"""
The X_h → 0 properties at fractional indices `(fi, fj, fk)` of the view `cv`
(of either precision), against a Float64 reference from its own nodes; the
names of those violated. `tol` is relative.
"""
function _comp_edge_violations(cv, fi, fj, fk, tol)
    E = EntropyEOS
    r = E.composition_index(cv, fi, fj, fk)
    bad = String[]
    c = cv.c
    num, den, Amax, Amin, Zmin, Zmax = 0.0, 0.0, -Inf, Inf, Inf, -Inf
    for (ii, jj, kk, w) in _comp_corners(cv, fi, fj, fk)
        xh, A, Z = Float64(c[E.COMP_X_H, ii, jj, kk]), Float64(c[E.COMP_Ā, ii, jj, kk]), Float64(c[E.COMP_Z̄, ii, jj, kk])
        num += w * xh * A
        den += w * xh
        if w * xh > 0
            Amin, Amax, Zmin, Zmax = min(Amin, A), max(Amax, A), min(Zmin, Z), max(Zmax, Z)
        end
    end
    if r.X_h == 0
        # The fallback, and only next to X_h = 0.
        (r.Ā == 1 && r.Z̄ == 0) || push!(bad, "X_h = 0 without the fallback")
        den <= 4 * floatmin(typeof(r.X_h)) || push!(bad, "X_h = 0 where the corners have heavy nuclei")
    else
        (r.Ā == 1 && r.Z̄ == 0 && Amin > 1) && push!(bad, "fallback with X_h > 0")
        Amin <= r.Ā <= Amax || push!(bad, "Ā outside its corners")
        Zmin <= r.Z̄ <= Zmax || push!(bad, "Z̄ outside its corners")
    end
    # X_h·Ā is the trilinear interpolant of the product field, which is
    # continuous across cells and vanishes with X_h.
    # Below floatmin the kernel takes X_h as 0 on purpose, so the floor is
    # absolute: floatmin·Ā.
    atol = (tol * den + 2 * Float64(floatmin(typeof(r.X_h)))) * max(Amax, 1.0)
    isapprox(Float64(r.X_h) * Float64(r.Ā), num; rtol=tol, atol=atol) ||
        push!(bad, "X_h·Ā is not the interpolated product")
    return bad
end

"""Up to `nmax` cells, as 0-based lower corners, whose corners mix X_h = 0 and X_h > 0."""
function _comp_edge_cells(cv; nmax=10_000)
    E = EntropyEOS
    _, nx, nu, ny = size(cv.c)
    xh = view(cv.c, E.COMP_X_H, :, :, :)
    out = NTuple{3,Int}[]
    for k in 1:(ny - 1), j in 1:(nu - 1), i in 1:(nx - 1)
        nz = count(>(0), (xh[i, j, k], xh[i + 1, j, k], xh[i, j + 1, k], xh[i + 1, j + 1, k], xh[i, j, k + 1],
                          xh[i + 1, j, k + 1], xh[i, j + 1, k + 1], xh[i + 1, j + 1, k + 1]))
        0 < nz < 8 && push!(out, (i - 1, j - 1, k - 1))
        length(out) >= nmax && return out
    end
    return out
end

"""Edge properties at `n` random points in edge cells, in Float64 and narrowed to Float32."""
function _test_comp_edge(cv, rng; n=2000)
    cells = _comp_edge_cells(cv)
    @test !isempty(cells)
    for (T, tol) in ((Float64, 1e-12), (Float32, 1e-5))
        cvT = T === Float64 ? cv : EntropyEOS.narrow(cv, T)
        bad = Dict{String,Int}()
        nfb = 0
        for _ in 1:n
            (i, j, k) = cells[rand(rng, 1:length(cells))]
            # A face or a corner now and then, where cells meet.
            f(i) = (r = rand(rng); T(i + (r < 0.1 ? 0.0 : r < 0.2 ? 1.0 : rand(rng))))
            fi, fj, fk = f(i), f(j), f(k)
            for b in _comp_edge_violations(cvT, fi, fj, fk, tol)
                bad[b] = get(bad, b, 0) + 1
            end
            nfb += EntropyEOS.composition_index(cvT, fi, fj, fk).X_h == 0
        end
        @test isempty(bad) || (@info "edge violations ($T)" bad; false)
        # Across a face, from the cells on either side: the two differ by one
        # ulp of the index, so by that times the product's change over a cell.
        jumps = 0.0
        for _ in 1:n
            (i, j, k) = cells[rand(rng, 1:length(cells))]
            i >= 1 || continue
            fj, fk = T(j + rand(rng)), T(k + rand(rng))
            lo = EntropyEOS.composition_index(cvT, prevfloat(T(i)), fj, fk)
            hi = EntropyEOS.composition_index(cvT, T(i), fj, fk)
            c = cvT.c
            scale = maximum(Float64(c[EntropyEOS.COMP_X_H, ii, jj, kk]) * Float64(c[EntropyEOS.COMP_Ā, ii, jj, kk])
                            for (ii, jj, kk, _) in vcat(vec(_comp_corners(cvT, i - 0.5, fj, fk)),
                                                        vec(_comp_corners(cvT, i + 0.5, fj, fk))))
            jumps = max(jumps, abs(Float64(lo.X_h * lo.Ā) - Float64(hi.X_h * hi.Ā)) / (scale * eps(T(i)) + 1e-300))
        end
        @test jumps <= 16
    end
    return nothing
end

@testset "composition" begin
    E = EntropyEOS
    tbl = make_synthetic_table(SyntheticOptions(; with_composition=true))
    eos = E.build_eos(tbl)
    v = EOSTableView(eos)
    ct = CompositionTable(tbl, eos)
    cv = CompositionView(ct)
    nx, nu, ny = size(tbl)
    lκ = log10(v.κ)

    @testset "build" begin
        @test isbitstype(Composition{Float64})
        @test ct.munu_offset == 0
        @test ct.light_fields == ["Xl"]
        @test q_np(cv) == q_np(ct) == E.Q_NP_MEV
        @test cv.κ === v.κ
        # The density origin is the EOS's spline's, to the bit.
        @test cv.x0 === v.σ.x0 && cv.hx === v.σ.hx
        @test fingerprint(cv) == fingerprint(ct) == fingerprint(eos) == fingerprint(v) != 0
        @test size(cv.c) == (E.COMP_NFIELDS, nx, nu, ny)
    end

    @testset "nodes are the datasets" begin
        f(n) = field(tbl, n)
        ok = true
        for k in 1:ny, j in 1:nu, i in 1:nx
            r = E.composition_index(cv, Float64(i - 1), Float64(j - 1), Float64(k - 1))
            ok &= r.μ_e === f("mu_e")[i, j, k] && r.μ̂ === f("muhat")[i, j, k] &&
                  r.μ_ν === f("mu_e")[i, j, k] - f("muhat")[i, j, k] &&
                  r.X_n === f("Xn")[i, j, k] && r.X_p === f("Xp")[i, j, k] && r.X_α === f("Xa")[i, j, k] &&
                  r.X_h === f("Xh")[i, j, k] && r.X_l === f("Xl")[i, j, k] &&
                  r.Ā === f("Abar")[i, j, k] && r.Z̄ === f("Zbar")[i, j, k]
        end
        @test ok
    end

    @testset "multilinear fields are exact" begin
        rng = StableRNG(1201)
        worst = 0.0
        for _ in 1:2000
            a, b, c = rand(rng), rand(rng), rand(rng)
            x = tbl.logρ[1] + a * (tbl.logρ[end] - tbl.logρ[1])
            u = tbl.logT[1] + b * (tbl.logT[end] - tbl.logT[1])
            y = tbl.Yₑ[1] + c * (tbl.Yₑ[end] - tbl.Yₑ[1])
            r = composition(cv, exp10(x + lκ), exp10(u), y)
            μe = 5 + 20a + 3b + 40c + 2a * b * c
            μh = 2 + 30a - b + 10c
            Xn, Xp, Xa, Xl = 0.1 + 0.2a, 0.1 + 0.1b, 0.1 + 0.1c, 0.05a * b
            for (got, want) in ((r.μ_e, μe), (r.μ̂, μh), (r.μ_ν, μe - μh), (r.X_n, Xn), (r.X_p, Xp), (r.X_α, Xa),
                                (r.X_l, Xl), (r.X_h, 1 - Xn - Xp - Xa - Xl))
                worst = max(worst, abs(got - want) / max(abs(want), 1.0))
            end
        end
        # log10 and exp10 round-trip the coordinates to a few ulps of x.
        @test worst < 1e-12
        @info "composition: worst relative error on the multilinear synthetic fields" worst
        # The host table evaluates as its view.
        @test composition(ct, 1e10, 1.0, 0.3) === composition(cv, 1e10, 1.0, 0.3)
    end

    @testset "clamped to the table's box" begin
        ρlo, ρhi = exp10(tbl.logρ[1] + lκ), exp10(tbl.logρ[end] + lκ)
        Tlo, Thi = exp10(tbl.logT[1]), exp10(tbl.logT[end])
        same(a, b) = all(getfield(a, f) ≈ getfield(b, f) for f in fieldnames(Composition))
        @test same(composition(cv, ρlo / 100, 1.0, 0.3), composition(cv, ρlo, 1.0, 0.3))
        @test same(composition(cv, ρhi * 100, 1.0, 0.3), composition(cv, ρhi, 1.0, 0.3))
        @test same(composition(cv, 1e10, Tlo / 100, 0.3), composition(cv, 1e10, Tlo, 0.3))
        @test same(composition(cv, 1e10, Thi * 100, 0.3), composition(cv, 1e10, Thi, 0.3))
        @test same(composition(cv, 1e10, 1.0, -1.0), composition(cv, 1e10, 1.0, tbl.Yₑ[1]))
        @test same(composition(cv, 1e10, 1.0, 2.0), composition(cv, 1e10, 1.0, tbl.Yₑ[end]))
        # A non-finite or non-positive input reads inside the array and gives
        # NaN or a clamped value, never an exception.
        @test isnan(composition(cv, NaN, 1.0, 0.3).μ_e)
        @test isnan(composition(cv, 1e10, 1.0, NaN).μ_e)
        @test isfinite(composition(cv, 1e10, 0.0, 0.3).μ_e)
        @test isnan(composition(cv, -1.0, 1.0, 0.3).μ_e)
    end

    @testset "munu_offset and the build's checks" begin
        Q = E.Q_NP_MEV
        mk() = make_synthetic_table(SyntheticOptions(; with_composition=true))
        # The synthetic composition is no ideal gas, so the convention comes
        # from munu. A munu that subtracts Q says muhat lacks it: μ̂ then
        # carries Q, and μ_ν is the table's munu.
        t = mk()
        add_field!(t, "munu", field(t, "mu_e") .- field(t, "muhat") .- Q)
        c = CompositionTable(t, E.build_eos(t))
        @test c.munu_offset ≈ -Q
        r = E.composition_index(c.view, 3.0, 4.0, 5.0)
        @test r.μ̂ ≈ field(t, "muhat")[4, 5, 6] + Q
        @test r.μ_ν ≈ field(t, "munu")[4, 5, 6]
        # Neither 0 nor −Q: refused, unless given.
        add_field!(t, "munu", field(t, "mu_e") .- field(t, "muhat") .+ 0.5)
        @test_throws ArgumentError CompositionTable(t, E.build_eos(t))
        @test CompositionTable(t, E.build_eos(t); munu_offset=0.5).munu_offset == 0.5
        # An ideal gas of free nucleons decides instead, and munu is not read:
        # here muhat includes Q although munu subtracts a further one, as on
        # O'Connor–Ott LS220 above 10⁸ g/cm³; then muhat lacks Q.
        mₙ = E.M_NEUTRON_G * E.C_LIGHT_CM_S^2 / E.MEV_TO_ERG
        for (shift, want) in ((0.0, 0.0), (-Q, -Q))
            t = mk()
            T = [exp10(t.logT[j]) for i in eachindex(t.logρ), j in eachindex(t.logT), k in eachindex(t.Yₑ)]
            μh = Q .+ T .* log.(field(t, "Xn") ./ field(t, "Xp") .* ((mₙ - Q) / mₙ)^1.5) .+ shift
            add_field!(t, "muhat", μh)
            add_field!(t, "munu", field(t, "mu_e") .- μh .- Q)
            c = CompositionTable(t, E.build_eos(t))
            @test c.munu_offset == want
            @test E.composition_index(c.view, 3.0, 4.0, 5.0).μ̂ === μh[4, 5, 6] - want
            add_field!(t, "muhat", μh .+ 0.5)
            @test_throws ArgumentError CompositionTable(t, E.build_eos(t))
        end
        # munu is optional with the offset given.
        t = mk()
        tm = RawTable(t.logρ, t.logT, t.Yₑ)
        for n in field_names(t)
            n == "munu" || add_field!(tm, n, field(t, n))
        end
        add_attribute!(tm, "energy_shift", energy_shift(t))
        @test_throws ArgumentError CompositionTable(tm, E.build_eos(tm))
        @test CompositionTable(tm, E.build_eos(tm); munu_offset=0.0).munu_offset == 0
        # A required field missing, or non-finite.
        tx = RawTable(t.logρ, t.logT, t.Yₑ)
        for n in field_names(t)
            n == "Abar" || add_field!(tx, n, field(t, n))
        end
        add_attribute!(tx, "energy_shift", energy_shift(t))
        @test_throws ArgumentError CompositionTable(tx, E.build_eos(tx))
        t = mk()
        bad = copy(field(t, "Xh"))
        bad[2, 2, 2] = NaN
        add_field!(t, "Xh", bad)
        @test_throws ArgumentError CompositionTable(t, E.build_eos(t))
        # Mass fractions out of [0, 1] are clamped at the nodes.
        t = mk()
        bad = copy(field(t, "Xn"))
        bad[2, 2, 2], bad[3, 3, 3] = -1e-15, 1.5
        add_field!(t, "Xn", bad)
        c = CompositionTable(t, E.build_eos(t))
        @test E.composition_index(c.view, 1.0, 1.0, 1.0).X_n === 0.0
        @test E.composition_index(c.view, 2.0, 2.0, 2.0).X_n === 1.0
    end

    @testset "table identity" begin
        # Another table, of the same grid, with a different energy.
        other = make_synthetic_table(SyntheticOptions(; with_composition=true, energy_shift=2.0e18))
        eos_o = E.build_eos(other)
        @test fingerprint(eos_o) != fingerprint(eos)
        @test !matches(ct, eos_o) && !matches(cv, EOSTableView(eos_o))
        @test matches(ct, eos) && matches(cv, eos) && matches(cv, v)
        @test_throws ArgumentError CompositionTable(other, eos)
        # Not the composition datasets: they are not part of the identity.
        t2 = make_synthetic_table(SyntheticOptions(; with_composition=true))
        add_field!(t2, "Xn", field(t2, "Xn") .* 0.5)
        @test fingerprint(E.build_eos(t2)) == fingerprint(eos)
        # Narrowing and moving keep it.
        v32, cv32 = narrow(v, Float32), narrow(cv, Float32)
        @test fingerprint(v32) == fingerprint(v) && fingerprint(cv32) == fingerprint(cv)
        @test matches(cv32, v) && matches(cv, v32)
        @test matches(Adapt.adapt(Array, cv), Adapt.adapt(Array, v))
        # A trimmed build is still the same file.
        et = E.build_eos(tbl, BuildOptions(; trim_t_hi_rows=1))
        @test fingerprint(et) == fingerprint(eos)
        @test matches(CompositionTable(tbl, et), et)
        # The analytic EOSs have no composition.
        ig = IdealGasEOS(; Γ=2.0, K_ref=100.0, s_ref=5.0, s_window=(1.0, 20.0), ρ_bounds=(1e-10, 1e-2),
                         yₑ_bounds=(0.0, 1.0))
        @test !matches(cv, ig)
        @test !hasmethod(composition, Tuple{typeof(ig),Float64,Float64,Float64})
        @test hasmethod(composition, Tuple{typeof(cv),Float64,Float64,Float64})
    end

    @testset "the edge of the heavy nuclei" begin
        # No heavy nuclei at low density, a band of subnormal-in-Float32 X_h
        # next to it, and absurd fill values that must never show through.
        t = make_synthetic_table(SyntheticOptions(; with_composition=true))
        Xh, A, Z = copy(field(t, "Xh")), copy(field(t, "Abar")), copy(field(t, "Zbar"))
        Xh[1:15, :, :] .= 0
        A[1:15, :, :] .= 1e6
        Z[1:15, :, :] .= -7
        Xh[16, :, 1:2:end] .= 1e-42
        add_field!(t, "Xh", Xh)
        add_field!(t, "Abar", A)
        add_field!(t, "Zbar", Z)
        c = CompositionView(CompositionTable(t, E.build_eos(t)))
        r = E.composition_index(c, 3.5, 4.5, 5.5)
        @test r.X_h === 0.0 && r.Ā === 1.0 && r.Z̄ === 0.0
        _test_comp_edge(c, StableRNG(1202))
        @test maximum(E.composition_index(c, 14.0 + rand(), 10.0, 5.0).Ā for _ in 1:100) < 1e6
    end

    @testset "precision and allocation" begin
        cv32 = narrow(cv, Float32)
        @test cv32 isa CompositionView{Float32}
        r64 = composition(cv, 1e10, 1.0, 0.3)
        r32 = composition(cv32, 1.0f10, 1.0f0, 0.3f0)
        @test r32 isa Composition{Float32}
        @test all(isapprox(Float64(getfield(r32, f)), getfield(r64, f); rtol=1e-5, atol=1e-6)
                  for f in fieldnames(Composition))
        # A Float32 table with Float64 arithmetic.
        @test composition(cv32, 1e10, 1.0, 0.3) isa Composition{Float64}
        @test_noallocs _comp_allocs(cv, 1e10, 1.0, 0.3)
        @test_noallocs _comp_allocs(cv32, 1.0f10, 1.0f0, 0.3f0)
        @test_noallocs _comp_allocs(cv32, 1e10, 1.0, 0.3)
    end
end

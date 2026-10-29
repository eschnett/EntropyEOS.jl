# The EOS interface and the analytic EOSs.

# Allocation probes at top level; see test/testutil.jl.
_an_eval(eos, ρ, s, y, u) = EntropyEOS.evaluate(eos, ρ, s, y, u).U
_an_c2p(eos, cin, opts) = EntropyEOS.con2prim(eos, cin, opts).ρ
_an_safe(eos, cin, opts, proj) = EntropyEOS.con2prim_safe(eos, cin, opts, proj).base.ρ
function _an_eval_allocs(eos, ρ, s, y, u)
    _an_eval(eos, ρ, s, y, u)
    return @allocated _an_eval(eos, ρ, s, y, u)
end
function _an_c2p_allocs(eos, cin, opts)
    _an_c2p(eos, cin, opts)
    return @allocated _an_c2p(eos, cin, opts)
end
function _an_safe_allocs(eos, cin, opts, proj)
    _an_safe(eos, cin, opts, proj)
    return @allocated _an_safe(eos, cin, opts, proj)
end

"""
A test-only EOS implementing the interface contract and nothing else: a
Γ = 5/3 gas written out independently of `IdealGasEOS`. That the solver and the
policy run on it end to end is the proof that the contract is sufficient.
"""
struct ToyEOS <: EntropyEOS.AbstractEOS{Float64} end
function EntropyEOS.evaluate(::ToyEOS, ρ::T, s::T, yₑ::T, u_guess::T) where {T}
    U = T(3) / 2 * exp(T(2) / 3 * s) * cbrt(ρ)^2 * T(1e-3)
    p = T(2) / 3 * ρ * U
    h = 1 + U + p / ρ
    U_ρ = p / ρ^2
    U_ρρ = -T(2) / 9 * U / ρ^2
    return EOSPoint{T}(U, U_ρ, T(2) / 3 * U, U_ρρ, T(4) / 9 * U / ρ, T(2) / 3 * U, p, h,
                       T(5) / 3 * p / (ρ * h), T(NaN), zero(T), T(NaN), Int32(0), UInt32(0))
end
EntropyEOS.srange(::ToyEOS, ρ::T, yₑ::T) where {T} = SRange{T}(T(2), T(12))
EntropyEOS.srange_extended(::ToyEOS, ρ::T, yₑ::T) where {T} = SRange{T}(T(1), T(14))
EntropyEOS.logρ_bounds(::ToyEOS) = (-8.0, -2.0)
EntropyEOS.yₑ_bounds(::ToyEOS) = (0.0, 1.0)
EntropyEOS.κ(::ToyEOS) = 1.0

"""
A test-only wrapper that records the highest density at which the solver
evaluates the EOS inside it, and otherwise passes everything through.
"""
struct SpyEOS{E<:EntropyEOS.AbstractEOS{Float64}} <: EntropyEOS.AbstractEOS{Float64}
    eos::E
    ρ_seen::Base.RefValue{Float64}
end
SpyEOS(eos) = SpyEOS(eos, Ref(0.0))
function EntropyEOS.evaluate(e::SpyEOS, ρ::T, s::T, yₑ::T, u_guess::T) where {T}
    e.ρ_seen[] = max(e.ρ_seen[], Float64(ρ))
    return evaluate(e.eos, ρ, s, yₑ, u_guess)
end
EntropyEOS.srange(e::SpyEOS, ρ::T, yₑ::T) where {T} = srange(e.eos, ρ, yₑ)
EntropyEOS.srange_extended(e::SpyEOS, ρ::T, yₑ::T) where {T} = srange_extended(e.eos, ρ, yₑ)
EntropyEOS.logρ_bounds(e::SpyEOS) = logρ_bounds(e.eos)
EntropyEOS.yₑ_bounds(e::SpyEOS) = yₑ_bounds(e.eos)
EntropyEOS.κ(e::SpyEOS) = EntropyEOS.κ(e.eos)

# The SLy crust of O'Boyle et al. (2020), Table II. The table prints 1.826e6 for
# the second break, which is inconsistent with its own K, Λ and a columns:
# 1.826e8 reproduces all three, and 1.826e6 makes the pressure negative.
const SLY_CRUST_ρ = (6.285e5, 1.826e8, 3.350e11, 5.317e11)
const SLY_CRUST_Γ = (1.611, 1.440, 1.269, -1.841, 1.382)
const SLY_CRUST_K = (5.214e-9, 5.726e-8, 1.662e-6, -7.957e29, 1.746e-8)
const SLY_CRUST_Λ = (0.0, -1.354, -6.025e3, 1.193e9, 7.077e8)
const SLY_CRUST_a = (0.0, -1.861e-5, -5.278e-4, 1.035e-2, 8.208e-3)

# Their eq. B1: where a core (K₁, Γ₁) meets the crust's last piece with a
# continuous dp/dρ.
gpp_match_density(K_crust, Γ_crust, K₁, Γ₁) = (K₁ * Γ₁ / (K_crust * Γ_crust))^(1 / (Γ_crust - Γ₁))

# SLY4 on that crust (Table III), in cgs with c = 1, plus a thermal part that
# ranges from negligible at s_min to comparable with the cold energy at s_max.
function sly_hybrid(::Type{T}=Float64) where {T}
    K_crust = EntropyEOS.gpp_constants(SLY_CRUST_ρ, SLY_CRUST_K[1], SLY_CRUST_Γ).K[end]
    ρ₀ = gpp_match_density(K_crust, SLY_CRUST_Γ[end], 10^-31.350, 3.045)
    return HybridEOS{T}(; ρ_breaks=(SLY_CRUST_ρ..., ρ₀, 10^14.87, 10^14.99), K₀=SLY_CRUST_K[1],
                        Γs=(SLY_CRUST_Γ..., 3.045, 2.884, 2.773), Γ_th=1.75, K_th_ref=2.4e-12, s_ref=20.0,
                        s_window=(1.0, 20.0), ρ_bounds=(1e6, 2e15), yₑ_bounds=(0.0, 1.0))
end

ideal_gas(Γ, ::Type{T}=Float64; s_window=(1.0, 20.0)) where {T} =
    IdealGasEOS{T}(; Γ, K_ref=100.0, s_ref=5.0, s_window, ρ_bounds=(1e-10, 1e-2), yₑ_bounds=(0.0, 1.0))

# An independent closed form of the hybrid's potential, for automatic
# differentiation to check `evaluate`'s hand-written derivatives against.
function hybrid_U(eos::HybridEOS, ρ, s)
    g = EntropyEOS.gpp_constants(eos)
    i = findlast(<=(ρ), g.ρ_lo)
    U_cold = g.K[i] * ρ^(g.Γ[i] - 1) / (g.Γ[i] - 1) + g.a[i] - g.Λ[i] / ρ
    Γ_th = eos.Γ_th
    K_th = exp(eos.log_K_th_ref + (Γ_th - 1) * (s - eos.s_ref))
    return U_cold + K_th * ρ^(Γ_th - 1) / (Γ_th - 1)
end
ideal_U(eos::IdealGasEOS, ρ, s) = exp(eos.log_K_ref + (eos.Γ - 1) * (s - eos.s_ref)) * ρ^(eos.Γ - 1) / (eos.Γ - 1)

function check_derivatives(eos, Uref, ρ, s)
    pt = evaluate(eos, ρ, s, 0.5, NaN)
    U_ρ(r, σ) = ForwardDiff.derivative(r′ -> Uref(eos, r′, σ), r)
    ok = isapprox(pt.U, Uref(eos, ρ, s); rtol=1e-13) &&
         isapprox(pt.U_ρ, U_ρ(ρ, s); rtol=1e-12) &&
         isapprox(pt.U_s, ForwardDiff.derivative(σ -> Uref(eos, ρ, σ), s); rtol=1e-12) &&
         isapprox(pt.U_ρρ, ForwardDiff.derivative(r -> U_ρ(r, s), ρ); rtol=1e-9, atol=1e-12 * abs(pt.U_ρ) / ρ) &&
         isapprox(pt.U_ρs, ForwardDiff.derivative(σ -> U_ρ(ρ, σ), s); rtol=1e-12)
    # evaluate itself must be differentiable, which is what makes it generic in
    # the number type rather than merely Float64-and-Float32.
    ok &= isapprox(ForwardDiff.derivative(r -> evaluate(eos, r, s, 0.5, NaN).U, ρ), pt.U_ρ; rtol=1e-12)
    ok &= isapprox(ForwardDiff.derivative(σ -> evaluate(eos, ρ, σ, 0.5, NaN).U, s), pt.U_s; rtol=1e-12)
    # The identities every EOSPoint owes the solver.
    ok &= pt.p == ρ * ρ * pt.U_ρ && pt.h == 1 + pt.U + pt.p / ρ && pt.T̂ == pt.U_s &&
          pt.cs² == (2ρ * pt.U_ρ + ρ * ρ * pt.U_ρρ) / pt.h
    return ok
end

"""
Round trip `con2prim_safe ∘ prim2con` over the EOS's box, and return the worst
errors, each scaled by the condition number of the quantity it measures.

The raw errors cannot all be 1e-12: when `U ≪ 1` the thermal energy is a tiny
fraction of τ, and at low entropy on the hybrid `U_s ≪ U` as well, so the
entropy is physically ill-determined by the conservatives; and when `hW² ≫ 1`
the rest mass is a small fraction of the energy, which does the same to ρ.
Scaling by those factors turns "accurate where the problem allows" into one
number per quantity.
"""
function round_trip(eos, n; T=Float64, cold=false, seed=1, w_max=acosh(10.0))
    rng = StableRNG(seed)
    opts = Con2PrimOptions{T}()
    proj = default_projection(eos, T(exp10(first(logρ_bounds(eos)))))
    x_lo, x_hi = logρ_bounds(eos)
    nfail = ntouched = 0
    eρ, es = Float64[], Float64[]
    for _ in 1:n
        # 5%-per-side margins on the box, and a warm start off by 1e-3, as the
        # C++ audit samples: corners and exact warm starts are both unrepresentative.
        ρ = T(exp10(x_lo + (x_hi - x_lo) * (0.05 + 0.9rand(rng))))
        sr = srange(eos, ρ, T(0.5))
        s = T(sr.s_min + (sr.s_max - sr.s_min) * (0.05 + 0.9rand(rng)))
        w = T(w_max * rand(rng))
        B² = T(rand(rng) < 0.5 ? ρ * 10.0^(-2 + 4rand(rng)) : 0.0)
        pt = evaluate(eos, ρ, s, T(0.5), T(NaN))
        c = prim2con(eos, ρ, s, T(0.5), w, B², T(2rand(rng) - 1), pt.u_solved)
        cin = Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²)
        out = cold ? con2prim_safe(eos, cin, opts, proj) :
              con2prim_safe(eos, cin, opts, proj, s * T(1 + 1e-3), w * T(1 + 1e-3), T(NaN))
        r = out.base.result
        (r === EntropyEOS.C2PResult.converged_newton || r === EntropyEOS.C2PResult.converged_fallback) ||
            (nfail += 1)
        out.policy_flags != 0 && (ntouched += 1)
        W² = cosh(Float64(w))^2
        κ_ρ = pt.h * W²
        κ_s = (1 + c.τ / (ρ * W² * pt.U)) * max(1, pt.U / (s * pt.U_s))
        push!(eρ, max(abs(out.base.ρ - ρ) / ρ, abs(out.base.w - w) / max(w, 1)) / κ_ρ)
        push!(es, abs(out.base.s - s) / s / κ_s)
    end
    p99(v) = sort(v)[round(Int, 0.99 * length(v))]
    return (; nfail, ntouched, eρ=maximum(eρ), es=maximum(es), eρ99=p99(eρ), es99=p99(es))
end

@testset "analytic EOSs" begin
    E = EntropyEOS

    @testset "interface" begin
        v = EOSTableView(E.build_eos(make_synthetic_table()))
        @test v isa AbstractEOS{Float64}
        @test logρ_bounds(v) === (v.x_lo, v.x_hi)
        @test yₑ_bounds(v) === (v.y_lo, v.y_hi)
        @test E.κ(v) === v.κ
        @test eltype(narrow(v, Float32)) === Float32
    end

    @testset "a toy EOS implementing only the contract runs the whole solver" begin
        toy = ToyEOS()
        opts = Con2PrimOptions()
        proj = default_projection(toy, 1e-8)
        @test proj.ρ_max ≈ 1e-2
        r = round_trip(toy, 300; cold=true)
        @test r.nfail == 0 && r.ntouched == 0
        @test r.eρ < 1e-11 && r.es < 1e-10
        out = con2prim_safe(toy, Con2PrimIn(1e-5, 1e-30, 0.0, 0.0, 0.0, 0.0), opts, proj)
        @test out.policy_flags == FLAG_POL_S_FLOORED
        @test isempty(policy_violations(toy, proj, opts, Con2PrimIn(1e-5, 1e-30, 0.0, 0.0, 0.0, 0.0), out))
    end

    @testset "ideal gas" begin
        for Γ in (4 / 3, 5 / 3, 2.0)
            eos = ideal_gas(Γ)
            @test isbits(eos)
            @test E.κ(eos) === 1.0
            for ρ in (1e-9, 1e-5, 1e-2), s in (1.0, 7.0, 20.0)
                @test check_derivatives(eos, ideal_U, ρ, s)
                pt = evaluate(eos, ρ, s, 0.5, NaN)
                @test pt.T̂ ≈ (Γ - 1) * pt.U rtol = 1e-14
                @test pt.T̂ ≈ pt.p / ρ rtol = 1e-14
                @test pt.h ≈ 1 + Γ * pt.U rtol = 1e-14
                @test pt.cs² ≈ Γ * pt.p / (ρ * pt.h) rtol = 1e-14
                @test pt.μ̃ == 0 && isnan(pt.T_MeV) && isnan(pt.u_solved) && pt.flags == 0
            end
        end
        # TOV's polytrope, p = 100ρ², is the Γ = 2 gas at one entropy.
        eos = ideal_gas(2.0)
        s = polytropic_entropy(eos, 100.0)
        for ρ in (1e-8, 1.28e-3)
            @test evaluate(eos, ρ, s, 0.5, NaN).p ≈ 100ρ^2 rtol = 1e-14
        end
        @test polytropic_entropy(ideal_gas(5 / 3), 100.0 * exp(2 / 3 * 3)) ≈ 8.0
        # Flags say where the point lies; nothing is clamped.
        @test evaluate(eos, 1e-11, 5.0, 0.5, NaN).flags == FLAG_EXT_ρ_LOW
        @test evaluate(eos, 1.0, 5.0, 0.5, NaN).flags == FLAG_OOB_ρ_HIGH
        @test evaluate(eos, 1e-5, 0.5, 0.5, NaN).flags == FLAG_EXT_S_LOW
        @test evaluate(eos, 1e-5, 25.0, 0.5, NaN).flags == FLAG_EXT_S_HIGH
        @test evaluate(eos, 1.0, 5.0, 0.5, NaN).p ≈ 100.0 rtol = 1e-14
        @test all(isnan, (evaluate(eos, -1.0, 5.0, 0.5, NaN).U, evaluate(eos, NaN, 5.0, 0.5, NaN).U))
    end

    @testset "generalized piecewise polytrope" begin
        @testset "reproduces O'Boyle et al. (2020)" begin
            # Their Table II from its first row alone. The tolerance is the
            # table's own rounding: Γ is printed to three decimals, and K
            # depends on Γ through ρ^Γ with ln ρ up to 27.
            g = E.gpp_constants(SLY_CRUST_ρ, SLY_CRUST_K[1], SLY_CRUST_Γ)
            @test all(isapprox.(g.K, SLY_CRUST_K; rtol=0.015))
            @test all(isapprox.(g.Λ, SLY_CRUST_Λ; rtol=0.01))
            @test all(isapprox.(g.a, SLY_CRUST_a; rtol=0.01))
            # Their Table III densities follow from Table II's crust by eq. B1.
            for (lρ₀, lK₁, Γ₁) in ((13.980, -31.350, 3.045), (14.040, -33.210, 3.169), (14.088, -40.301, 3.662))
                @test log10(gpp_match_density(SLY_CRUST_K[end], SLY_CRUST_Γ[end], 10^lK₁, Γ₁)) ≈ lρ₀ atol = 2e-3
            end
            # And the hybrid built on it has the tabulated core K₁.
            @test log10(E.gpp_constants(sly_hybrid()).K[6]) ≈ -31.350 rtol = 1e-12
        end

        eos = sly_hybrid()
        @test isbits(eos)
        cold(ρ) = evaluate(eos, ρ, -1e3, 0.5, NaN)   # U_th underflows to zero

        @testset "p, ε and cs² are continuous at every break" begin
            for ρ_b in E.gpp_constants(eos).ρ_lo[2:end]
                lo, hi = cold(prevfloat(ρ_b)), cold(ρ_b)
                @test lo.U ≈ hi.U rtol = 1e-13
                @test lo.p ≈ hi.p rtol = 1e-13
                # The point of the generalized form: a classic piecewise
                # polytrope jumps here by Γᵢ₊₁/Γᵢ, often tens of percent.
                @test lo.cs² ≈ hi.cs² rtol = 1e-13
            end
        end

        @testset "the cold part is physical" begin
            ρs = exp10.(range(-10, log10(2e15); length=400))
            pts = cold.(ρs)
            @test all(pt -> pt.p > 0 && pt.U > 0 && pt.cs² > 0, pts)
            @test issorted(getfield.(pts, :U)) && issorted(getfield.(pts, :p))
            @test cold(1e-30).U < 1e-25          # ε_cold(0) = 0
            @test maximum(getfield.(pts, :cs²)) < 1
        end

        @testset "derivatives" begin
            ρs = (1e6, 3e9, 1e13, 2e14, 1.5e15)
            breaks = E.gpp_constants(eos).ρ_lo[2:end]
            for ρ in (ρs..., prevfloat.(breaks)..., breaks...), s in (1.0, 10.0, 20.0)
                @test check_derivatives(eos, hybrid_U, ρ, s)
            end
        end

        @testset "a single piece is a polytrope plus a thermal part" begin
            one_piece = HybridEOS(; ρ_breaks=(), K₀=100.0, Γs=(2.0,), Γ_th=5 / 3, K_th_ref=1.0, s_ref=0.0,
                                  s_window=(1.0, 10.0), ρ_bounds=(1e-10, 1e-3), yₑ_bounds=(0.0, 1.0))
            @test evaluate(one_piece, 1e-3, -1e3, 0.5, NaN).p ≈ 100 * 1e-6 rtol = 1e-14
            @test check_derivatives(one_piece, hybrid_U, 1e-4, 3.0)
        end
    end

    @testset "con2prim_safe ∘ prim2con is the identity" begin
        # Thresholds are about 10× the measured worst case at 1000 samples.
        for eos in (ideal_gas(4 / 3), ideal_gas(5 / 3), ideal_gas(2.0), sly_hybrid()), cold in (false, true)
            r = round_trip(eos, 1000; cold)
            @test r.nfail == 0
            @test r.ntouched == 0
            @test r.eρ < 3e-11
            @test r.es < 1e-10
        end
    end

    @testset "con2prim stays inside the density box" begin
        # The hybrid is causal on its box, but its last polytrope continues
        # above it and turns acausal: cs² is 0.97 at the box's top, 2·10¹⁵, and
        # 1.26 at 3·10¹⁵. A solve whose rapidity bracket reached down to rest
        # swept densities up to D = ρW through that, and found extra roots of
        # the inner solve there: 99 of these 300 states failed cold.
        spy = SpyEOS(sly_hybrid())
        x_lo, x_hi = logρ_bounds(spy)
        ρ_top = exp10(x_hi) * (1 + E.face_slack(Float64))
        opts = Con2PrimOptions()
        conv(out) = out.result === E.C2PResult.converged_newton || out.result === E.C2PResult.converged_fallback
        function cons(rng, ρ, w)
            sr = srange(spy.eos, ρ, 0.5)
            s = sr.s_min + (sr.s_max - sr.s_min) * (0.02 + 0.96rand(rng))
            pt = evaluate(spy.eos, ρ, s, 0.5, NaN)
            B² = rand(rng) < 0.5 ? pt.p * exp10(-3 + 4rand(rng)) : 0.0
            c = prim2con(spy.eos, ρ, s, 0.5, w, B², 2rand(rng) - 1, NaN)
            return Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²), pt
        end
        rng = StableRNG(3)
        nfail = 0
        eρ = 0.0
        for _ in 1:300
            ρ = exp10(x_hi - 0.3rand(rng))
            w = 2.5 + 3.5rand(rng)
            cin, pt = cons(rng, ρ, w)
            out = con2prim(spy, cin, opts)
            conv(out) || (nfail += 1)
            # Per condition number, as in round_trip, and the momentum
            # equation's own: dφ/dw at the root falls as 1 − cs²·v², and cs²
            # reaches 0.97 here, at W up to 200.
            eρ = max(eρ, abs(out.ρ - ρ) / ρ * (1 - pt.cs² * tanh(w)^2) / (pt.h * cosh(w)^2))
        end
        @test nfail == 0
        @test eρ < 2e-11
        @test spy.ρ_seen[] <= ρ_top * (1 + 1e-12)

        # Above the box there is nothing to find. A state there comes back
        # unsolved, or as a state inside the box with the same conservatives,
        # which the acausal extension makes possible; the projection's contract
        # holds either way.
        proj = default_projection(spy.eos, exp10(x_lo + 1))
        spy.ρ_seen[] = 0.0
        nconv = 0
        for _ in 1:100
            cin, _ = cons(rng, exp10(x_hi + rand(rng)), 4rand(rng))
            out = con2prim(spy, cin, opts)
            conv(out) && (nconv += 1)
            @test out.ρ <= ρ_top * (1 + 1e-12)
            safe = con2prim_safe(spy.eos, cin, opts, proj)
            @test isempty(policy_violations(spy.eos, proj, opts, cin, safe))
        end
        @test nconv < 50
        @test spy.ρ_seen[] <= ρ_top * (1 + 1e-12)
    end

    @testset "Float32 smoke test" begin
        # Pure Float32 loses the rest mass once hW² approaches 1/eps, so the
        # ideal gas is sampled where U stays moderate. The hybrid's low-entropy
        # corner is ill-conditioned in s (see above), so there the policy may
        # floor the entropy; ρ must still come back.
        for (eos, touched_ok) in ((ideal_gas(5 / 3, Float32; s_window=(1.0, 8.0)), false), (sly_hybrid(Float32), true))
            for cold in (false, true)
                r = round_trip(eos, 1000; T=Float32, cold)
                @test r.nfail == 0
                touched_ok || @test r.ntouched == 0
                @test r.eρ99 < 1e-3
            end
        end
    end

    @testset "projection" begin
        for eos in (ideal_gas(2.0), sly_hybrid())
            x_lo, x_hi = logρ_bounds(eos)
            ρ_min = exp10(x_lo + 1)
            proj = default_projection(eos, ρ_min)
            opts = Con2PrimOptions()
            @test proj.ρ_max ≈ exp10(x_hi)

            ρ = exp10(0.5 * (x_lo + x_hi))
            sr = srange(eos, ρ, 0.5)
            g = let c = prim2con(eos, ρ, 0.5 * (sr.s_min + sr.s_max), 0.5, 0.8, 0.0, 0.0, NaN)
                Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²)
            end
            for (cin, bit) in ((Con2PrimIn(g.D, NaN, g.D_Y, g.S_par, g.S_perp, g.B²), FLAG_POL_NONFINITE),
                               (Con2PrimIn(ρ_min * 1e-3, g.τ * 1e-20, 0.0, 0.0, 0.0, 0.0), FLAG_POL_ρ_FLOORED),
                               (Con2PrimIn(g.D, g.τ * 1e-10, g.D_Y, g.S_par, g.S_perp, g.B²), FLAG_POL_S_FLOORED),
                               (Con2PrimIn(g.D, g.τ * 1e6, g.D_Y, g.S_par, g.S_perp, g.B²), FLAG_POL_S_CEILED),
                               (Con2PrimIn(g.D, g.τ, g.D_Y, 1e30, 0.0, g.B²), FLAG_POL_W_CEILED),
                               (Con2PrimIn(1e30, g.τ, 0.5e30, g.S_par, g.S_perp, g.B²), FLAG_POL_ρ_CEILED))
                out = con2prim_safe(eos, cin, opts, proj)
                @test out.policy_flags & bit == bit
                @test isempty(policy_violations(eos, proj, opts, cin, out))
            end

            # Projection is bitwise idempotent, over and outside the box.
            rng = StableRNG(7)
            for _ in 1:300
                ps = PrimState(exp10(x_lo - 1 + (x_hi - x_lo + 2) * rand(rng)), 25rand(rng) - 2,
                               1.2rand(rng) - 0.1, 5rand(rng))
                ps2, flags = project_prim_state(eos, ps, proj)
                @test flags == check_prim_state(eos, ps, proj)
                @test inside_window(eos, ps2.ρ, ps2.s, ps2.ye)
                ps3, flags3 = project_prim_state(eos, ps2, proj)
                @test ps3 === ps2 && flags3 == 0
            end
        end
    end

    @testset "device readiness" begin
        for eos in (ideal_gas(5 / 3), sly_hybrid())
            opts = Con2PrimOptions()
            proj = default_projection(eos, exp10(first(logρ_bounds(eos)) + 1))
            ρ = exp10(sum(logρ_bounds(eos)) / 2)
            c = prim2con(eos, ρ, 10.0, 0.5, 0.8, 0.1ρ, 0.3, NaN)
            cin = Con2PrimIn(c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²)

            @test isbits(eos)
            # Adapt needs no rule for an isbits EOS: it is its own device copy.
            @test Adapt.adapt(Array, eos) === eos
            @inferred evaluate(eos, ρ, 10.0, 0.5, NaN)
            @inferred con2prim(eos, cin, opts)
            out = @inferred con2prim_safe(eos, cin, opts, proj)
            @inferred apply_policy(eos, cin, out, proj, PolicyOptions())
            @test_noallocs _an_eval_allocs(eos, ρ, 10.0, 0.5, NaN)
            @test_noallocs _an_c2p_allocs(eos, cin, opts)
            @test_noallocs _an_safe_allocs(eos, cin, opts, proj)

            e32 = narrow(eos, Float32)
            @test eltype(e32) === Float32 && isbits(e32)
            pt32 = @inferred evaluate(e32, Float32(ρ), 10.0f0, 0.5f0, NaN32)
            @test pt32 isa EOSPoint{Float32}
            @test pt32.p ≈ evaluate(eos, ρ, 10.0, 0.5, NaN).p rtol = 1e-5
            # A Float32 EOS evaluated with Float64 arguments computes in Float64.
            @test evaluate(e32, ρ, 10.0, 0.5, NaN) isa EOSPoint{Float64}
        end

        # A downstream code's state that the projection puts on the ρ_min face:
        # the Newton and the nested fallback both run, and the scan finds no
        # bracket. Under `--check-bounds=yes` the scan's scratch was once
        # heap-allocated here.
        for T in (Float64, Float32)
            eos = ideal_gas(2.0, T)
            opts = Con2PrimOptions{T}()
            proj = default_projection(eos, T(1e-10))
            D = T(5e-10)
            cin = Con2PrimIn(D, T(1e-9 - 5e-13), D / 2, zero(T), T(1e-8), zero(T))
            @test !is_done(con2prim_safe_start(eos, cin, opts, proj)[2])
            out = con2prim_safe(eos, cin, opts, proj)
            @test out.base.result === EntropyEOS.C2PResult.failed_no_bracket
            @test out.policy_flags == FLAG_POL_ρ_FLOORED | FLAG_POL_S_FLOORED
            @test_noallocs _an_safe_allocs(eos, cin, opts, proj)
        end
    end

    @testset "constructors validate" begin
        good = (; Γ=2.0, K_ref=100.0, s_ref=5.0, s_window=(1.0, 20.0), ρ_bounds=(1e-10, 1e-2), yₑ_bounds=(0.0, 1.0))
        @test IdealGasEOS(; good...) isa IdealGasEOS{Float64}
        @test IdealGasEOS{Float32}(; good...) isa IdealGasEOS{Float32}
        for bad in ((; Γ=1.0), (; Γ=NaN), (; K_ref=-1.0), (; s_window=(0.0, 1.0)), (; s_window=(2.0, 1.0)),
                    (; ρ_bounds=(1e-2, 1e-10)), (; yₑ_bounds=(1.0, 0.0)), (; s_window=(1.0, Inf)))
            @test_throws ArgumentError IdealGasEOS(; good..., bad...)
        end
        # A stiff gas is acausal once hot enough: cs² → Γ−1 as U grows.
        @test_throws ArgumentError IdealGasEOS(; good..., Γ=3.0)
        @test IdealGasEOS(; good..., Γ=3.0, s_window=(1.0, 2.0)) isa IdealGasEOS

        hy = (; ρ_breaks=(1e-4,), K₀=100.0, Γs=(2.0, 2.5), Γ_th=5 / 3, K_th_ref=1.0, s_ref=0.0,
              s_window=(1.0, 5.0), ρ_bounds=(1e-10, 1e-3), yₑ_bounds=(0.0, 1.0))
        @test HybridEOS(; hy...) isa HybridEOS{Float64,2}
        @test HybridEOS{Float32,2}(; hy...) isa HybridEOS{Float32,2}
        for bad in ((; Γs=(1.0, 2.5)), (; Γs=(2.0, 1.0)), (; Γs=(2.0, 0.0)), (; K₀=0.0), (; Γ_th=1.0),
                    (; ρ_breaks=(-1.0,)), (; ρ_breaks=(1e-4, 1e-5), Γs=(2.0, 2.5, 2.2)))
            @test_throws ArgumentError HybridEOS(; hy..., bad...)
        end
        @test_throws ArgumentError HybridEOS{Float64,3}(; hy...)
        # A core that stiffens without bound is acausal at the top of the box.
        @test_throws ArgumentError HybridEOS(; hy..., Γs=(2.0, 4.0), ρ_bounds=(1e-10, 1.0))
    end
end

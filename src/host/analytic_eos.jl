# Building the analytic EOSs: validate the parameters, derive the constants of
# the generalized piecewise polytrope, and check causality once on the host so
# the kernels never have to.

function analytic_require(cond::Bool, msg::AbstractString)
    cond || throw(ArgumentError(msg))
    return nothing
end

function analytic_box(::Type{T}, s_window, ρ_bounds, yₑ_bounds) where {T}
    s_min, s_max = Float64.(s_window)
    ρ_lo, ρ_hi = Float64.(ρ_bounds)
    y_lo, y_hi = Float64.(yₑ_bounds)
    analytic_require(all(isfinite, (s_min, s_max, ρ_lo, ρ_hi, y_lo, y_hi)), "the bounds must be finite")
    analytic_require(0 < s_min < s_max, "s_window must satisfy 0 < s_min < s_max, got $s_window")
    analytic_require(0 < ρ_lo < ρ_hi, "ρ_bounds must satisfy 0 < ρ_min < ρ_max, got $ρ_bounds")
    analytic_require(y_lo <= y_hi, "yₑ_bounds must satisfy y_min ≤ y_max, got $yₑ_bounds")
    Δ = (s_max - s_min) / 10
    return AnalyticBox{T}(T(ρ_lo), T(ρ_hi), T(log10(ρ_lo)), T(log10(ρ_hi)), T(y_lo), T(y_hi),
                          T(s_min), T(s_max), T(max(s_min - Δ, s_min / 2)), T(s_max + Δ))
end

# The gas is checked on a log-spaced density grid at both ends of the entropy
# window, plus both sides of every break. For the ideal gas cs² rises
# monotonically with U, so the grid's corner (ρ_max, s_max) makes this exact.
function analytic_check_causal(eos::AnalyticEOS, extra_ρ=())
    b = eos.box
    ρs = vcat(exp10.(range(Float64(b.x_lo), Float64(b.x_hi); length=64)), Float64(b.ρ_lo), Float64(b.ρ_hi))
    for ρ_b in extra_ρ
        Float64(b.ρ_lo) <= ρ_b <= Float64(b.ρ_hi) && append!(ρs, (prevfloat(ρ_b), ρ_b))
    end
    for ρ in ρs, s in (Float64(b.s_min), Float64(b.s_max))
        cs² = evaluate(eos, ρ, s, Float64(b.y_lo), NaN).cs²
        analytic_require(cs² < 1, "the EOS is acausal on its box: cs² = $cs² at ρ = $ρ, s = $s")
    end
    return nothing
end

function IdealGasEOS{T}(; Γ, K_ref, s_ref, s_window, ρ_bounds, yₑ_bounds) where {T<:AbstractFloat}
    analytic_require(all(isfinite, (Γ, K_ref, s_ref)), "Γ, K_ref and s_ref must be finite")
    analytic_require(Γ > 1, "Γ must exceed 1, got $Γ")
    analytic_require(K_ref > 0, "K_ref must be positive, got $K_ref")
    box = analytic_box(T, s_window, ρ_bounds, yₑ_bounds)
    eos = IdealGasEOS{T}(T(Γ), T(log(Float64(K_ref))), T(s_ref), box)
    analytic_check_causal(eos)
    return eos
end

IdealGasEOS(; kwargs...) = IdealGasEOS{Float64}(; kwargs...)

"""
    polytropic_entropy(eos::IdealGasEOS, K)

The entropy at which `eos` is the polytrope `p = Kρ^Γ`, i.e. the `s` with
`K(s) = K`.
"""
function polytropic_entropy(eos::IdealGasEOS{T}, K::Real) where {T}
    analytic_require(K > 0, "K must be positive, got $K")
    return eos.s_ref + (log(T(K)) - eos.log_K_ref) / (eos.Γ - one(T))
end

"""
    gpp_constants(ρ_breaks, K₀, Γs) -> (; ρ_lo, K, Γ, Λ, a)

The per-piece constants of a generalized piecewise polytrope (O'Boyle et al.
2020, eqs. 4.6–4.8), starting from `Λ = a = 0` on the first piece so that
`ε(0) = 0`. Each result is a tuple with one entry per piece; `ρ_lo` is where
the piece starts.
"""
function gpp_constants(ρ_breaks, K₀, Γs)
    N = length(Γs)
    K, Λ, a = Float64(K₀), 0.0, 0.0
    Ks, Λs, as = [K], [Λ], [a]
    for i in 1:(N - 1)
        ρ_b, g, G = Float64(ρ_breaks[i]), Float64(Γs[i]), Float64(Γs[i + 1])
        P = K * ρ_b^g                       # the polytropic part of p at the break
        Λ += (1 - g / G) * P                # p continuous
        a += g * (G - g) / ((G - 1) * (g - 1)) * P / ρ_b    # ε continuous
        K *= (g / G) * ρ_b^(g - G)          # dp/dρ continuous
        push!(Ks, K)
        push!(Λs, Λ)
        push!(as, a)
    end
    ρ_lo = (0.0, Float64.(Tuple(ρ_breaks))...)
    return (; ρ_lo, K=Tuple(Ks), Γ=Float64.(Tuple(Γs)), Λ=Tuple(Λs), a=Tuple(as))
end

function HybridEOS{T,N}(; ρ_breaks, K₀, Γs, Γ_th, K_th_ref, s_ref, s_window, ρ_bounds,
                        yₑ_bounds) where {T<:AbstractFloat,N}
    analytic_require(length(Γs) == N, "Γs must have N = $N entries, got $(length(Γs))")
    analytic_require(length(ρ_breaks) == N - 1, "ρ_breaks must have N−1 = $(N - 1) entries, got $(length(ρ_breaks))")
    analytic_require(all(isfinite, (K₀, Γ_th, K_th_ref, s_ref)) && all(isfinite, Γs) && all(isfinite, ρ_breaks),
                     "every parameter must be finite")
    analytic_require(K₀ > 0, "K₀ must be positive, got $K₀")
    analytic_require(Γs[1] > 1, "the first exponent must exceed 1 so that ε_cold(0) = 0, got $(Γs[1])")
    analytic_require(all(g -> g != 0 && g != 1, Γs), "no exponent may be 0 or 1, got $Γs")
    analytic_require(Γ_th > 1, "Γ_th must exceed 1, got $Γ_th")
    analytic_require(K_th_ref > 0, "K_th_ref must be positive, got $K_th_ref")
    analytic_require(all(>(0), ρ_breaks) && issorted(ρ_breaks; lt=<=),
                     "ρ_breaks must be positive and strictly increasing, got $ρ_breaks")
    box = analytic_box(T, s_window, ρ_bounds, yₑ_bounds)

    gpp = gpp_constants(ρ_breaks, K₀, Γs)
    pieces = ntuple(Val(N)) do i
        # Scale each piece to a density where it is evaluated: its own lower
        # break, or for the first piece the first break (or the box's top).
        ρ̄ = i > 1 ? gpp.ρ_lo[i] : (N > 1 ? gpp.ρ_lo[2] : Float64(box.ρ_hi))
        c = gpp.K[i] * ρ̄^(gpp.Γ[i] - 1)
        return GPPPiece{T}(T(gpp.ρ_lo[i]), T(ρ̄), T(c), T(gpp.Γ[i]), T(gpp.a[i]), T(gpp.Λ[i]))
    end
    eos = HybridEOS{T,N}(pieces, T(Γ_th), T(log(Float64(K_th_ref))), T(s_ref), box)
    analytic_check_causal(eos, gpp.ρ_lo[2:end])
    return eos
end

HybridEOS{T}(; Γs, kwargs...) where {T<:AbstractFloat} = HybridEOS{T,length(Γs)}(; Γs, kwargs...)
HybridEOS(; kwargs...) = HybridEOS{Float64}(; kwargs...)

"""
    gpp_constants(eos::HybridEOS) -> (; ρ_lo, K, Γ, Λ, a)

The derived constants of `eos`'s cold part, per piece, in the storage type.
"""
function gpp_constants(eos::HybridEOS)
    ps = eos.pieces
    return (; ρ_lo=map(q -> q.ρ_lo, ps), K=map(q -> q.c / q.ρ̄^(q.Γ - 1), ps), Γ=map(q -> q.Γ, ps),
            Λ=map(q -> q.Λ, ps), a=map(q -> q.a, ps))
end

# An analytic, exactly consistent ideal gas, tabulated onto a RawTable.
#
# Translated from `entropy_eos/host/synthetic.{hpp,cpp}`.
#
# This is the ground truth the tests measure against, and it is shipped rather
# than kept in `test/` because it is the only way to obtain a table without a
# multi-hundred-megabyte file -- useful for a downstream hydro code's own tests,
# for benchmarks and for documentation examples.
#
# With g = 1 + Yₑ and kT in erg:
#
#   ε   = 1.5 · g · kT / m_B                        erg/g
#   p   = g · ρ · kT / m_B                          dyn/cm²
#   s   = g · (s₀ + 1.5·ln(T/T₀) - ln(ρ/ρ₀))        k_B per baryon
#   cs² = (5/3) · g · kT / (m_B · h · c²)
#
# It is smooth and well behaved by construction, so it exercises none of the
# pathologies real tables have. The defect injectors below manufacture those
# deliberately; they are applied in a fixed order, and there is no RNG anywhere,
# so a generated table is bit-for-bit reproducible.

"""Perturb one node of one field by an additive delta."""
Base.@kwdef struct SeededViolation
    field::String = "entropy"
    iρ::Int = 1
    jT::Int = 1
    kYₑ::Int = 1
    delta::Float64 = 0.0
end

"""Make a field constant along T over an index block."""
Base.@kwdef struct FlattenDefect
    field::String = "entropy"
    iρ0::Int = 1
    iρ1::Int = 1
    kYₑ0::Int = 1
    kYₑ1::Int = 1
    jT0::Int = 1
    jT1::Int = 1
end

"""Add an oscillation along T over an index block."""
Base.@kwdef struct WiggleDefect
    field::String = "entropy"
    iρ0::Int = 1
    iρ1::Int = 1
    kYₑ0::Int = 1
    kYₑ1::Int = 1
    jT0::Int = 1
    jT1::Int = 1
    amplitude::Float64 = 0.0
    period::Float64 = 1.0
end

"""Add a constant offset over an index block."""
Base.@kwdef struct OffsetDefect
    field::String = "entropy"
    iρ0::Int = 1
    iρ1::Int = 1
    kYₑ0::Int = 1
    kYₑ1::Int = 1
    jT0::Int = 1
    jT1::Int = 1
    offset::Float64 = 0.0
end

"""Stiffen the high-density corner, manufacturing an acausal region."""
Base.@kwdef struct StiffenDefect
    field::String = "logenergy"
    ρ_c_gcc::Float64 = 1.0
    α::Float64 = 0.0
    amplitude::Float64 = 0.0
end

"""Set one node of one field to an exact value."""
Base.@kwdef struct SetValue
    field::String = "entropy"
    iρ::Int = 1
    jT::Int = 1
    kYₑ::Int = 1
    value::Float64 = 0.0
end

"""
    SyntheticOptions

Grid, gas parameters and the defects to inject. Defects are applied in the
order the fields are declared below: flatten, wiggle, offset, stiffen, seed,
setvalue.
"""
Base.@kwdef struct SyntheticOptions
    nρ::Int = 40
    ρ_min_gcc::Float64 = 1.0e5
    ρ_max_gcc::Float64 = 1.0e15
    nT::Int = 30
    T_min_MeV::Float64 = 0.05
    T_max_MeV::Float64 = 50.0
    nYₑ::Int = 10
    Yₑ_min::Float64 = 0.05
    Yₑ_max::Float64 = 0.55
    energy_shift::Float64 = 1.5e18
    s₀::Float64 = 12.0
    T₀_MeV::Float64 = 0.05
    ρ₀_gcc::Float64 = 1.0e16
    with_aux_fields::Bool = false
    flatten::Vector{FlattenDefect} = FlattenDefect[]
    wiggle::Vector{WiggleDefect} = WiggleDefect[]
    offset::Vector{OffsetDefect} = OffsetDefect[]
    stiffen::Vector{StiffenDefect} = StiffenDefect[]
    seed::Vector{SeededViolation} = SeededViolation[]
    setvalue::Vector{SetValue} = SetValue[]
end

# ---------------------------------------------------------------------------
# The analytic model
#
# These closed forms are the ground truth the rest of the package is measured
# against, so they are written exactly as specified -- same operands in the
# same order -- and must not be algebraically rearranged. `opts` is threaded
# through all four even where it is unused, so a caller can swap one for
# another without touching the call site.
# ---------------------------------------------------------------------------

"""
    synthetic_eps(ρ_gcc, T_MeV, Yₑ, opts) -> Float64

Specific internal energy in erg/g: `1.5·g·kT/m_B` with `g = 1 + Yₑ`.
Independent of ρ, as an ideal gas must be.
"""
function synthetic_eps(ρ_gcc::Real, T_MeV::Real, Yₑ::Real, opts::SyntheticOptions)
    g = 1.0 + Yₑ
    kT_erg = T_MeV * MEV_TO_ERG
    return 1.5 * g * kT_erg / M_AMU_G
end

"""
    synthetic_p(ρ_gcc, T_MeV, Yₑ, opts) -> Float64

Pressure in dyn/cm²: `g·ρ·kT/m_B`.
"""
function synthetic_p(ρ_gcc::Real, T_MeV::Real, Yₑ::Real, opts::SyntheticOptions)
    g = 1.0 + Yₑ
    kT_erg = T_MeV * MEV_TO_ERG
    return g * ρ_gcc * kT_erg / M_AMU_G
end

"""
    synthetic_s(ρ_gcc, T_MeV, Yₑ, opts) -> Float64

Entropy in k_B per baryon: `g·(s₀ + 1.5·ln(T/T₀) - ln(ρ/ρ₀))`.

Note the reference density `opts.ρ₀_gcc` sits *above* the default grid, which
keeps the `-ln(ρ/ρ₀)` term positive everywhere and hence keeps `s > 0` over
the whole default grid.
"""
function synthetic_s(ρ_gcc::Real, T_MeV::Real, Yₑ::Real, opts::SyntheticOptions)
    g = 1.0 + Yₑ
    return g * (opts.s₀ + 1.5 * log(T_MeV / opts.T₀_MeV) - log(ρ_gcc / opts.ρ₀_gcc))
end

"""
    synthetic_cs2(ρ_gcc, T_MeV, Yₑ, opts) -> Float64

Sound speed squared in units of c², for the ideal gas above at Γ = 5/3.

`tests/test_check.cpp` carries an independent inline copy of this expression
and relies on it being bit-identical, so the operand order here is load-bearing
in the C++ and is preserved.
"""
function synthetic_cs2(ρ_gcc::Real, T_MeV::Real, Yₑ::Real, opts::SyntheticOptions)
    g = 1.0 + Yₑ
    kT_erg = T_MeV * MEV_TO_ERG
    ε = synthetic_eps(ρ_gcc, T_MeV, Yₑ, opts)
    p = synthetic_p(ρ_gcc, T_MeV, Yₑ, opts)
    m_B = M_B_DEFAULT_G
    c = C_LIGHT_CM_S
    h = 1.0 + (ε + p / ρ_gcc) / (c * c)
    return (5.0 / 3.0) * g * kT_erg / (m_B * h * c * c)
end

# ---------------------------------------------------------------------------
# Axes
#
# Written out rather than delegated to `range`: `range`'s twice-rounded
# endpoint-preserving arithmetic is *better*, but it is not the arithmetic the
# C++ does, and the grid coordinates feed straight into the stored values.
# ---------------------------------------------------------------------------

"""`n` log-uniform points in `[lo, hi]`, returned as their log10."""
function log_uniform_log10_axis(n::Integer, lo::Real, hi::Real)
    log_lo = log10(lo)
    log_hi = log10(hi)
    return [log_lo + (n > 1 ? (i - 1) / (n - 1) : 0.0) * (log_hi - log_lo) for i in 1:n]
end

"""`n` points linear in `[lo, hi]`."""
function linear_axis(n::Integer, lo::Real, hi::Real)
    return [lo + (n > 1 ? (i - 1) / (n - 1) : 0.0) * (hi - lo) for i in 1:n]
end

# 2π to full Float64 precision. Spelled out so the wiggle phase is reproducible
# against the C++ literal rather than against whatever `2 * π` rounds to.
const TWO_PI = 6.283185307179586476925286766559005768394338798750

# ---------------------------------------------------------------------------
# Defect injection
#
# All six injectors work on the *stored* representation -- post-log10 for
# "logenergy" -- and none of them uses an RNG, so a table is reproducible
# bit-for-bit from its options alone.
# ---------------------------------------------------------------------------

"""
Validate one block defect's inclusive index ranges against the grid.

The C++ raises `std::out_of_range` here; `ArgumentError` is its Julia
counterpart for a caller-supplied value that cannot be honoured, and matches
how `RawTable` reports a bad field shape.
"""
function check_block(t::RawTable, iρ0, iρ1, kYₑ0, kYₑ1, jT0, jT1, what::AbstractString)
    if iρ0 > iρ1 || iρ0 < 1 || iρ1 > nρ(t)
        throw(ArgumentError("$what: iρ range [$iρ0, $iρ1] invalid for nρ=$(nρ(t))"))
    end
    if kYₑ0 > kYₑ1 || kYₑ0 < 1 || kYₑ1 > nYₑ(t)
        throw(ArgumentError("$what: kYₑ range [$kYₑ0, $kYₑ1] invalid for nYₑ=$(nYₑ(t))"))
    end
    if jT0 > jT1 || jT0 < 1 || jT1 > nT(t)
        throw(ArgumentError("$what: jT range [$jT0, $jT1] invalid for nT=$(nT(t))"))
    end
    return nothing
end

"""Step 1: replace each block column along T by its value at `jT0`."""
function apply_flatten!(t::RawTable, defects::AbstractVector{FlattenDefect})
    for d in defects
        check_block(t, d.iρ0, d.iρ1, d.kYₑ0, d.kYₑ1, d.jT0, d.jT1, "FlattenDefect")
        data = field(t, d.field)
        for kYₑ in d.kYₑ0:d.kYₑ1, iρ in d.iρ0:d.iρ1
            v0 = data[iρ, d.jT0, kYₑ]     # read before the column is overwritten
            data[iρ, d.jT0:d.jT1, kYₑ] .= v0
        end
    end
    return t
end

"""Step 2: add `amplitude·sin(2π(jT - jT0)/period)` along T over the block."""
function apply_wiggle!(t::RawTable, defects::AbstractVector{WiggleDefect})
    for d in defects
        check_block(t, d.iρ0, d.iρ1, d.kYₑ0, d.kYₑ1, d.jT0, d.jT1, "WiggleDefect")
        data = field(t, d.field)
        for kYₑ in d.kYₑ0:d.kYₑ1, iρ in d.iρ0:d.iρ1, jT in d.jT0:d.jT1
            phase = TWO_PI * (jT - d.jT0) / d.period
            data[iρ, jT, kYₑ] += d.amplitude * sin(phase)
        end
    end
    return t
end

"""Step 3: add a constant to the whole block."""
function apply_offset!(t::RawTable, defects::AbstractVector{OffsetDefect})
    for d in defects
        check_block(t, d.iρ0, d.iρ1, d.kYₑ0, d.kYₑ1, d.jT0, d.jT1, "OffsetDefect")
        data = field(t, d.field)
        data[d.iρ0:d.iρ1, d.jT0:d.jT1, d.kYₑ0:d.kYₑ1] .+= d.offset
    end
    return t
end

"""
Step 4: add `amplitude·(ρ/ρ_c)^α` erg/g to the physical energy, everywhere.

The field is stored as a log10, so the term is added to `10^value` and the
result re-logged -- identical to adding it to ε, since `energy_shift` is a
constant. There is no cutoff: for α > 0 the term falls off as ρ^-α and drops
below the last bit of `ε + energy_shift` on its own, which is what keeps the
defect analytic in ρ and exactly independent of T and Yₑ.
"""
function apply_stiffen!(t::RawTable, defects::AbstractVector{StiffenDefect})
    for d in defects
        d.ρ_c_gcc > 0 || throw(ArgumentError("StiffenDefect: ρ_c_gcc must be positive, got $(d.ρ_c_gcc)"))
        data = field(t, d.field)
        for iρ in 1:nρ(t)
            term = d.amplitude * (density(t, iρ) / d.ρ_c_gcc)^d.α
            for kYₑ in 1:nYₑ(t), jT in 1:nT(t)
                data[iρ, jT, kYₑ] = log10(exp10(data[iρ, jT, kYₑ]) + term)
            end
        end
    end
    return t
end

"""Step 5: add `delta` at one node."""
function apply_seed!(t::RawTable, defects::AbstractVector{SeededViolation})
    for v in defects
        check_point(t, v.iρ, v.jT, v.kYₑ, "SeededViolation")
        field(t, v.field)[v.iρ, v.jT, v.kYₑ] += v.delta
    end
    return t
end

"""
Step 6, last: overwrite one node outright.

Last on purpose -- a planted `Inf`/`NaN` must not be perturbable by anything
that runs after it.
"""
function apply_setvalue!(t::RawTable, defects::AbstractVector{SetValue})
    for v in defects
        check_point(t, v.iρ, v.jT, v.kYₑ, "SetValue")
        field(t, v.field)[v.iρ, v.jT, v.kYₑ] = v.value
    end
    return t
end

"""Validate one single-node defect's index triple against the grid."""
function check_point(t::RawTable, iρ, jT, kYₑ, what::AbstractString)
    if !(1 <= iρ <= nρ(t) && 1 <= jT <= nT(t) && 1 <= kYₑ <= nYₑ(t))
        throw(ArgumentError("$what: index ($iρ, $jT, $kYₑ) out of range for grid $(size(t))"))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

"""
    make_synthetic_table(opts = SyntheticOptions()) -> RawTable{Float64}

Sample the analytic model on the grid described by `opts` and apply its
defects.

Stored fields are `"logenergy" = log10(ε + energy_shift)`, `"entropy" = s` and
`"logpress" = log10(p)`, plus the `"energy_shift"` attribute. With
`opts.with_aux_fields` three more follow -- `"cs2"`, `"gamma"` (5/3 flat) and
`"mu_e"` -- which the check/repair pipeline never interprets, so a defect
planted in one of them can only ever be reported as non-finite, never as fatal.

Defects are applied to the stored representation, after every field is in
place, in the fixed order flatten, wiggle, offset, stiffen, seed, setvalue,
each list running to completion before the next begins.
"""
function make_synthetic_table(opts::SyntheticOptions=SyntheticOptions())
    logρ = log_uniform_log10_axis(opts.nρ, opts.ρ_min_gcc, opts.ρ_max_gcc)
    logT = log_uniform_log10_axis(opts.nT, opts.T_min_MeV, opts.T_max_MeV)
    Yₑ = linear_axis(opts.nYₑ, opts.Yₑ_min, opts.Yₑ_max)
    table = RawTable(logρ, logT, Yₑ)

    dims = (opts.nρ, opts.nT, opts.nYₑ)
    logenergy = Array{Float64,3}(undef, dims)
    entropy = Array{Float64,3}(undef, dims)
    logpress = Array{Float64,3}(undef, dims)

    for kYₑ in 1:opts.nYₑ
        ye = Yₑ[kYₑ]
        for jT in 1:opts.nT
            T_MeV = exp10(logT[jT])
            for iρ in 1:opts.nρ
                ρ = exp10(logρ[iρ])
                ε = synthetic_eps(ρ, T_MeV, ye, opts)
                p = synthetic_p(ρ, T_MeV, ye, opts)
                s = synthetic_s(ρ, T_MeV, ye, opts)
                # The whole test suite treats this model as a *clean* table, so
                # a grid that pushes s through zero is a caller error, not a
                # defect to be tolerated.
                @assert s > 0 "synthetic entropy model must stay positive over the grid"
                logenergy[iρ, jT, kYₑ] = log10(ε + opts.energy_shift)
                entropy[iρ, jT, kYₑ] = s
                logpress[iρ, jT, kYₑ] = log10(p)
            end
        end
    end

    add_field!(table, "logenergy", logenergy)
    add_field!(table, "entropy", entropy)
    add_field!(table, "logpress", logpress)
    add_attribute!(table, "energy_shift", opts.energy_shift)

    if opts.with_aux_fields
        cs2 = Array{Float64,3}(undef, dims)
        gamma = Array{Float64,3}(undef, dims)
        mu_e = Array{Float64,3}(undef, dims)
        for kYₑ in 1:opts.nYₑ
            ye = Yₑ[kYₑ]
            for jT in 1:opts.nT
                T_MeV = exp10(logT[jT])
                kT_erg = T_MeV * MEV_TO_ERG
                for iρ in 1:opts.nρ
                    ρ = exp10(logρ[iρ])
                    cs2[iρ, jT, kYₑ] = synthetic_cs2(ρ, T_MeV, ye, opts)
                    gamma[iρ, jT, kYₑ] = 5.0 / 3.0
                    mu_e[iρ, jT, kYₑ] = (kT_erg / M_AMU_G) * ye
                end
            end
        end
        add_field!(table, "cs2", cs2)
        add_field!(table, "gamma", gamma)
        add_field!(table, "mu_e", mu_e)
    end

    apply_flatten!(table, opts.flatten)
    apply_wiggle!(table, opts.wiggle)
    apply_offset!(table, opts.offset)
    apply_stiffen!(table, opts.stiffen)
    apply_seed!(table, opts.seed)
    apply_setvalue!(table, opts.setvalue)

    return table
end

"""
    dirty_synthetic_options() -> SyntheticOptions

The default 40 × 30 × 10 grid with aux fields and a fixed set of defects, each
chosen to mimic a pathology actually observed in the LS220/SRO stellarcollapse
tables. Nothing here is random; the resulting table is the same every time.
"""
function dirty_synthetic_options()
    return SyntheticOptions(;
        with_aux_fields=true,

        # LS220's clustered non-monotone entropy across a T-window at high
        # ρ/Yₑ. The amplitude is comparable to the model's own per-T-step
        # entropy increment there (g·1.5·Δln T ≈ 0.35-0.5 k_B/baryon), so
        # several adjacent pairs inside the window actually decrease.
        wiggle=[WiggleDefect("entropy", 31, 36, 7, 9, 11, 21, 0.5, 4.0)],

        # SRO's near-flat logenergy plateaus, which repair's PAVA strictifies.
        flatten=[FlattenDefect("logenergy", 6, 16, 3, 5, 4, 13)],

        # SRO's slightly negative cold-corner entropies. The offset has to
        # clear the block's actual maximum, not merely nudge it: ρ₀_gcc (1e16)
        # sits above the whole default ρ grid, so -ln(ρ/ρ₀) is positive
        # throughout and largest at the smallest ρ, and this corner's s reaches
        # ~44.2 k_B/baryon. -50.0 carries enough margin to drive every entry in
        # the block below zero.
        offset=[OffsetDefect("entropy", 1, 4, 1, 3, 1, 3, -50.0)],

        # The LS220/SRO acausal high-density corner: a smooth ρ² energy excess
        # that makes the constructed U superluminal over the top ρ layers at
        # every T and Yₑ, while leaving monotonicity in T -- which it does not
        # depend on -- untouched. At ρ = 1e15 the term is 1.35e17·100² =
        # 1.35e21 erg/g, about 1.5 c², giving cs² up to ~1.63 at the top nodes.
        stiffen=[StiffenDefect("logenergy", 1.0e13, 2.0, 1.35e17)],

        # LS220's handful of genuinely non-finite cs2/gamma points.
        setvalue=[
            SetValue("cs2", 21, 16, 6, Inf),
            SetValue("cs2", 22, 17, 6, NaN),
            SetValue("gamma", 21, 16, 6, NaN),
        ],
    )
end

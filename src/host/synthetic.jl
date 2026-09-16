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

# Implementation lands in milestone M3:
#   synthetic_eps/synthetic_p/synthetic_s/synthetic_cs2(ρ, T, Yₑ, opts)
#   make_synthetic_table(opts = SyntheticOptions()) -> RawTable{Float64}
#   dirty_synthetic_options() -> SyntheticOptions

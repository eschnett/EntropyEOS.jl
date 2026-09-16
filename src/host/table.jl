# A verbatim in-memory image of a table file.
#
# Translated from `entropy_eos/host/table.{hpp,cpp}`.
#
# "Verbatim" is the point: every dataset is carried, not just the ones the
# library interprets, and no unit conversion happens on the way in. A
# convert-on-read pipeline would perturb datasets nothing ever looks at.
#
# Storage order matches the file exactly. The C++ flattens a field to
# `irho + nrho*(jT + ntemp*kYe)`; here a field is an `Array{T,3}` of size
# `(nρ, nT, nYₑ)`, whose column-major layout is the same bytes in the same
# order. Loading therefore needs no permutation, and adding one is a bug.

"""
    RawTable{T}

Axes, named 3-D fields, and named scalar attributes, exactly as stored in the
file. Field insertion order is preserved so that a write-back reproduces the
source dataset order.

Axes are `log10(ρ [g/cm³])`, `log10(T [MeV])` and linear `Yₑ`.
"""
struct RawTable{T<:AbstractFloat}
    logρ::Vector{T}
    logT::Vector{T}
    Yₑ::Vector{T}

    field_names::Vector{String}
    field_data::Vector{Array{T,3}}
    field_index::Dict{String,Int}

    attribute_names::Vector{String}
    attribute_values::Vector{T}
    attribute_index::Dict{String,Int}
end

"""
    RawTable{T}(logρ, logT, Yₑ)

An empty table on the given axes.
"""
function RawTable{T}(logρ::AbstractVector, logT::AbstractVector, Yₑ::AbstractVector) where {T<:AbstractFloat}
    return RawTable{T}(
        collect(T, logρ), collect(T, logT), collect(T, Yₑ),
        String[], Array{T,3}[], Dict{String,Int}(),
        String[], T[], Dict{String,Int}(),
    )
end

function RawTable(logρ::AbstractVector{T}, logT::AbstractVector, Yₑ::AbstractVector) where {T<:AbstractFloat}
    return RawTable{T}(logρ, logT, Yₑ)
end

Base.eltype(::RawTable{T}) where {T} = T

"""Number of density points."""
nρ(t::RawTable) = length(t.logρ)
"""Number of temperature points."""
nT(t::RawTable) = length(t.logT)
"""Number of electron-fraction points."""
nYₑ(t::RawTable) = length(t.Yₑ)

Base.size(t::RawTable) = (nρ(t), nT(t), nYₑ(t))

"""Density at index `i`, in g/cm³."""
density(t::RawTable, i::Integer) = exp10(t.logρ[i])
"""Temperature at index `j`, in MeV."""
temperature(t::RawTable, j::Integer) = exp10(t.logT[j])
"""Electron fraction at index `k`."""
electron_fraction(t::RawTable, k::Integer) = t.Yₑ[k]

"""
    validate_axes(t)

Check that every axis is finite and strictly increasing. Throws `ArgumentError`
naming the axis and the offending index.
"""
function validate_axes(t::RawTable)
    validate_axis(t.logρ, "logrho")
    validate_axis(t.logT, "logtemp")
    validate_axis(t.Yₑ, "ye")
    return nothing
end

function validate_axis(axis::AbstractVector, name::AbstractString)
    isempty(axis) && throw(ArgumentError("axis '$name' is empty"))
    for i in eachindex(axis)
        isfinite(axis[i]) || throw(ArgumentError("axis '$name' has a non-finite value at index $i"))
        if i > firstindex(axis) && !(axis[i] > axis[i - 1])
            throw(ArgumentError("axis '$name' is not strictly increasing at index $i"))
        end
    end
    return nothing
end

"""
    add_field!(t, name, data)

Add or replace a 3-D field. Re-adding an existing name overwrites the data in
place and keeps the name's position in the insertion order.
"""
function add_field!(t::RawTable{T}, name::AbstractString, data::AbstractArray) where {T}
    size(data) == size(t) ||
        throw(ArgumentError("field '$name' has size $(size(data)), expected $(size(t))"))
    arr = convert(Array{T,3}, data)
    key = String(name)
    idx = get(t.field_index, key, 0)
    if idx == 0
        push!(t.field_names, key)
        push!(t.field_data, arr)
        t.field_index[key] = length(t.field_names)
    else
        t.field_data[idx] = arr
    end
    return t
end

has_field(t::RawTable, name::AbstractString) = haskey(t.field_index, String(name))

"""
    field(t, name)

The named 3-D field, indexed `[iρ, jT, kYₑ]`. Throws `KeyError` if absent.
"""
function field(t::RawTable, name::AbstractString)
    key = String(name)
    idx = get(t.field_index, key, 0)
    idx == 0 && throw(KeyError(key))
    return t.field_data[idx]
end

"""Field names, in insertion order."""
field_names(t::RawTable) = t.field_names

"""
    add_attribute!(t, name, value)

Add or replace a scalar attribute. In the stellarcollapse format these are
stored as one-element datasets rather than as HDF5 attributes.
"""
function add_attribute!(t::RawTable{T}, name::AbstractString, value) where {T}
    key = String(name)
    idx = get(t.attribute_index, key, 0)
    if idx == 0
        push!(t.attribute_names, key)
        push!(t.attribute_values, T(value))
        t.attribute_index[key] = length(t.attribute_names)
    else
        t.attribute_values[idx] = T(value)
    end
    return t
end

has_attribute(t::RawTable, name::AbstractString) = haskey(t.attribute_index, String(name))

"""The named scalar attribute. Throws `KeyError` if absent."""
function attribute(t::RawTable, name::AbstractString)
    key = String(name)
    idx = get(t.attribute_index, key, 0)
    idx == 0 && throw(KeyError(key))
    return t.attribute_values[idx]
end

"""Attribute names, in insertion order."""
attribute_names(t::RawTable) = t.attribute_names

"""The `energy_shift` attribute, in erg/g."""
energy_shift(t::RawTable) = attribute(t, "energy_shift")

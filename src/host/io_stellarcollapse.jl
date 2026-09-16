# Reading stellarcollapse.org HDF5 tables.
#
# Translated from the read half of `entropy_eos/host/io_stellarcollapse.cpp`.
# The write, provenance and checksum halves belong to the offline C++ repair
# tool and are deliberately not ported.
#
# Format notes that are easy to get wrong:
#
#   * `energy_shift`, `have_rel_cs2` and `pointsrho`/`pointstemp`/`pointsye` are
#     one-element *datasets*, not HDF5 attributes. Reading the file's attributes
#     finds nothing.
#   * `points*` are cross-checks against the axis lengths, not axes themselves,
#     and are stored as `Int32` on real files.
#   * 3-D fields are stored C-order with dims `(nYₑ, nT, nρ)`, which HDF5.jl
#     hands back as a Julia column-major `(nρ, nT, nYₑ)` array. That is already
#     the layout `RawTable` wants, so no permutation is applied -- adding one is
#     the bug this note exists to prevent.
#   * Any root dataset of rank 3 with the right dims and a float element type
#     becomes a field. Everything else -- provenance blobs, groups such as
#     `/repair` -- is skipped.

# Qualified rather than `using HDF5`: HDF5 exports common words (`name`,
# `datatype`, `attributes`, ...) that would shadow or be shadowed by this
# package's own table vocabulary.
import HDF5

# Every scalar is read through a `Float64` memory type, exactly as the C++ does
# with `H5T_NATIVE_DOUBLE`. Real files store `points*` as `Int32` and
# `energy_shift` as `Float64`, and this makes both paths the same code.
function read_scalar_as_float64(dset::HDF5.Dataset, path::AbstractString, name::AbstractString)
    value = try
        read(dset)
    catch e
        error("read_stellarcollapse: '$path': failed to read scalar '$name' ($e)")
    end
    # A stellarcollapse "scalar" may have a `{1}` dataspace rather than a true
    # scalar one, so `read` gives back either a number or a one-element array.
    return Float64(value isa AbstractArray ? first(value) : value)
end

"""Read a required 1-D float axis. Throws if missing, not 1-D, or unreadable."""
function read_axis(file::HDF5.File, name::AbstractString, path::AbstractString)
    haskey(file, name) || error("read_stellarcollapse: '$path' has no dataset '$name'")
    obj = try
        file[name]
    catch e
        error("read_stellarcollapse: '$path': failed to open dataset '$name' ($e)")
    end
    try
        obj isa HDF5.Dataset || error("read_stellarcollapse: '$path': '$name' is not a dataset")
        ndims(obj) == 1 || error("read_stellarcollapse: '$path': dataset '$name' is not 1-D")
        data = try
            read(obj)
        catch e
            error("read_stellarcollapse: '$path': failed to read dataset '$name' ($e)")
        end
        return collect(Float64, data)
    finally
        close(obj)
    end
end

"""
Cross-check an optional `points*` scalar dataset against the axis size already
read. Absent is fine -- the contract only checks it "when present" -- but a
mismatch means the file's own bookkeeping disagrees with its axes.
"""
function check_points_dataset(file::HDF5.File, name::AbstractString, expected::Integer, path::AbstractString)
    haskey(file, name) || return nothing
    obj = try
        file[name]
    catch e
        error("read_stellarcollapse: '$path': failed to open '$name' ($e)")
    end
    try
        obj isa HDF5.Dataset || error("read_stellarcollapse: '$path': failed to open '$name'")
        rounded = round(Int, read_scalar_as_float64(obj, path, name))
        rounded == expected ||
            error("read_stellarcollapse: '$path': '$name' = $rounded does not match axis size $expected")
    finally
        close(obj)
    end
    return nothing
end

"""True iff the dataset's dataspace holds exactly one element, whatever its rank."""
is_scalar_dataset(dset::HDF5.Dataset) = length(dset) == 1

"""
True iff the dataset's stored type is a float type (f4 or f8).

Both are read identically: `read` upconverts an f4 dataset losslessly, so there
is no separate code path for "single" vs "double" fields. The HDF5 *class* is
what is tested, not a Julia type, because non-numeric datasets (opaque
provenance blobs, fixed-length strings) must be rejected rather than converted.
"""
function is_float_dataset(dset::HDF5.Dataset)
    dtype = HDF5.datatype(dset)
    try
        return HDF5.API.h5t_get_class(dtype) == HDF5.API.H5T_FLOAT
    finally
        close(dtype)
    end
end

"""
    read_stellarcollapse(path) -> RawTable{Float64}

Load a stellarcollapse.org-format HDF5 table verbatim: the three axes, every
conforming 3-D field, and the `energy_shift`/`have_rel_cs2` scalars. Nothing is
converted or interpreted on the way in.

Throws `ErrorException` naming the file (and the dataset, when there is one) if
the file cannot be opened, an axis is missing or not 1-D, or a `points*`
cross-check disagrees with the corresponding axis length.
"""
function read_stellarcollapse(path::AbstractString)
    file = try
        HDF5.h5open(path, "r")
    catch e
        error("read_stellarcollapse: cannot open file '$path' ($e)")
    end
    try
        logρ = read_axis(file, "logrho", path)
        logT = read_axis(file, "logtemp", path)
        Yₑ = read_axis(file, "ye", path)

        t = RawTable{Float64}(logρ, logT, Yₑ)
        nρ, nT, nYₑ = size(t)

        check_points_dataset(file, "pointsrho", nρ, path)
        check_points_dataset(file, "pointstemp", nT, path)
        check_points_dataset(file, "pointsye", nYₑ, path)

        # `keys` enumerates the root group by HDF5's name index, i.e.
        # alphabetically. That is the same order the C++ settles on for real
        # tables, which carry no link-creation-order index.
        for name in keys(file)
            name in ("logrho", "logtemp", "ye") && continue        # consumed as axes above
            name in ("pointsrho", "pointstemp", "pointsye") && continue  # cross-checked above

            obj = file[name]
            try
                # Not a dataset (e.g. a `/repair` provenance group) -- ignore.
                obj isa HDF5.Dataset || continue

                if name in ("energy_shift", "have_rel_cs2")
                    is_scalar_dataset(obj) && add_attribute!(t, name, read_scalar_as_float64(obj, path, name))
                    continue
                end

                # `size` of an HDF5 dataset is already reversed into Julia
                # order, so the file's C-order (nYₑ, nT, nρ) reads back as
                # (nρ, nT, nYₑ) -- which is what `RawTable` stores. Anything
                # else is an opaque blob or a non-conforming dataset and is
                # left out of the table entirely.
                if size(obj) == (nρ, nT, nYₑ) && is_float_dataset(obj)
                    data = try
                        read(obj)
                    catch e
                        error("read_stellarcollapse: '$path': failed to read field '$name' ($e)")
                    end
                    add_field!(t, name, data)
                end
            finally
                close(obj)
            end
        end

        return t
    finally
        close(file)
    end
end

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

# Implementation lands in milestone M4:
#   read_stellarcollapse(path) -> RawTable{Float64}

# Physical constants and unit conventions used by the host-side table and
# adapter code.
#
# Translated from `entropy_eos/host/units.hpp`.
#
# Nothing in `src/core/` uses any of this: the kernel-side view carries
# pre-computed conversion scalars (`conv_t`, `shift_hat`, `inv_c²`) instead, so
# the run-time path is unit-free. Keeping that true is a useful invariant.
#
# Table-side units are cgs plus MeV plus k_B per baryon:
#
#   ρ                 g/cm³, stored as log10
#   T                 MeV, stored as log10
#   Yₑ                dimensionless, linear
#   ε (`logenergy`)   erg/g, stored as log10(ε + energy_shift)
#   p (`logpress`)    dyn/cm², stored as log10
#   s (`entropy`)     k_B per baryon, linear
#
# Adapter-side units are dimensionless, geometrized by c²: ρ means ρ* in
# κ-rescaled g/cm³, U = ε/c², p is returned as p/c² in g/cm³, and D, τ, S∥, S⊥
# and B² are all g/cm³.

"""MeV to erg. CODATA 2018 exact: 1 eV = 1.602176634e-19 J."""
const MEV_TO_ERG = 1.602176634e-6

"""Boltzmann constant in erg/K. CODATA 2018 exact."""
const K_B_ERG_PER_K = 1.380649e-16

"""Speed of light in cm/s. Exact by SI definition."""
const C_LIGHT_CM_S = 2.99792458e10

"""Atomic mass unit in g. CODATA 2018."""
const M_AMU_G = 1.66053906892e-24

"""
Neutron mass in g. CODATA 2022.

Empirically this is the baryon-mass convention of the SRO
(Schneider-Roberts-Ott) tables: rebuilding the adapter with it collapses the
δT fidelity quantiles from a flat ~8.7e-3 -- the mₙ/mᵤ ratio -- to ~1.6e-5.
"""
const M_NEUTRON_G = 1.67492749804e-24

"""
Default baryon mass for converting between per-baryon and per-gram quantities.

The convention is per table family: some formats carry it explicitly, others
rely on a documented constant. This default is a placeholder that callers must
be able to override, and it must never be assumed for a real table without
checking that table's documented convention first. LS220, DD2 and SFHo use the
atomic mass unit; the SRO tables use [`M_NEUTRON_G`](@ref).
"""
const M_B_DEFAULT_G = M_AMU_G

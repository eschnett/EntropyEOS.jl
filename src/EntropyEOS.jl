"""
    EntropyEOS

Tabulated equations of state for general-relativistic hydrodynamics.

Three layers, each built on the one below:

  * loading and checking a tabulated EOS in the stellarcollapse format;
  * an *adapter* turning the usual `F(ρ, T, Yₑ)` table into the single
    thermodynamic potential `U(ρ, s, Yₑ)`, with every derivative a solver
    needs;
  * a `prim2con`/`con2prim` pair over that potential, using entropy and
    rapidity as iteration variables, plus a policy layer that never fails.

The evaluation and inversion kernels are allocation-free, exception-free and
generic in the scalar type, so they inline into a hydro code's inner loop and
run on a GPU once the coefficient arrays are moved there with `Adapt.adapt`.

This is a translation of the C++ library at
<https://github.com/eschnett/EntropyEOS>. Table *repair* is deliberately not
included: it is an offline activity performed once before a simulation
campaign, and the run-time path contains no repair logic. Use `check_table` to
detect a table that has not been repaired.

!!! note "Precision"
    All defaults and all validation are at `Float64`. Narrower scalar types
    are supported by the plumbing -- the code is type-generic, type-stable and
    allocation-free at `Float32` -- but the numerics are not validated there,
    and some tolerances are not representable.
"""
module EntropyEOS

using Adapt: Adapt
using EnumX: @enumx
using StaticArrays: MVector, SVector

# Kernel-side: no allocation, no exceptions, generic in the scalar type.
include("core/defs.jl")
include("core/bspline_eval.jl")
include("core/adapter_eval.jl")
include("core/adapt.jl")
include("core/prim2con.jl")
include("core/con2prim.jl")
include("core/state_policy.jl")

# Host-side: owns memory, may throw, never needed on a device.
include("host/units.jl")
include("host/table.jl")
include("host/bspline_fit.jl")
include("host/adapter_build.jl")
include("host/check.jl")
include("host/synthetic.jl")
include("host/io_stellarcollapse.jl")

export
    # Flags and outcomes
    FLAG_CLAMP_YE, FLAG_EXT_S_LOW, FLAG_EXT_S_HIGH, FLAG_EXT_ρ_LOW,
    FLAG_OOB_ρ_HIGH, FLAG_MAXITER,
    FLAG_POL_ATMOSPHERE, FLAG_POL_CEILING, FLAG_POL_S_FLOORED, FLAG_POL_S_CEILED,
    FLAG_POL_W_CAPPED, FLAG_POL_ρ_CLAMPED, FLAG_POL_YE_CLAMPED, FLAG_POL_NONFINITE,
    FLAG_POL_ANY,
    Status, C2PResult,
    # B-splines
    BsplineView1, BsplineView3, BsplineEval1, BsplineEval3,
    bspline_eval1, bspline_eval3, BandedLU, fit_bspline_1d, fit_bspline_3d,
    # Adapter
    EOSPoint, SRange, UHighTailInfo, EOSTableView, narrow,
    evaluate, eval_at, srange, srange_extended, sigma_extended, u_high_tail_info,
    # Solver
    Prim2ConOut, Con2PrimIn, Con2PrimOptions, Con2PrimOut, prim2con,
    PolicyOptions, PrimState, Con2PrimSafeOut,
    # Tables
    RawTable, validate_axes, add_field!, has_field, field, field_names,
    add_attribute!, has_attribute, attribute, attribute_names, energy_shift,
    density, temperature, electron_fraction,
    # Adapter build and checking
    BuildOptions, AdapterAudit, MonotonicityAudit, AuditLoc, EOSTable, build_eos,
    CheckOptions, CheckReport, CheckClassResult, CheckLoc,
    # Synthetic tables
    SyntheticOptions, SeededViolation, FlattenDefect, WiggleDefect, OffsetDefect,
    StiffenDefect, SetValue,
    make_synthetic_table, dirty_synthetic_options,
    synthetic_eps, synthetic_p, synthetic_s, synthetic_cs2,
    # Loading and checking
    read_stellarcollapse, check_table,
    # Units
    MEV_TO_ERG, K_B_ERG_PER_K, C_LIGHT_CM_S, M_AMU_G, M_NEUTRON_G, M_B_DEFAULT_G

# Exported as each milestone lands: prim2con, con2prim, con2prim_safe,
# default_policy, check_prim_state, project_prim_state, check_con_state.

end # module EntropyEOS

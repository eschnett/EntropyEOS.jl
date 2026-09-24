using EntropyEOS
using Test

# Shared by several test files. StableRNGs rather than Random: Julia's default
# streams are not guaranteed stable across minor versions, and a test that
# silently samples different points after an upgrade is not a test.
using ForwardDiff
using HDF5
using LinearAlgebra
using StaticArrays
using Adapt
using Aqua
using JET
using StableRNGs


# Ordered by dependency, so the first failing testset is the lowest broken
# layer rather than the alphabetically first one.
include("testutil.jl")

@testset "EntropyEOS" begin
    include("test_defs.jl")
    include("test_bspline_eval.jl")
    include("test_bspline_fit.jl")
    include("test_table.jl")
    include("test_synthetic.jl")
    include("test_io_stellarcollapse.jl")
    include("test_check.jl")
    include("test_adapter_tail.jl")
    include("test_adapter.jl")
    include("test_prim2con.jl")
    include("test_con2prim.jl")
    include("test_state_policy.jl")
    include("test_analytic_eos.jl")
    include("test_device_api.jl")
    include("test_gpu.jl")
    include("test_real_tables.jl")
    include("test_quality.jl")
end

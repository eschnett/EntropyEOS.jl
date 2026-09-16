using EntropyEOS
using Test

# Ordered by dependency, so the first failing testset is the lowest broken
# layer rather than the alphabetically first one.
@testset "EntropyEOS" begin
    include("test_defs.jl")
    include("test_bspline_eval.jl")
    include("test_table.jl")
end

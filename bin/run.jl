# Runs one of the C++ tools that EntropyEOS_jll ships, passing the remaining
# arguments through and exiting with the tool's own exit code:
#
#     julia --project=bin bin/run.jl eos_repair IN.h5 OUT.h5
#
# The shell scripts next to this file do exactly that. The tools live in this
# separate environment so that EntropyEOS itself never depends on the JLL, nor
# on the MPI build of HDF5 it pulls in.

using Pkg

# A fresh checkout has no installed environment yet; set it up on first use only.
try
    @eval using EntropyEOS_jll
catch
    Pkg.instantiate()
    @eval using EntropyEOS_jll
end

const TOOLS = Dict("eos_crop" => EntropyEOS_jll.eos_crop,
                   "eos_repair" => EntropyEOS_jll.eos_repair,
                   "eos_test" => EntropyEOS_jll.eos_test)

if isempty(ARGS) || !haskey(TOOLS, ARGS[1])
    println(stderr, "usage: julia --project=bin bin/run.jl {", join(sort!(collect(keys(TOOLS))), "|"), "} [ARGS...]")
    exit(64)
end
if !EntropyEOS_jll.is_available()
    println(stderr, "EntropyEOS_jll has no build for this platform")
    exit(69)
end

# `ignorestatus`: a nonzero exit is the tool's verdict (`eos_repair` exits 1
# when it changed something), to be passed on rather than thrown.
exit(run(ignorestatus(`$(TOOLS[ARGS[1]]()) $(ARGS[2:end])`)).exitcode)

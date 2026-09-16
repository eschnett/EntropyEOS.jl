# Shared test helpers.

"""
Whether an allocation measurement means anything in this session.

The kernels are allocation-free only as *optimized* code. Two common test
configurations defeat that, and neither indicates a bug:

  * `--check-bounds=yes` disables every `@inbounds` in the package, so the
    bounds checks come back and with them the possibility of a throw. The
    bracket scan's stack scratch is then heap-allocated (528 bytes for the two
    33-element buffers) rather than promoted to an `alloca`.
  * `--code-coverage` instruments every line and blocks many optimizations.

`julia-actions/julia-runtest` turns *both* on by default, which is why the CI
workflow runs one extra job with them off — that job is where these assertions
actually execute. Keeping bounds checking on elsewhere is worth more than the
allocation gate: it catches real indexing bugs.
"""
const ALLOCATION_TESTS_MEANINGFUL =
    Base.JLOptions().check_bounds != 1 && Base.JLOptions().code_coverage == 0

"""
    @test_noallocs expr

Assert `expr == 0`, or skip when an allocation measurement would be meaningless
here (see [`ALLOCATION_TESTS_MEANINGFUL`](@ref)). Skipped assertions show up in
the summary's Broken column rather than silently passing.
"""
macro test_noallocs(ex)
    return quote
        if ALLOCATION_TESTS_MEANINGFUL
            @test $(esc(ex)) == 0
        else
            @test_skip $(esc(ex)) == 0
        end
    end
end

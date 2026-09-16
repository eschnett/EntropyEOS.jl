# Generate documentation with `julia make.jl`

using Documenter
using EntropyEOS

makedocs(; sitename="EntropyEOS", format=Documenter.HTML(), modules=[EntropyEOS])
deploydocs(; repo="github.com/eschnett/EntropyEOS.jl.git", devbranch="main", push_preview=true)

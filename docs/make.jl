# Generate documentation with `julia make.jl`

using Documenter
using EntropyEOS

makedocs(;
    sitename="EntropyEOS",
    format=Documenter.HTML(),
    modules=[EntropyEOS],
    pages=[
        "Home" => "index.md",
        "Tables" => "tables.md",
        "Adapter" => "adapter.md",
        "Solver" => "solver.md",
        "Precision and GPUs" => "precision.md",
        "Internals" => "internals.md",
    ],
)
deploydocs(; repo="github.com/eschnett/EntropyEOS.jl.git", devbranch="main", push_preview=true)

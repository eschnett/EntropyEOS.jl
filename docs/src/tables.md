# Tables

Loading a tabulated EOS, checking it, and generating an analytic one.

Table *repair* is not part of this package: it is an offline activity performed
once before a simulation campaign, and the run-time path contains no repair
logic. [`check_table`](@ref) is what detects a table that has not been repaired.

```@autodocs
Modules = [EntropyEOS]
Pages = ["host/table.jl", "host/io_stellarcollapse.jl", "host/check.jl",
         "host/synthetic.jl", "host/units.jl"]
```

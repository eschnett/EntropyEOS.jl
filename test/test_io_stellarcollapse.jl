@testset "io_stellarcollapse" begin
    E = EntropyEOS
    # HDF5 is a dependency of EntropyEOS but not of the test environment, so the
    # test reaches it through the package rather than `using HDF5` directly.
    H5 = E.HDF5

    scratch = mktempdir(; cleanup = true)

    # Deliberately three different axis lengths: any accidental `permutedims`
    # in the reader then fails loudly on a size mismatch instead of silently
    # transposing a cube.
    nρ, nT, nYₑ = 5, 4, 3
    logρ = collect(range(5.0, 9.0; length = nρ))
    logT = collect(range(-1.0, 1.0; length = nT))
    Yₑ = collect(range(0.05, 0.55; length = nYₑ))
    n = nρ * nT * nYₑ

    logenergy = reshape(collect(1.0:n), nρ, nT, nYₑ)
    entropy = reshape(collect(101.0:(100.0 + n)), nρ, nT, nYₑ)
    cs2 = Float32.(reshape(collect(1:n) ./ (2n), nρ, nT, nYₑ))

    """Write a stellarcollapse-shaped file, then let `extra!` add to it."""
    function write_table_file(name; skip = String[], extra! = _ -> nothing)
        path = joinpath(scratch, name)
        H5.h5open(path, "w") do f
            "logrho" in skip || (f["logrho"] = logρ)
            "logtemp" in skip || (f["logtemp"] = logT)
            "ye" in skip || (f["ye"] = Yₑ)
            # `points*` are Int32 one-element datasets on real files, and
            # `energy_shift` is a Float64 one -- both are read through a
            # Float64 memory type.
            "pointsrho" in skip || (f["pointsrho"] = Int32[nρ])
            "pointstemp" in skip || (f["pointstemp"] = Int32[nT])
            "pointsye" in skip || (f["pointsye"] = Int32[nYₑ])
            f["energy_shift"] = [1.5e18]
            f["have_rel_cs2"] = Int32[1]
            f["logenergy"] = logenergy
            f["entropy"] = entropy
            f["cs2"] = cs2
            extra!(f)
        end
        return path
    end

    @testset "round trip" begin
        path = write_table_file("basic.h5")
        t = E.read_stellarcollapse(path)

        @test t isa RawTable{Float64}
        @test size(t) == (nρ, nT, nYₑ)
        @test t.logρ ≈ logρ
        @test t.logT ≈ logT
        @test t.Yₑ ≈ Yₑ

        @test has_field(t, "logenergy") && has_field(t, "entropy") && has_field(t, "cs2")
        @test field(t, "logenergy") == logenergy
        @test field(t, "entropy") == entropy
        # An f4 field is upconverted losslessly, not read as Float32.
        @test field(t, "cs2") isa Array{Float64,3}
        @test field(t, "cs2") ≈ cs2

        # The file stores C-order (nYₑ, nT, nρ); the reader must apply no
        # permutation on the way in. ρ varies fastest in the Julia array.
        f = field(t, "logenergy")
        @test f[2, 1, 1] - f[1, 1, 1] == 1
        @test f[1, 2, 1] - f[1, 1, 1] == nρ
        @test f[1, 1, 2] - f[1, 1, 1] == nρ * nT

        @test has_attribute(t, "energy_shift") && has_attribute(t, "have_rel_cs2")
        @test energy_shift(t) == 1.5e18
        @test attribute(t, "have_rel_cs2") == 1.0
    end

    @testset "raw file layout" begin
        # Independent confirmation that what HDF5.jl wrote really is C-order
        # (nYₑ, nT, nρ), so the round-trip test above is not two cancelling
        # permutations.
        path = write_table_file("layout.h5")
        H5.h5open(path, "r") do f
            dset = f["logenergy"]
            space = H5.dataspace(dset)
            dims, _ = H5.API.h5s_get_simple_extent_dims(space)
            @test dims == [nYₑ, nT, nρ]      # HDF5's own, C order
            @test size(dset) == (nρ, nT, nYₑ) # HDF5.jl's reversed, Julia order
        end
    end

    @testset "non-conforming datasets are skipped" begin
        path = write_table_file("extras.h5"; extra! = function (f)
            f["wrong_shape"] = zeros(nρ, nT, nYₑ + 1)       # right rank, wrong dims
            f["wrong_rank"] = zeros(nρ, nT)                 # right dims, wrong rank
            f["wrong_type"] = fill(Int32(3), nρ, nT, nYₑ)   # right shape, integer class
            f["opaque_blob"] = UInt8[1, 2, 3]               # a provenance blob
            g = H5.create_group(f, "repair")                # a group, not a dataset
            g["old_value"] = [1.0, 2.0]
        end)
        t = E.read_stellarcollapse(path)

        @test sort(field_names(t)) == ["cs2", "entropy", "logenergy"]
        for skipped in ("wrong_shape", "wrong_rank", "wrong_type", "opaque_blob", "repair",
                        "pointsrho", "pointstemp", "pointsye", "logrho", "logtemp", "ye")
            @test !has_field(t, skipped)
        end
        # `points*` are cross-checks, never attributes.
        @test attribute_names(t) == ["energy_shift", "have_rel_cs2"]
    end

    @testset "missing points datasets are not an error" begin
        path = write_table_file("nopoints.h5"; skip = ["pointsrho", "pointstemp", "pointsye"])
        t = E.read_stellarcollapse(path)
        @test size(t) == (nρ, nT, nYₑ)
    end

    @testset "errors" begin
        @test_throws ErrorException E.read_stellarcollapse(joinpath(scratch, "does_not_exist.h5"))

        for missing_axis in ("logrho", "logtemp", "ye")
            path = write_table_file("no_$missing_axis.h5"; skip = [missing_axis])
            e = try
                E.read_stellarcollapse(path)
                nothing
            catch err
                err
            end
            @test e isa ErrorException
            @test occursin(missing_axis, e.msg) && occursin(path, e.msg)
        end

        # A `points*` value that disagrees with its axis means the file's own
        # bookkeeping is inconsistent.
        path = write_table_file("badpoints.h5"; skip = ["pointstemp"], extra! = f -> (f["pointstemp"] = Int32[nT + 7]))
        e = try
            E.read_stellarcollapse(path)
            nothing
        catch err
            err
        end
        @test e isa ErrorException
        @test occursin("pointstemp", e.msg) && occursin(string(nT + 7), e.msg) && occursin(string(nT), e.msg)

        # An axis of the wrong rank is an error, not a silently skipped field.
        path = write_table_file("rank2axis.h5"; skip = ["ye"], extra! = f -> (f["ye"] = zeros(2, 2)))
        e = try
            E.read_stellarcollapse(path)
            nothing
        catch err
            err
        end
        @test e isa ErrorException
        @test occursin("ye", e.msg) && occursin("1-D", e.msg)
    end

    @testset "real LS220 crop" begin
        # Not in git: skip cleanly where the file is absent (CI), but exercise
        # the real format -- Int32 `points*`, an f4/f8 field mix, Inf in `cs2`
        # -- wherever it is present.
        crop = "/Users/eschnett/src/EntropyEOS/tables/LS220_san_crop.h5"
        if !isfile(crop)
            @info "LS220 crop table not found -- skipping" path = crop
        else
            t = E.read_stellarcollapse(crop)
            @test t isa RawTable{Float64}
            @test validate_axes(t) === nothing
            @test all(("logenergy", "entropy", "logpress", "cs2")) do name
                has_field(t, name)
            end
            @test has_attribute(t, "energy_shift")
            @test energy_shift(t) > 0
            @test size(field(t, "entropy")) == size(t)
            @test !has_field(t, "pointsrho")
        end
    end
end

@testset "table" begin
    E = EntropyEOS

    logρ = [1.0, 2.0, 3.0]
    logT = [0.0, 0.5, 1.0, 1.5]
    Yₑ = [0.1, 0.2]

    @testset "construction and axes" begin
        t = RawTable(logρ, logT, Yₑ)
        @test size(t) == (3, 4, 2)
        @test E.nρ(t) == 3 && E.nT(t) == 4 && E.nYₑ(t) == 2
        @test eltype(t) === Float64
        @test density(t, 1) ≈ 10.0
        @test temperature(t, 3) ≈ 10.0
        @test electron_fraction(t, 2) == 0.2
        @test validate_axes(t) === nothing
    end

    @testset "axis validation" begin
        @test_throws ArgumentError validate_axes(RawTable([1.0, 1.0], logT, Yₑ))       # repeated
        @test_throws ArgumentError validate_axes(RawTable([2.0, 1.0], logT, Yₑ))       # decreasing
        @test_throws ArgumentError validate_axes(RawTable([1.0, NaN], logT, Yₑ))       # NaN
        @test_throws ArgumentError validate_axes(RawTable([1.0, Inf], logT, Yₑ))       # Inf
        @test_throws ArgumentError validate_axes(RawTable(logρ, [1.0, 0.0, 2.0], Yₑ))  # wrong axis
        @test_throws ArgumentError validate_axes(RawTable(logρ, logT, [0.2, 0.1]))
    end

    @testset "storage layout" begin
        # The C++ flattens a field to irho + nrho*(jT + ntemp*kYe); the Julia
        # Array{T,3} is column-major over the same axes, so the bytes are in the
        # same order and loading needs no permutation. Adding one is the bug
        # this test exists to catch.
        t = RawTable(logρ, logT, Yₑ)
        d = reshape(collect(1.0:24.0), 3, 4, 2)
        add_field!(t, "entropy", d)
        f = field(t, "entropy")
        @test f == d
        @test f[2, 1, 1] - f[1, 1, 1] == 1        # ρ varies fastest
        @test f[1, 2, 1] - f[1, 1, 1] == 3        # then T
        @test f[1, 1, 2] - f[1, 1, 1] == 12       # then Yₑ
        @test vec(f) == collect(1.0:24.0)
    end

    @testset "fields" begin
        t = RawTable(logρ, logT, Yₑ)
        add_field!(t, "entropy", zeros(3, 4, 2))
        add_field!(t, "logenergy", ones(3, 4, 2))
        @test field_names(t) == ["entropy", "logenergy"]     # insertion order preserved
        @test has_field(t, "entropy") && !has_field(t, "nope")
        # Re-adding overwrites in place and keeps the position.
        add_field!(t, "entropy", fill(7.0, 3, 4, 2))
        @test field_names(t) == ["entropy", "logenergy"]
        @test all(==(7.0), field(t, "entropy"))
        @test_throws ArgumentError add_field!(t, "bad", zeros(2, 2, 2))
        @test_throws KeyError field(t, "nope")
    end

    @testset "attributes" begin
        t = RawTable(logρ, logT, Yₑ)
        add_attribute!(t, "energy_shift", 1.5e18)
        add_attribute!(t, "have_rel_cs2", 1.0)
        @test attribute_names(t) == ["energy_shift", "have_rel_cs2"]
        @test energy_shift(t) == 1.5e18
        @test has_attribute(t, "have_rel_cs2") && !has_attribute(t, "nope")
        add_attribute!(t, "energy_shift", 2.0e18)
        @test attribute_names(t) == ["energy_shift", "have_rel_cs2"]
        @test energy_shift(t) == 2.0e18
        @test_throws KeyError attribute(t, "nope")
        @test_throws KeyError energy_shift(RawTable(logρ, logT, Yₑ))
    end

    @testset "scalar type" begin
        t = RawTable{Float32}(logρ, logT, Yₑ)
        @test eltype(t) === Float32
        add_field!(t, "entropy", zeros(Float64, 3, 4, 2))
        @test field(t, "entropy") isa Array{Float32,3}
    end
end

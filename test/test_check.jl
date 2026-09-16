@testset "check" begin
    E = EntropyEOS

    # ------------------------------------------------------------------
    # A hand-built ideal-gas table.
    #
    # `make_synthetic_table` is not available here, so the reference table is
    # constructed inline from the closed-form non-relativistic ideal gas with
    # Γ = 5/3 and a Yₑ-dependent particle count g = 1 + Yₑ:
    #
    #   p = ρ g kT / m_B,   ε = (3/2) g kT / m_B,   s = g (3/2 ln T - ln ρ) + s₀
    #
    # That `s` is exactly the one the two Maxwell relations the checker tests
    # demand (∂ε/∂T = (kT/m_B) ∂s/∂T and ∂s/∂ρ = -(m_B/ρ²k) ∂p/∂T), so every
    # class D/E metric on this table is pure finite-difference truncation
    # error. `s₀` is chosen large enough that `s` stays positive over the grid.
    # ------------------------------------------------------------------

    m_B = E.M_B_DEFAULT_G
    c = E.C_LIGHT_CM_S
    SHIFT = 1.0e17
    S0 = 50.0

    """A clean table on a `(nρ, nT, nYₑ)` grid, optionally without some fields."""
    function ideal_gas_table(; nρ = 40, nT = 40, nYₑ = 4, fields = ("logenergy", "entropy", "logpress", "cs2"))
        logρ = collect(range(6.0, 9.0; length = nρ))
        logT = collect(range(-1.0, 1.0; length = nT))
        Yₑ = collect(range(0.05, 0.55; length = nYₑ))
        t = RawTable(logρ, logT, Yₑ)

        logenergy = zeros(nρ, nT, nYₑ)
        entropy = zeros(nρ, nT, nYₑ)
        logpress = zeros(nρ, nT, nYₑ)
        cs2 = zeros(nρ, nT, nYₑ)
        for k in 1:nYₑ, j in 1:nT, i in 1:nρ
            ρ = density(t, i)
            temp = temperature(t, j)
            g = 1.0 + electron_fraction(t, k)
            kT = temp * E.MEV_TO_ERG
            p = ρ * g * kT / m_B
            ε = 1.5 * g * kT / m_B
            h = 1.0 + (ε + p / ρ) / c^2
            logenergy[i, j, k] = log10(ε + SHIFT)
            entropy[i, j, k] = g * (1.5 * log(temp) - log(ρ)) + S0
            logpress[i, j, k] = log10(p)
            cs2[i, j, k] = (5 / 3) * g * kT / (m_B * h * c^2)
        end

        # Insertion order is the report's `nonfinite_*` order, so keep it fixed.
        for (name, data) in (("logenergy", logenergy), ("entropy", entropy),
                             ("logpress", logpress), ("cs2", cs2))
            name in fields && add_field!(t, name, data)
        end
        add_attribute!(t, "energy_shift", SHIFT)
        return t
    end

    find_class(r, name) = (i = findfirst(c -> c.name == name, r.classes); i === nothing ? nothing : r.classes[i])
    class_names(r) = [c.name for c in r.classes]

    # ------------------------------------------------------------------

    @testset "clean table" begin
        t = ideal_gas_table()
        r = E.check_table(t)

        @test r isa E.CheckReport{Float64}
        @test r.status === Status.ok
        @test isempty(r.fatal_messages)

        # Exact class set and order: both are part of the contract.
        @test class_names(r) == ["entropy_negative", "entropy_nonmonotone_T", "logenergy_nonmonotone_T",
                                 "delta_T", "delta_p", "maxwell_s_rho", "cs2_out_of_range", "cs2_vs_fd"]

        for name in ("entropy_negative", "entropy_nonmonotone_T", "logenergy_nonmonotone_T",
                     "cs2_out_of_range")
            cls = find_class(r, name)
            @test cls.count == 0
            @test isempty(cls.worst)
            @test cls.max == 0.0 && cls.rms == 0.0
        end

        # Class D/E on this model is finite-difference truncation error only.
        for name in ("delta_T", "delta_p", "maxwell_s_rho", "cs2_vs_fd")
            cls = find_class(r, name)
            @test isfinite(cls.max) && isfinite(cls.rms)
            @test cls.max < 3.0e-2
            @test cls.count == 0
            # A diagnostic class keeps its worst offenders even when nothing
            # exceeds the threshold -- that is what distinguishes it from a
            # violation class.
            @test length(cls.worst) == E.CheckOptions().worst_n
        end
    end

    @testset "entropy_negative" begin
        t = ideal_gas_table(; nρ = 6, nT = 6, nYₑ = 3)
        field(t, "entropy")[3, 4, 2] = -7.5
        r = E.check_table(t)

        @test r.status === Status.ok
        cls = find_class(r, "entropy_negative")
        @test cls.count == 1
        @test length(cls.worst) == 1
        loc = cls.worst[1]
        @test (loc.iρ, loc.jT, loc.kYₑ) == (3, 4, 2)
        @test loc.value == -7.5
        @test loc.ρ ≈ density(t, 3)
        @test loc.temp ≈ temperature(t, 4)
        @test loc.ye ≈ electron_fraction(t, 2)
        # A violation class takes rms over *all* points, but only violating
        # points contribute to the sum.
        @test cls.rms ≈ sqrt(7.5^2 / length(field(t, "entropy")))
    end

    @testset "monotonicity in T" begin
        t = ideal_gas_table(; nρ = 6, nT = 8, nYₑ = 3)
        # A large drop at (4, 5, 2) makes the pair (jT = 4 -> 5) decrease while
        # leaving (5 -> 6) increasing, so exactly one pair violates.
        field(t, "entropy")[4, 5, 2] -= 1000.0
        field(t, "logenergy")[2, 6, 1] -= 1000.0
        r = E.check_table(t)

        @test r.status === Status.ok
        @test isempty(r.fatal_messages)

        cs = find_class(r, "entropy_nonmonotone_T")
        @test cs.count == 1
        @test (cs.worst[1].iρ, cs.worst[1].jT, cs.worst[1].kYₑ) == (4, 4, 2)
        @test cs.worst[1].value < 0

        ce = find_class(r, "logenergy_nonmonotone_T")
        @test ce.count == 1
        @test (ce.worst[1].iρ, ce.worst[1].jT, ce.worst[1].kYₑ) == (2, 5, 1)
        @test ce.worst[1].value < 0
    end

    @testset "cs2 range and FD diagnostic" begin
        t = ideal_gas_table(; nρ = 12, nT = 12, nYₑ = 3)
        field(t, "cs2")[10, 7, 2] = 1.5      # above the causal bound
        field(t, "cs2")[4, 3, 1] = -0.25     # negative
        r = E.check_table(t)

        cls = find_class(r, "cs2_out_of_range")
        @test cls.count == 2
        @test length(cls.worst) == 2
        # Sorted by |value| descending; the metric is `v - 1` above the bound
        # and `v` itself at or below zero.
        @test (cls.worst[1].iρ, cls.worst[1].jT, cls.worst[1].kYₑ) == (10, 7, 2)
        @test cls.worst[1].value ≈ 0.5
        @test (cls.worst[2].iρ, cls.worst[2].jT, cls.worst[2].kYₑ) == (4, 3, 1)
        @test cls.worst[2].value ≈ -0.25

        # cs2_vs_fd is report-only: the corrupted points show up as its worst
        # offenders but nothing about them is fatal.
        vs_fd = find_class(r, "cs2_vs_fd")
        @test vs_fd !== nothing
        @test isfinite(vs_fd.max)
        @test r.status === Status.ok
    end

    @testset "skipped classes" begin
        # No logpress: a single NaN-valued placeholder, and no cs2_vs_fd.
        t = ideal_gas_table(; nρ = 5, nT = 5, nYₑ = 2, fields = ("logenergy", "entropy", "cs2"))
        r = E.check_table(t)
        @test class_names(r) == ["entropy_negative", "entropy_nonmonotone_T", "logenergy_nonmonotone_T",
                                 "maxwell_consistency", "cs2_out_of_range"]
        cls = find_class(r, "maxwell_consistency")
        @test cls.count == 0
        @test isnan(cls.max) && isnan(cls.rms)   # NaN, not a misleadingly clean zero
        @test isempty(cls.worst)

        # Fewer than 3 points on a differentiated axis: fd3 cannot run.
        t2 = ideal_gas_table(; nρ = 2, nT = 5, nYₑ = 2)
        r2 = E.check_table(t2)
        @test isnan(find_class(r2, "maxwell_consistency").max)
        @test find_class(r2, "cs2_vs_fd") === nothing
        @test find_class(r2, "cs2_out_of_range") !== nothing
    end

    @testset "non-finite in an interpreted field is fatal" begin
        for (name, value) in (("entropy", NaN), ("logenergy", Inf))
            t = ideal_gas_table(; nρ = 5, nT = 5, nYₑ = 2)
            field(t, name)[3, 2, 2] = value
            r = E.check_table(t)

            @test r.status === Status.fatal
            @test any(m -> occursin(name, m), r.fatal_messages)
            @test any(m -> occursin("iρ=3", m) && occursin("jT=2", m) && occursin("kYₑ=2", m),
                      r.fatal_messages)
            # Once a structural check fails the later classes are skipped.
            @test isempty(r.classes)
        end
    end

    @testset "non-finite in a non-interpreted field is not fatal" begin
        # The shipped LS220 table carries Inf in cs2/gamma, and logpress is
        # diagnostic-only, so neither may take the table down.
        t = ideal_gas_table(; nρ = 6, nT = 6, nYₑ = 3)
        field(t, "logpress")[3, 4, 3] = Inf
        field(t, "cs2")[2, 2, 1] = NaN
        r = E.check_table(t)

        @test r.status === Status.ok
        @test isempty(r.fatal_messages)

        nf_p = find_class(r, "nonfinite_logpress")
        @test nf_p.count == 1
        @test (nf_p.worst[1].iρ, nf_p.worst[1].jT, nf_p.worst[1].kYₑ) == (3, 4, 3)
        @test nf_p.worst[1].value == Inf       # the metric is 1, the location keeps the value
        @test nf_p.max == 1.0

        nf_c = find_class(r, "nonfinite_cs2")
        @test nf_c.count == 1
        # A NaN cs2 must not be double-reported as a range violation.
        @test find_class(r, "cs2_out_of_range").count == 0

        # The classes appear right after entropy_negative, in field order.
        @test class_names(r)[1:3] == ["entropy_negative", "nonfinite_logpress", "nonfinite_cs2"]

        # Diagnostics still ran, and stayed finite despite the poisoned points.
        @test isfinite(find_class(r, "delta_T").max)
        @test isfinite(find_class(r, "delta_T").rms)

        # A clean table gets no nonfinite class at all.
        @test !any(startswith("nonfinite_"), class_names(E.check_table(ideal_gas_table(; nρ = 5, nT = 5, nYₑ = 2))))
    end

    @testset "structural failures" begin
        logρ, logT, Yₑ = [6.0, 7.0, 8.0], [-1.0, 0.0, 1.0], [0.1, 0.3]

        # Missing 'entropy'.
        t = RawTable(logρ, logT, Yₑ)
        add_field!(t, "logenergy", fill(18.0, 3, 3, 2))
        add_attribute!(t, "energy_shift", 1.5e18)
        r = E.check_table(t)
        @test r.status === Status.fatal
        @test any(m -> occursin("entropy", m), r.fatal_messages)

        # Missing 'energy_shift'.
        t = RawTable(logρ, logT, Yₑ)
        add_field!(t, "logenergy", fill(18.0, 3, 3, 2))
        add_field!(t, "entropy", fill(1.0, 3, 3, 2))
        r = E.check_table(t)
        @test r.status === Status.fatal
        @test any(m -> occursin("energy_shift", m), r.fatal_messages)

        # A bad axis is caught rather than thrown.
        t = RawTable([7.0, 6.0, 8.0], logT, Yₑ)
        add_field!(t, "logenergy", fill(18.0, 3, 3, 2))
        add_field!(t, "entropy", fill(1.0, 3, 3, 2))
        add_attribute!(t, "energy_shift", 1.5e18)
        r = E.check_table(t)
        @test r.status === Status.fatal
        @test any(m -> startswith(m, "axes: "), r.fatal_messages)

        # Every sub-check runs: an empty table reports all of them at once.
        t = RawTable(logρ, [1.0, 0.0, -1.0], Yₑ)
        r = E.check_table(t)
        @test r.status === Status.fatal
        @test length(r.fatal_messages) == 4      # axes, logenergy, entropy, energy_shift
        @test isempty(r.classes)
    end

    @testset "worst list is bounded and ordered" begin
        t = ideal_gas_table(; nρ = 6, nT = 6, nYₑ = 3, fields = ("logenergy", "entropy"))
        s = field(t, "entropy")
        # Twelve negative entropies of strictly increasing magnitude.
        for i in 1:6, j in (2, 5)
            s[i, j, 1] = -(10.0 * i + j)
        end
        r = E.check_table(t, E.CheckOptions(; worst_n = 4))

        cls = find_class(r, "entropy_negative")
        @test cls.count == 12
        @test length(cls.worst) == 4
        values = [loc.value for loc in cls.worst]
        @test issorted(abs.(values); rev = true)
        # Keeps the four largest magnitudes, not the four seen first.
        @test abs.(values) == sort([10.0 * i + j for i in 1:6 for j in (2, 5)]; rev = true)[1:4]

        # worst_n = 0 keeps nothing but still counts.
        r0 = E.check_table(t, E.CheckOptions(; worst_n = 0))
        @test find_class(r0, "entropy_negative").count == 12
        @test isempty(find_class(r0, "entropy_negative").worst)
    end

    @testset "options" begin
        t = ideal_gas_table(; nρ = 20, nT = 20, nYₑ = 3)
        # A tolerance far below the grid's truncation error turns the class D
        # diagnostics into nonzero counts without changing max/rms.
        loose = E.check_table(t, E.CheckOptions(; tol_consistency = 0.05))
        tight = E.check_table(t, E.CheckOptions(; tol_consistency = 1.0e-12))
        @test find_class(loose, "delta_p").count == 0
        @test find_class(tight, "delta_p").count > 0
        @test find_class(tight, "delta_p").max == find_class(loose, "delta_p").max

        # m_B_g enters delta_T directly, so a wrong baryon mass shows up there.
        wrong = E.check_table(t, E.CheckOptions(; m_B_g = 2 * E.M_B_DEFAULT_G))
        @test find_class(wrong, "delta_T").max > 0.4
    end

    @testset "show" begin
        t = ideal_gas_table(; nρ = 6, nT = 6, nYₑ = 3, fields = ("logenergy", "entropy", "cs2"))
        field(t, "entropy")[2, 3, 1] = -1.0
        field(t, "cs2")[1, 1, 1] = Inf
        r = E.check_table(t)

        text = sprint(show, MIME"text/plain"(), r)
        @test occursin("check_table report", text)
        @test occursin("ok", text)
        for name in class_names(r)
            @test occursin(name, text)
        end
        @test occursin("skipped", text)            # the maxwell_consistency placeholder
        @test occursin("worst offenders", text)

        # A fatal report prints its messages and does not throw.
        rf = E.check_table(RawTable([6.0, 7.0], [0.0, 1.0], [0.1, 0.2]))
        ftext = sprint(show, MIME"text/plain"(), rf)
        @test occursin("fatal", ftext)
        @test occursin("missing required field 'entropy'", ftext)
    end
end

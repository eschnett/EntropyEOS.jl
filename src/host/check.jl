# Table diagnostics.
#
# Translated from `entropy_eos/host/check.{hpp,cpp}`.
#
# `check_table` is pure and never throws: structural problems that would make
# the later checks unsafe become fatal messages instead, and the remaining
# classes are then skipped. That makes it cheap enough to run in-process at
# startup, right after loading a table, which is the intended use.
#
# What it validates: axes finite and strictly increasing; the required fields
# and the `energy_shift` attribute present and finite; entropy nonnegative;
# entropy and logenergy strictly increasing in T; Maxwell consistency against
# the stored pressure by finite differences; and the stored sound speed both
# for range and against a finite-difference estimate.
#
# Non-finite values in a field the pipeline never interprets are reported but
# are *not* fatal -- which is exactly what lets LS220 be used despite carrying
# a few genuine Inf points in `cs2` and `gamma`.

"""
    CheckOptions

`m_B_g` is the table family's baryon-mass convention; see [`M_B_DEFAULT_G`](@ref).
"""
Base.@kwdef struct CheckOptions
    m_B_g::Float64 = M_B_DEFAULT_G
    tol_consistency::Float64 = 0.05
    worst_n::Int = 10
end

"""A grid location and the metric value found there, with physical coordinates."""
struct CheckLoc{T<:AbstractFloat}
    iρ::Int
    jT::Int
    kYₑ::Int
    value::T
    ρ::T
    temp::T
    ye::T
end

"""
    CheckClassResult

One named class of finding.

For a *violation* class, `count`, `max` and `rms` are taken over violating
points only. For a *diagnostic* class, `max` and `rms` are over every evaluated
point and `count` is how many exceeded the threshold. A skipped class carries
NaN rather than a misleading zero.
"""
struct CheckClassResult{T<:AbstractFloat}
    name::String
    count::Int
    max::T
    rms::T
    worst::Vector{CheckLoc{T}}
end

"""
    CheckReport

`status` is `fatal` when the table is structurally broken -- a missing or
non-finite interpreted field, a bad axis -- as opposed to merely carrying
physics noise.
"""
struct CheckReport{T<:AbstractFloat}
    status::Status.T
    fatal_messages::Vector{String}
    classes::Vector{CheckClassResult{T}}
end

# ---------------------------------------------------------------------------
# Accumulators
#
# Indices in a `CheckLoc` are 1-based, unlike the C++ `size_t`s they come from:
# they are meant to be fed straight back to `density`/`temperature`/
# `electron_fraction` and to `field(t, name)[iρ, jT, kYₑ]`.
# ---------------------------------------------------------------------------

function make_loc(t::RawTable{T}, iρ::Int, jT::Int, kYₑ::Int, value) where {T}
    return CheckLoc{T}(iρ, jT, kYₑ, T(value), T(density(t, iρ)), T(temperature(t, jT)), T(electron_fraction(t, kYₑ)))
end

"""
Bounded "top `n` by `|value|`" collector.

Eviction replaces the *smallest-magnitude* entry currently held, and only on a
strict improvement, so equal-magnitude candidates never displace an incumbent.
Insertion is O(n) with n a small constant, which is fine at one call per
evaluated point.
"""
struct TopK{T<:AbstractFloat}
    n::Int
    items::Vector{CheckLoc{T}}
end

TopK{T}(n::Integer) where {T} = TopK{T}(Int(n), CheckLoc{T}[])

function consider!(top::TopK{T}, loc::CheckLoc{T}) where {T}
    top.n == 0 && return top
    if length(top.items) < top.n
        push!(top.items, loc)
        return top
    end
    min_idx = 1
    min_abs = abs(top.items[1].value)
    for i in 2:length(top.items)
        v = abs(top.items[i].value)
        if v < min_abs
            min_abs = v
            min_idx = i
        end
    end
    if abs(loc.value) > min_abs
        top.items[min_idx] = loc
    end
    return top
end

merge_from!(top::TopK, other::TopK) = (foreach(l -> consider!(top, l), other.items); top)

sorted(top::TopK) = sort(top.items; by = l -> abs(l.value), rev = true)

"""Running statistics for one [`CheckClassResult`](@ref)."""
mutable struct Accum{T<:AbstractFloat}
    count::Int
    max_abs::T
    sum_sq::T
    n_points::Int
    const top::TopK{T}
end

Accum{T}(worst_n::Integer) where {T} = Accum{T}(0, zero(T), zero(T), 0, TopK{T}(worst_n))

"""
Record a point of a *violation* class (B, C, `cs2_out_of_range`).

Benign points contribute nothing but their weight to the rms denominator, so
the worst list is never padded with meaningless zero-metric entries.
"""
function add_violation!(acc::Accum{T}, metric, is_violation::Bool, loc) where {T}
    acc.n_points += 1
    is_violation || return acc
    acc.count += 1
    m = T(metric)
    acc.max_abs = max(acc.max_abs, abs(m))
    acc.sum_sq += m * m
    consider!(acc.top, loc)
    return acc
end

"""
Record a point of a *diagnostic* class (D, `cs2_vs_fd`).

The metric is meaningful everywhere, so `max`/`rms`/`worst` cover every
evaluated point and `is_violation` only feeds `count`.
"""
function add_diagnostic!(acc::Accum{T}, metric, is_violation::Bool, loc::CheckLoc{T}) where {T}
    acc.n_points += 1
    is_violation && (acc.count += 1)
    m = T(metric)
    acc.max_abs = max(acc.max_abs, abs(m))
    acc.sum_sq += m * m
    consider!(acc.top, loc)
    return acc
end

function finalize_class(acc::Accum{T}, name::AbstractString) where {T}
    rms = acc.n_points > 0 ? sqrt(acc.sum_sq / acc.n_points) : zero(T)
    return CheckClassResult{T}(String(name), acc.count, acc.max_abs, rms, sorted(acc.top))
end

"""
A class that could not be evaluated because a field it needs is absent.

`max`/`rms` are NaN, which `show` renders as "skipped" rather than as a
misleadingly clean zero.
"""
skipped_class(::Type{T}, name::AbstractString) where {T} =
    CheckClassResult{T}(String(name), 0, T(NaN), T(NaN), CheckLoc{T}[])

# ---------------------------------------------------------------------------
# Finite differences
# ---------------------------------------------------------------------------

"""
    fd3(f, x, j)

Three-point finite-difference derivative of `f(j)` with respect to `x[j]`:
central in the interior, one-sided (still second-order) at the two edges, and
valid for a non-uniformly spaced `x`. Requires `length(x) >= 3`.
"""
function fd3(f::F, x::AbstractVector{T}, j::Int) where {F,T}
    n = length(x)
    if j == 1
        h1 = x[2] - x[1]
        h2 = x[3] - x[2]
        return -((2h1 + h2) / (h1 * (h1 + h2))) * f(1) + ((h1 + h2) / (h1 * h2)) * f(2) -
               (h1 / (h2 * (h1 + h2))) * f(3)
    elseif j == n
        h1 = x[n - 1] - x[n - 2]
        h2 = x[n] - x[n - 1]
        return (h2 / (h1 * (h1 + h2))) * f(n - 2) - ((h1 + h2) / (h1 * h2)) * f(n - 1) +
               ((h1 + 2h2) / (h2 * (h1 + h2))) * f(n)
    end
    h1 = x[j] - x[j - 1]
    h2 = x[j + 1] - x[j]
    return -(h2 / (h1 * (h1 + h2))) * f(j - 1) + ((h2 - h1) / (h1 * h2)) * f(j) +
           (h1 / (h2 * (h1 + h2))) * f(j + 1)
end

# ---------------------------------------------------------------------------
# The individual classes
# ---------------------------------------------------------------------------

"""
Non-finite values in a field the pipeline does not *interpret* (anything other
than `logenergy`/`entropy`).

Reported rather than fatal: repair and the adapter never read those fields, and
a write-back passes them through byte-identically. This is not hypothetical --
the shipped stellarcollapse LS220 table carries Inf points in `cs2` and
`gamma`.
"""
function check_nonfinite_field(t::RawTable{T}, opts::CheckOptions, name::AbstractString) where {T}
    acc = Accum{T}(opts.worst_n)
    data = field(t, name)
    for I in CartesianIndices(data)
        v = data[I]
        if !isfinite(v)
            # Metric 1, not `v`: feeding Inf/NaN into max/rms would make the
            # class statistics themselves non-finite. The offending value is
            # still carried by the location's `value`.
            add_violation!(acc, one(T), true, make_loc(t, I[1], I[2], I[3], v))
        else
            add_violation!(acc, zero(T), false, nothing)
        end
    end
    return finalize_class(acc, "nonfinite_" * name)
end

# --- B. Range/positivity ---------------------------------------------------
#
# `ε + shift > 0` and `p > 0` are automatic once logenergy/logpress are finite
# (the structural pass already checked that): `10^finite` is strictly positive.
# So neither needs a class of its own.
function check_entropy_negative(t::RawTable{T}, opts::CheckOptions) where {T}
    acc = Accum{T}(opts.worst_n)
    entropy = field(t, "entropy")
    for I in CartesianIndices(entropy)
        s = entropy[I]
        if s < 0
            add_violation!(acc, s, true, make_loc(t, I[1], I[2], I[3], s))
        else
            add_violation!(acc, zero(T), false, nothing)
        end
    end
    return finalize_class(acc, "entropy_negative")
end

# --- C. Monotonicity in T --------------------------------------------------
#
# Per (iρ, kYₑ) column, count adjacent pairs with `entropy[j+1] <= entropy[j]`
# resp. `logenergy[j+1] <= logenergy[j]`. The metric is the (non-positive)
# difference, so the worst list holds the largest-magnitude drops.
#
# The columns are independent, which is where the C++ puts an OpenMP loop. Not
# threaded here: the per-thread accumulator merge order changes the summation
# order of `sum_sq` and hence the last bits of `rms`, so threading this is a
# deliberate decision rather than a free speedup.
function check_monotonicity_T(t::RawTable{T}, opts::CheckOptions) where {T}
    entropy = field(t, "entropy")
    logenergy = field(t, "logenergy")
    acc_s = Accum{T}(opts.worst_n)
    acc_e = Accum{T}(opts.worst_n)

    for kYₑ in 1:nYₑ(t), iρ in 1:nρ(t), jT in 1:(nT(t) - 1)
        ds = entropy[iρ, jT + 1, kYₑ] - entropy[iρ, jT, kYₑ]
        add_violation!(acc_s, ds, ds <= 0, make_loc(t, iρ, jT, kYₑ, ds))

        de = logenergy[iρ, jT + 1, kYₑ] - logenergy[iρ, jT, kYₑ]
        add_violation!(acc_e, de, de <= 0, make_loc(t, iρ, jT, kYₑ, de))
    end

    return finalize_class(acc_s, "entropy_nonmonotone_T"), finalize_class(acc_e, "logenergy_nonmonotone_T")
end

# --- D. Maxwell/thermodynamic-consistency diagnostics ----------------------
#
# Needs `logpress` and at least 3 points on each of the ρ/T axes (`fd3`'s
# requirement); the caller substitutes a skipped class when that is not met.
# Evaluated at *every* point -- the one-sided `fd3` formulas at the edges make
# that safe. Parallel over kYₑ in the C++; see `check_monotonicity_T` for why
# not here.
function check_consistency(t::RawTable{T}, opts::CheckOptions) where {T}
    logenergy = field(t, "logenergy")
    entropy = field(t, "entropy")
    logpress = field(t, "logpress")
    shift = T(energy_shift(t))
    m_B = T(opts.m_B_g)
    tol = T(opts.tol_consistency)
    tiny = tiny_denom(T)
    L10 = ln10(T)

    acc_T = Accum{T}(opts.worst_n)
    acc_p = Accum{T}(opts.worst_n)
    acc_sr = Accum{T}(opts.worst_n)

    for kYₑ in 1:nYₑ(t), jT in 1:nT(t), iρ in 1:nρ(t)
        temp = T(temperature(t, jT))
        ρ = T(density(t, iρ))
        p = exp10(logpress[iρ, jT, kYₑ])

        ε_T = jj -> exp10(logenergy[iρ, jj, kYₑ]) - shift
        s_T = jj -> entropy[iρ, jj, kYₑ]
        p_T = jj -> exp10(logpress[iρ, jj, kYₑ])
        ε_R = ii -> exp10(logenergy[ii, jT, kYₑ]) - shift
        s_R = ii -> entropy[ii, jT, kYₑ]

        dε_dT = fd3(ε_T, t.logT, jT) / (temp * L10)
        ds_dT = fd3(s_T, t.logT, jT) / (temp * L10)
        dp_dT = fd3(p_T, t.logT, jT) / (temp * L10)
        dε_dρ = fd3(ε_R, t.logρ, iρ) / (ρ * L10)
        ds_dρ = fd3(s_R, t.logρ, iρ) / (ρ * L10)

        kT_erg = temp * T(MEV_TO_ERG)

        δ_T = abs(dε_dT - (kT_erg / m_B) * ds_dT) / max(abs(dε_dT), tiny)
        δ_p = abs(ρ^2 * dε_dρ - (p - temp * dp_dT)) / p
        maxwell_s_ρ = abs(ds_dρ + dp_dT * m_B / (ρ^2 * T(MEV_TO_ERG))) / max(abs(ds_dρ), tiny)

        # A non-finite metric means a non-finite input reached the stencil
        # (possible in `logpress`, which is diagnostic-only and therefore not
        # structurally fatal); such points are skipped here and reported by the
        # corresponding `nonfinite_<field>` class instead.
        isfinite(δ_T) && add_diagnostic!(acc_T, δ_T, δ_T > tol, make_loc(t, iρ, jT, kYₑ, δ_T))
        isfinite(δ_p) && add_diagnostic!(acc_p, δ_p, δ_p > tol, make_loc(t, iρ, jT, kYₑ, δ_p))
        isfinite(maxwell_s_ρ) &&
            add_diagnostic!(acc_sr, maxwell_s_ρ, maxwell_s_ρ > tol, make_loc(t, iρ, jT, kYₑ, maxwell_s_ρ))
    end

    return finalize_class(acc_T, "delta_T"), finalize_class(acc_p, "delta_p"),
           finalize_class(acc_sr, "maxwell_s_rho")
end

# --- E. Sound speed (report-only diagnostic) -------------------------------
#
# Stored-cs2 unit conventions vary between table providers -- some store c_s²
# in units of c², some do not even claim causal normalization -- so `cs2_vs_fd`
# is a diagnostic comparison, never a pass/fail. `cs2_out_of_range` is the only
# cs2-based class that can ever fail a table.
function check_cs2_out_of_range(t::RawTable{T}, opts::CheckOptions) where {T}
    acc = Accum{T}(opts.worst_n)
    cs2 = field(t, "cs2")
    for I in CartesianIndices(cs2)
        v = cs2[I]
        if !isfinite(v)
            # Reported by `nonfinite_cs2`; counting Inf here too would
            # double-report it as a range violation.
            add_violation!(acc, zero(T), false, nothing)
            continue
        end
        metric = zero(T)
        bad = false
        if v <= 0
            metric = v
            bad = true
        elseif v >= 1
            metric = v - one(T)
            bad = true
        end
        if bad
            add_violation!(acc, metric, true, make_loc(t, I[1], I[2], I[3], metric))
        else
            add_violation!(acc, zero(T), false, nothing)
        end
    end
    return finalize_class(acc, "cs2_out_of_range")
end

function check_cs2_vs_fd(t::RawTable{T}, opts::CheckOptions) where {T}
    logenergy = field(t, "logenergy")
    entropy = field(t, "entropy")
    logpress = field(t, "logpress")
    cs2 = field(t, "cs2")
    shift = T(energy_shift(t))
    c = T(C_LIGHT_CM_S)
    tol = T(opts.tol_consistency)
    tiny = tiny_denom(T)
    L10 = ln10(T)

    acc = Accum{T}(opts.worst_n)

    for kYₑ in 1:nYₑ(t), jT in 1:nT(t), iρ in 1:nρ(t)
        ρ = T(density(t, iρ))
        temp = T(temperature(t, jT))

        s_T = jj -> entropy[iρ, jj, kYₑ]
        p_T = jj -> exp10(logpress[iρ, jj, kYₑ])
        s_R = ii -> entropy[ii, jT, kYₑ]
        p_R = ii -> exp10(logpress[ii, jT, kYₑ])

        p = exp10(logpress[iρ, jT, kYₑ])
        ε = exp10(logenergy[iρ, jT, kYₑ]) - shift

        ds_dT = fd3(s_T, t.logT, jT) / (temp * L10)
        dp_dT = fd3(p_T, t.logT, jT) / (temp * L10)
        ds_dρ = fd3(s_R, t.logρ, iρ) / (ρ * L10)
        dp_dρ = fd3(p_R, t.logρ, iρ) / (ρ * L10)

        h = one(T) + (ε + p / ρ) / c^2
        cs2_fd = (dp_dρ - dp_dT * (ds_dρ / ds_dT)) / (h * c^2)

        metric = abs(cs2[iρ, jT, kYₑ] - cs2_fd) / max(abs(cs2_fd), tiny)
        # Skip points poisoned by non-finite inputs (stored cs2 or logpress
        # feeding the stencil); those are reported by `nonfinite_<field>`.
        isfinite(metric) && add_diagnostic!(acc, metric, metric > tol, make_loc(t, iρ, jT, kYₑ, metric))
    end

    return finalize_class(acc, "cs2_vs_fd")
end

# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------

"""
    check_table(t, opts = CheckOptions()) -> CheckReport

Audit a [`RawTable`](@ref) and report what is wrong with it. Never throws:
structural problems that would make the later checks unsafe are caught and
reported as `fatal_messages` with `status = Status.fatal`, and the
non-structural classes are then skipped.

  * **A. Structural (fatal).** Axes strictly increasing and finite; required
    fields `logenergy`/`entropy` and attribute `energy_shift` present; those two
    fields finite everywhere.
  * **B. Range/positivity.** `entropy_negative`, plus a `nonfinite_<field>`
    class for each *non-interpreted* field that carries a non-finite value.
  * **C. Monotonicity in T.** `entropy_nonmonotone_T`, `logenergy_nonmonotone_T`.
  * **D. Maxwell consistency.** `delta_T`, `delta_p`, `maxwell_s_rho`; needs
    `logpress` and at least 3 ρ and T points, otherwise a single skipped
    `maxwell_consistency` placeholder.
  * **E. Sound speed.** `cs2_out_of_range` and, when D's inputs are also
    available, the report-only `cs2_vs_fd`; only when a `cs2` field exists.
"""
function check_table(t::RawTable{T}, opts::CheckOptions = CheckOptions()) where {T}
    fatal_messages = String[]
    classes = CheckClassResult{T}[]

    # --- A. Structural -----------------------------------------------------
    #
    # Every sub-check runs unconditionally -- no short-circuit on the first
    # failure -- so one call reports everything wrong with a badly broken
    # table. Only afterwards do we decide whether to continue to B-E.

    try
        validate_axes(t)
    catch e
        push!(fatal_messages, "axes: " * (e isa Exception ? sprint(showerror, e) : string(e)))
    end

    for required in ("logenergy", "entropy")
        has_field(t, required) || push!(fatal_messages, "missing required field '$required'")
    end
    has_attribute(t, "energy_shift") || push!(fatal_messages, "missing required attribute 'energy_shift'")

    # Finiteness is fatal only for the fields the pipeline interprets. Anything
    # else becomes a `nonfinite_<field>` class in section B.
    for name in ("logenergy", "entropy")
        has_field(t, name) || continue          # absence already reported above
        data = field(t, name)
        bad_count = 0
        first_bad = nothing
        for I in CartesianIndices(data)
            if !isfinite(data[I])
                bad_count += 1
                first_bad === nothing && (first_bad = I)
            end
        end
        if bad_count > 0
            iρ, jT, kYₑ = Tuple(first_bad)
            push!(fatal_messages,
                  "field '$name' has $bad_count non-finite value(s), first at iρ=$iρ jT=$jT kYₑ=$kYₑ " *
                  "(ρ=$(density(t, iρ)) T=$(temperature(t, jT)) Yₑ=$(electron_fraction(t, kYₑ)))")
        end
    end

    isempty(fatal_messages) || return CheckReport{T}(Status.fatal, fatal_messages, classes)

    # --- B. Range/positivity ------------------------------------------------
    push!(classes, check_entropy_negative(t, opts))

    # Only offending fields get a class, so a clean table is not padded with one
    # zero-count class per auxiliary field.
    for name in field_names(t)
        name in ("logenergy", "entropy") && continue    # fatal above
        r = check_nonfinite_field(t, opts, name)
        r.count > 0 && push!(classes, r)
    end

    # --- C. Monotonicity in T -----------------------------------------------
    mono_s, mono_e = check_monotonicity_T(t, opts)
    push!(classes, mono_s)
    push!(classes, mono_e)

    # --- D. Maxwell consistency ---------------------------------------------
    have_fd_inputs = has_field(t, "logpress") && nρ(t) >= 3 && nT(t) >= 3
    if have_fd_inputs
        δ_T, δ_p, maxwell_s_ρ = check_consistency(t, opts)
        push!(classes, δ_T)
        push!(classes, δ_p)
        push!(classes, maxwell_s_ρ)
    else
        push!(classes, skipped_class(T, "maxwell_consistency"))
    end

    # --- E. Sound speed -----------------------------------------------------
    if has_field(t, "cs2")
        push!(classes, check_cs2_out_of_range(t, opts))
        have_fd_inputs && push!(classes, check_cs2_vs_fd(t, opts))
    end

    return CheckReport{T}(Status.ok, fatal_messages, classes)
end

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

# `d.ddddeNN`, matching the C++ report's `scientific`/`setprecision(4)`. Hand
# rolled because `Printf` is a stdlib this package does not depend on, and
# `round(x; sigdigits = 4)` leaks the binary representation of the rounded
# value ("2.9980000000000002e29").
function sci4(x::Real)
    v = Float64(x)
    isfinite(v) || return string(v)
    iszero(v) && return "0.0000e0"
    a = abs(v)
    e = floor(Int, log10(a))
    m = round(Int, a / exp10(e) * 10^4)   # five digits: one before the point, four after
    if m >= 10^5                          # rounding carried into the next decade
        m = div(m, 10)
        e += 1
    elseif m < 10^4                       # exp10(e) overshot at an exact power of ten
        m = 10^4
    end
    ds = string(m)
    return string(signbit(v) ? "-" : "", ds[1], ".", ds[2:end], "e", e)
end

function Base.show(io::IO, ::MIME"text/plain", r::CheckReport)
    println(io, "check_table report: status = ", r.status)

    if !isempty(r.fatal_messages)
        println(io, "fatal structural problems:")
        for msg in r.fatal_messages
            println(io, "  - ", msg)
        end
    end

    for c in r.classes
        println(io)
        print(io, c.name, ": count=", c.count)
        if isnan(c.max)
            println(io, " (skipped: required field not present)")
            continue
        end
        println(io, " max=", sci4(c.max), " rms=", sci4(c.rms))
        isempty(c.worst) && continue
        println(io, "  worst offenders (i,j,k -> ρ [g/cm³], T [MeV], Yₑ : value):")
        for loc in c.worst
            println(io, "    (", lpad(loc.iρ, 4), ",", lpad(loc.jT, 4), ",", lpad(loc.kYₑ, 3),
                    ") -> ρ=", sci4(loc.ρ), " T=", sci4(loc.temp), " Yₑ=", sci4(loc.ye),
                    " : value=", sci4(loc.value))
        end
    end
    return nothing
end

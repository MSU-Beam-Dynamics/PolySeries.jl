using Test, PolySeries

# Independent reference for mul!: convolution of exponent vectors over dense
# coefficient arrays, truncated at the descriptor order. It never calls mul!,
# so a wrong product cannot agree with it by construction. findindex is used
# to rank the product monomial; it is validated separately in index_tests.jl.

ref_exponents(desc, i) = [Int(desc.polymap.map[i, v]) for v in 2:desc.nv + 1]

function reference_product(desc, probe::CTPS, a::AbstractVector, b::AbstractVector)
    pm = desc.polymap.map
    ref = zeros(promote_type(eltype(a), eltype(b)), desc.N)
    for i in 1:desc.N
        iszero(a[i]) && continue
        di = Int(pm[i, 1])
        ei = ref_exponents(desc, i)
        for j in 1:desc.N
            iszero(b[j]) && continue
            di + Int(pm[j, 1]) > desc.order && continue
            ref[findindex(probe, ei .+ ref_exponents(desc, j))] += a[i] * b[j]
        end
    end
    return ref
end

# Reads every coefficient through the mask-respecting accessor: an active but
# unwritten slot shows up as poison, an inactive slot as zero.
ref_coefficients(p::CTPS) = [element(p, ref_exponents(p.desc, i)) for i in 1:p.desc.N]

# Deterministic pseudo-random values; the test target has no Random dependency.
ref_value(::Type{Float64}, i, seed) = sin(1.7 * i + 0.37 * seed) + 0.25 * cos(3.1 * i * seed)
ref_value(::Type{Float32}, i, seed) = Float32(ref_value(Float64, i, seed))
ref_value(::Type{ComplexF64}, i, seed) =
    complex(ref_value(Float64, i, seed), ref_value(Float64, i + 1000, seed))

# A CTPS whose inactive degree blocks hold poison and whose active blocks hold
# deterministic values, together with the dense reference vector.
function sparse_fixture(::Type{T}, desc, active_degrees, seed; poison=T(NaN)) where T
    p = CTPS(T, desc)
    fill!(p.c, poison)
    values = zeros(T, desc.N)
    mask = UInt64(0)
    for d in active_degrees
        d <= desc.order || continue
        s = desc.off[d + 1]
        e = s + desc.Nd[d + 1] - 1
        for i in s:e
            values[i] = p.c[i] = ref_value(T, i, seed)
        end
        mask |= UInt64(1) << d
    end
    p.degree_mask[] = mask
    return p, values
end

# Degree patterns worth covering: dense, empty, constant-only, linear-only,
# parity-separated, top-only, and gapped runs.
function degree_patterns(order)
    patterns = Vector{Int}[collect(0:order), Int[], [0], [1], collect(0:2:order),
                           collect(1:2:order), [order]]
    order >= 2 && push!(patterns, [0, order])
    order >= 3 && push!(patterns, [1, order - 1], [0, 1, order])
    order >= 4 && push!(patterns, [2, 3])
    return unique(patterns)
end

reference_tolerance(::Type{Float64}, expected) = 1e-12 * max(1.0, sum(abs, expected))
reference_tolerance(::Type{ComplexF64}, expected) = 1e-12 * max(1.0, sum(abs, expected))
reference_tolerance(::Type{Float32}, expected) = 2e-5 * max(1.0, sum(abs, expected))

function check_products(::Type{T}, nv, order) where T
    desc = PSDesc(nv, order)
    probe = CTPS(T, desc)
    patterns = degree_patterns(order)
    for (ia, pa) in enumerate(patterns), (ib, pb) in enumerate(patterns)
        a, av = sparse_fixture(T, desc, pa, ia)
        b, bv = sparse_fixture(T, desc, pb, 100 + ib)
        expected = reference_product(desc, probe, av, bv)
        tol = reference_tolerance(T, expected)

        # Allocating product.
        @test isapprox(ref_coefficients(a * b), expected; atol=tol)

        # Fresh output whose storage is poisoned: mul! must initialize its band.
        r = CTPS(T, desc)
        fill!(r.c, T(NaN))
        mul!(r, a, b)
        @test isapprox(ref_coefficients(r), expected; atol=tol)

        # Output aliasing the first / second operand.
        a2 = CTPS(a); mul!(a2, a2, b)
        @test isapprox(ref_coefficients(a2), expected; atol=tol)
        b2 = CTPS(b); mul!(b2, a, b2)
        @test isapprox(ref_coefficients(b2), expected; atol=tol)

        # Output sharing the first operand's buffer through a different wrapper.
        a3 = CTPS(a)
        shared = CTPS{T}(a3.c, desc, Ref(a3.degree_mask[]))
        mul!(shared, a3, b)
        @test isapprox(ref_coefficients(shared), expected; atol=tol)

        # Square through a single buffer.
        sq = CTPS(a); mul!(sq, sq, sq)
        expected_sq = reference_product(desc, probe, av, av)
        @test isapprox(ref_coefficients(sq), expected_sq; atol=reference_tolerance(T, expected_sq))
    end
    return nothing
end

@testset "mul! against exponent-convolution reference" begin
    for (nv, order) in ((1, 0), (1, 5), (2, 1), (2, 5), (3, 4), (4, 5), (6, 4))
        @testset "Float64 nv=$nv order=$order" begin
            check_products(Float64, nv, order)
        end
    end
    @testset "ComplexF64 nv=2 order=4" begin
        check_products(ComplexF64, 2, 4)
    end
    @testset "Float32 nv=3 order=3" begin
        check_products(Float32, 3, 3)
    end
end

@testset "pow!/^ against repeated convolution" begin
    for (nv, order) in ((1, 6), (2, 5), (3, 4))
        desc = PSDesc(nv, order)
        probe = CTPS(Float64, desc)
        for (seed, pattern) in enumerate((collect(0:order), [1], [0, 1], [1, 2], [0, 2]))
            a, av = sparse_fixture(Float64, desc, pattern, seed)
            power = copy(av)
            for n in 2:7
                power = reference_product(desc, probe, power, av)
                tol = 1e-12 * max(1.0, sum(abs, power))
                @test isapprox(ref_coefficients(a^n), power; atol=tol)
                r = CTPS(Float64, desc)
                fill!(r.c, NaN)
                pow!(r, a, n)
                @test isapprox(ref_coefficients(r), power; atol=tol)
                aliased = CTPS(a)
                pow!(aliased, aliased, n)
                @test isapprox(ref_coefficients(aliased), power; atol=tol)
            end
        end
    end
end

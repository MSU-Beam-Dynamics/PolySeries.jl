# Construct the reference coefficients independently of CTPS arithmetic. Every
# inactive slot is initialized to poison, so failures never depend on the heap.
function masked_fixture(desc, mask, factor; poison=NaN)
    p = CTPS(Float64)
    fill!(p.c, poison)
    expected = zeros(desc.N)
    for i in 1:desc.N
        d = Int(desc.polymap.map[i, 1])
        if !iszero(mask & (UInt64(1) << d))
            p.c[i] = expected[i] = factor * i
        end
    end
    p.degree_mask[] = mask
    return p, expected
end

function masked_coefficients(p)
    [element(p, collect(Int, PolySeries.getindexmap(p.desc.polymap, i)))
     for i in 1:p.desc.N]
end

@testset "Separated degree blocks with initialized storage" begin
    set_descriptor!(1, 2)
    x = CTPS(0.0, 1)
    b = CTPS(1.0)
    b.c[3] = 1.0
    PolySeries.update_degree_mask!(b) # b = 1 + x², including a stored zero at x
    r = CTPS(x)
    addto!(r, b)
    @test masked_coefficients(r) == [1.0, 1.0, 1.0]
    r = CTPS(x)
    subfrom!(r, b)
    @test masked_coefficients(r) == [-1.0, 1.0, -1.0]
    r = CTPS(x)
    PolySeries._add_scaled!(r, b, 2.0)
    @test masked_coefficients(r) == [2.0, 1.0, 2.0]
end

@testset "Degree masks: poisoned inactive storage" begin
    # Exhaust every mask and pair of masks, including empty, contiguous,
    # alternating, interleaved and overlapping degree runs. nv=2 additionally
    # checks that each active degree is processed as a whole coefficient block.
    # NaN is the only poison worth using: a finite garbage value multiplied by
    # a zero coefficient would vanish, NaN propagates into the result.
    for nv in (1, 2), poison in (NaN,)
        desc = set_descriptor!(nv, 3)
        for ma in UInt64(0):UInt64(15)
            a, av = masked_fixture(desc, ma, 2.0; poison)
            @test masked_coefficients(CTPS(a)) == av
            @test masked_coefficients(-a) == -av
            @test masked_coefficients(3.0 * a) == 3av
            @test masked_coefficients(a / 2.0) == av / 2
            shifted = copy(av); shifted[1] += 3.0
            @test masked_coefficients(a + 3.0) == shifted
            shifted = -av; shifted[1] += 3.0
            @test masked_coefficients(3.0 - a) == shifted
            args = ntuple(_ -> 0.25, nv)
            expected_value = sum(av[i] * 0.25^Int(desc.polymap.map[i, 1]) for i in 1:desc.N)
            @test a(args...) == expected_value
            idx, pooled = PolySeries._ctps_pooled_copy(a, desc)
            try
                @test masked_coefficients(pooled) == av
                @test pooled(args...) == expected_value
            finally
                PolySeries._pool_release!(idx, pooled, desc)
            end
            for mb in UInt64(0):UInt64(15)
                b, bv = masked_fixture(desc, mb, -3.0; poison)
                @test masked_coefficients(a + b) == av + bv
                @test masked_coefficients(a - b) == av - bv
                # Exercise reuse with unrelated old destination masks and the
                # aliasing supported by these elementwise arithmetic kernels.
                for alias in (:neither, :left, :right)
                    for (op, expected) in ((add!, av + bv), (sub!, av - bv))
                        aa, _ = masked_fixture(desc, ma, 2.0; poison)
                        bb, _ = masked_fixture(desc, mb, -3.0; poison)
                        r = alias == :left ? aa : alias == :right ? bb : first(masked_fixture(desc, ~ma & 15, 7.0; poison))
                        op(r, aa, bb)
                        @test masked_coefficients(r) == expected
                    end
                    for (sa, sb) in ((2.0, -1.0), (0.0, 1.0), (1.0, 0.0))
                        aa, _ = masked_fixture(desc, ma, 2.0; poison)
                        bb, _ = masked_fixture(desc, mb, -3.0; poison)
                        r = alias == :left ? aa : alias == :right ? bb : first(masked_fixture(desc, ~ma & 15, 7.0; poison))
                        scaleadd!(r, sa, aa, sb, bb)
                        @test masked_coefficients(r) == sa * av + sb * bv
                    end
                end
                for (op, expected) in ((addto!, av + bv), (subfrom!, av - bv))
                    r, _ = masked_fixture(desc, ma, 2.0; poison)
                    op(r, b)
                    @test masked_coefficients(r) == expected
                end
                r, _ = masked_fixture(desc, ma, 2.0; poison)
                PolySeries._add_scaled!(r, b, 2.0)
                @test masked_coefficients(r) == av + 2bv
                r, _ = masked_fixture(desc, ma, 2.0; poison)
                copy!(r, b)
                @test masked_coefficients(r) == bv
                r, _ = masked_fixture(desc, ma, 2.0; poison)
                scale!(r, b, 2.0)
                @test masked_coefficients(r) == 2bv
            end
        end
    end
end

@testset "Degree runs at the UInt64 boundary" begin
    desc = set_descriptor!(1, 63)
    for mask in (UInt64(0), UInt64(1) << 63, typemax(UInt64),
                 UInt64(0x8000000000000001), UInt64(0xf00000000000000f),
                 UInt64(0xaaaaaaaaaaaaaaaa))
        a, av = masked_fixture(desc, mask, 2.0)
        b, bv = masked_fixture(desc, ~mask, -3.0)
        r = CTPS(Float64)
        add!(r, a, b)
        @test masked_coefficients(r) == av + bv
        addto!(a, b)
        @test masked_coefficients(a) == av + bv
    end
end


# Exponent-convolution reference for products of the masked fixtures above;
# independent of mul! (only findindex is used, to rank the product monomial).
function masked_product_reference(desc, probe, av, bv)
    pm = desc.polymap.map
    ref = zeros(desc.N)
    for i in 1:desc.N, j in 1:desc.N
        (iszero(av[i]) || iszero(bv[j])) && continue
        Int(pm[i, 1]) + Int(pm[j, 1]) > desc.order && continue
        k = findindex(probe, [Int(pm[i, v]) + Int(pm[j, v]) for v in 2:desc.nv + 1])
        ref[k] += av[i] * bv[j]
    end
    return ref
end

@testset "Degree masks: multiplication and powers with poisoned inactive storage" begin
    # Same exhaustive mask sweep as the elementwise kernels, now for mul! and
    # pow!: every (ma, mb) pair, every aliasing pattern, poisoned gaps. The
    # fixture values are small integers, so products and sums are exact.
    for nv in (1, 2), poison in (NaN,)
        desc = set_descriptor!(nv, 3)
        probe = CTPS(Float64)
        for ma in UInt64(0):UInt64(15)
            a, av = masked_fixture(desc, ma, 2.0; poison)

            power = copy(av)
            for n in 2:5
                power = masked_product_reference(desc, probe, power, av)
                @test masked_coefficients(a^n) == power
                r, _ = masked_fixture(desc, ~ma & 15, 7.0; poison)
                pow!(r, a, n)
                @test masked_coefficients(r) == power
                aa, _ = masked_fixture(desc, ma, 2.0; poison)
                pow!(aa, aa, n)
                @test masked_coefficients(aa) == power
            end

            aa, _ = masked_fixture(desc, ma, 2.0; poison)
            mul!(aa, aa, aa)
            @test masked_coefficients(aa) == masked_product_reference(desc, probe, av, av)

            for mb in UInt64(0):UInt64(15)
                b, bv = masked_fixture(desc, mb, -3.0; poison)
                expected = masked_product_reference(desc, probe, av, bv)
                @test masked_coefficients(a * b) == expected
                for alias in (:neither, :left, :right)
                    aa, _ = masked_fixture(desc, ma, 2.0; poison)
                    bb, _ = masked_fixture(desc, mb, -3.0; poison)
                    r = alias == :left ? aa : alias == :right ? bb : first(masked_fixture(desc, ~ma & 15, 7.0; poison))
                    mul!(r, aa, bb)
                    @test masked_coefficients(r) == expected
                end
            end
        end
    end
end

using Test, PolySeries

@testset "Active multiplication schedule selection" begin
    # Compare with an independent full-table filter, including both asymmetric
    # orientations, empty masks, gaps, and pairs truncated by the order.
    for order in 0:5
        desc = PSDesc(1, order)
        for last_degree in 0:order
            stop = desc.mul_offsets[last_degree + 1] + min(last_degree, order - last_degree)
            @test [(Int(s.di), Int(s.dj)) for s in
                   PolySeries.PrefixMulSchedules(desc.mul, stop)] ==
                  [(Int(s.di), Int(s.dj)) for s in desc.mul if s.di <= last_degree]
        end
        for a in UInt64(0):(UInt64(1) << (order + 1))-1,
            b in UInt64(0):(UInt64(1) << (order + 1))-1
            expected = [(Int(s.di), Int(s.dj)) for s in desc.mul if
                ((a >> s.di) & 1 != 0 && (b >> s.dj) & 1 != 0) ||
                ((b >> s.di) & 1 != 0 && (a >> s.dj) & 1 != 0)]
            actual = [(Int(s.di), Int(s.dj))
                      for s in PolySeries.ActiveMulSchedules(desc, a, b)]
            @test actual == expected
        end
    end
    desc = PSDesc(2, 12)
    @test length(desc.mul) == 49
    @test [(Int(s.di), Int(s.dj)) for s in
           PolySeries.ActiveMulSchedules(desc, UInt64(3), UInt64(3))] ==
          [(0, 0), (1, 0), (1, 1)]

    desc = PSDesc(1, 63)
    high = UInt64(1) << 63
    for (a, b, expected) in ((high, UInt64(1), [(63, 0)]),
                              (UInt64(1), high, [(63, 0)]),
                              (high, high, Tuple{Int,Int}[]),
                              (high | 1, high | 1, [(0, 0), (63, 0)]))
        @test [(Int(s.di), Int(s.dj)) for s in
               PolySeries.ActiveMulSchedules(desc, a, b)] == expected
    end
end

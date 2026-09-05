using Test, PolySeries, Enzyme

# Keep the callable and scalar argument types unchanged: Enzyme must reuse its
# compiled derivative after the registry publishes a larger snapshot.
_enzyme_switch_exp(a) = cst(exp(CTPS(a, 1)))

@testset "Compiled Enzyme derivatives follow descriptor registry updates" begin
    original = set_descriptor!(1, 3)
    # Pick genuinely new descriptors even when earlier tests populated caches.
    candidates = [(2, 4); [(nv, 2) for nv in 2:20]]
    fresh = filter(shape -> !haskey(PolySeries.DESC_CACHE, shape), candidates)[1:2]
    for a in (0.0, 0.5)
        @test Enzyme.gradient(Reverse, _enzyme_switch_exp, a)[1] ≈ exp(a)
        @test Enzyme.autodiff(Forward, _enzyme_switch_exp, Duplicated(a, 1.0))[1] ≈ exp(a)
    end
    for (nv, order) in fresh
        desc = set_descriptor!(nv, order)
        @test desc !== original
        for a in (0.0, 0.5)
            @test Enzyme.gradient(Reverse, _enzyme_switch_exp, a)[1] ≈ exp(a)
            @test Enzyme.autodiff(Forward, _enzyme_switch_exp, Duplicated(a, 1.0))[1] ≈ exp(a)
        end
    end
    set_descriptor!(original.nv, original.order)
    @test Enzyme.gradient(Reverse, _enzyme_switch_exp, 0.5)[1] ≈ exp(0.5)
end

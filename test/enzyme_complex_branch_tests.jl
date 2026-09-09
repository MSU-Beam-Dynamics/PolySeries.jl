using Test, PolySeries, Enzyme

@testset "Enzyme sensitivities along complex branch cuts" begin
    desc = PSDesc(1, 3)
    for center in (-2.0, 2.0), side in (0.0, -0.0), degree in (0, 1)
        z = complex(center, side)
        root = cos(asin(z))
        expected = imag(degree == 0 ? inv(root) : z/root^3)
        for (f, sign) in ((asin, 1), (acos, -1))
            coefficient = a -> imag(element(f(CTPS(complex(a, side), 1, desc)), [degree]))
            @test Enzyme.autodiff(Forward, coefficient, Duplicated(center, 1.0))[1] ≈ sign*expected
            @test Enzyme.gradient(Reverse, coefficient, center)[1] ≈ sign*expected
        end
    end
end

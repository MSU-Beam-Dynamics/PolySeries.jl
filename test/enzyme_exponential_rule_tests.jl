using Test,PolySeries,Enzyme

function exponential_coefficient(p)
    y = exp(p)
    return element(y,[4])+2cst(y)
end
function exponential_alias(p)
    exp!(p,p)
    return element(p,[4])+2cst(p)
end
function exponential_overwrite(p)
    y = exp(p)
    result = element(y,[4])+2cst(y)
    PolySeries.zero!(p)
    PolySeries.zero!(y)
    return result
end

@testset "Mathematical exponential AD rule" begin
    for T in (Float32,Float64), center in (T(-3),zero(T),T(0.5)), nonlinear in (false,true)
        d = PSDesc(1,4)
        coefficients = nonlinear ? T[center,0.2,0,-0.1,0.03] : T[center,0,0,0,0]
        a,b,c,e,f = BigFloat.(coefficients)
        scalar = exp(a)
        reference = scalar .* [one(a),b,c+b^2/2,e+b*c+b^3/6,
                                f+b*e+c^2/2+b^2*c/2+b^4/24]
        expected = T[reference[5]+2reference[1],reference[4],reference[3],reference[2],reference[1]]
        p = CTPS(T,d)
        p.c .= coefficients
        PolySeries.update_degree_mask!(p)
        for degree in 0:4
            (p.degree_mask[] >> degree)&1 == 0 && (p.c[degree+1] = T(NaN))
        end
        @test Enzyme.gradient(Reverse,exponential_coefficient,p)[1].c ≈ expected rtol=100eps(T)
        for fn in (exponential_alias,exponential_overwrite)
            primal,shadow = CTPS(p),Enzyme.make_zero(p)
            Enzyme.autodiff(Reverse,fn,Active,Duplicated(primal,shadow))
            @test shadow.c ≈ expected rtol=100eps(T)
        end
        tangent = Enzyme.make_zero(p)
        tangent.c .= T[1,2,3,4,5]
        expected_tangent = sum(expected.*tangent.c)
        @test Enzyme.autodiff(Forward,exponential_coefficient,Duplicated(p,tangent))[1] ≈ expected_tangent rtol=100eps(T)
        second = Enzyme.make_zero(p)
        second.c .= 2tangent.c
        batch = Enzyme.autodiff(Forward,exponential_coefficient,BatchDuplicated(p,(tangent,second)))[1]
        @test batch[1] ≈ expected_tangent rtol=100eps(T)
        @test batch[2] ≈ 2expected_tangent rtol=100eps(T)
        fill!(tangent.c,zero(T)); fill!(second.c,zero(T))
        arg = BatchDuplicated(p,(tangent,second))
        forward,backward = Enzyme.autodiff_thunk(ReverseSplitWithPrimal,
            Const{typeof(exponential_coefficient)},Active,typeof(arg))
        tape,_,_ = forward(Const(exponential_coefficient),arg)
        backward(Const(exponential_coefficient),arg,(one(T),T(2)),tape)
        @test tangent.c ≈ expected rtol=100eps(T)
        @test second.c ≈ 2expected rtol=100eps(T)
    end
end

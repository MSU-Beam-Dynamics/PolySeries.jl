using Test, PolySeries, Enzyme

product_coefficient(a,b) = element(a*b,[2]) + 2cst(a*b)
product_square(a) = element(a*a,[2]) + cst(a*a)
function product_then_overwrite(a,b)
    y = a*b
    PolySeries.zero!(a)
    PolySeries.zero!(b)
    return element(y,[2]) + 2cst(y)
end

function selected_parameter_loss(parameters,desc)
    a = sum(parameters)/length(parameters)
    b = sum(abs2,parameters)/length(parameters)
    x = CTPS(0.0,1,desc)
    y = exp(a*x)*(1+b*x^2)
    return element(y,[1]) + 2element(y,[3])
end

function product_weighted(a,b,weights)
    y = a*b
    value = zero(eltype(a.c))
    for i in eachindex(weights)
        value += weights[i]*PolySeries._coefficient(y.c,y.degree_mask[],i,Int(y.desc.polymap.map[i,1]))
    end
    return value
end

@testset "Multivariate convolution adjoint reference" begin
    for T in (Float32,Float64), nv in (2,3)
        d = PSDesc(nv,4)
        a,b = CTPS(T,d),CTPS(T,d)
        a.degree_mask[] = UInt64(5) # Degrees 0,2; other storage is poisoned.
        b.degree_mask[] = UInt64(10) # Degrees 1,3.
        av,bv = zeros(T,d.N),zeros(T,d.N)
        for i in 1:d.N
            degree = Int(d.polymap.map[i,1])
            av[i] = (a.degree_mask[] >> degree)&1 != 0 ? T(i)/T(100) : zero(T)
            bv[i] = (b.degree_mask[] >> degree)&1 != 0 ? T((-1)^i)/T(i+1) : zero(T)
            a.c[i] = (a.degree_mask[] >> degree)&1 != 0 ? av[i] : T(NaN)
            b.c[i] = (b.degree_mask[] >> degree)&1 != 0 ? bv[i] : T(NaN)
        end
        weights = T[(-1)^i/(i+1) for i in 1:d.N]
        expected_a,expected_b = zeros(T,d.N),zeros(T,d.N)
        for i in 1:d.N, j in 1:d.N
            Int(d.polymap.map[i,1])+Int(d.polymap.map[j,1]) <= d.order || continue
            ex = Int.(d.polymap.map[i,2:end]).+Int.(d.polymap.map[j,2:end])
            k = findindex(a,ex)
            expected_a[i] += weights[k]*bv[j]
            expected_b[j] += weights[k]*av[i]
        end
        da,db = Enzyme.make_zero(a),Enzyme.make_zero(b)
        Enzyme.autodiff(Reverse,product_weighted,Active,Duplicated(a,da),Duplicated(b,db),Const(weights))
        @test da.c ≈ expected_a rtol=100eps(T)
        @test db.c ≈ expected_b rtol=100eps(T)
        fill!(db.c,zero(T))
        Enzyme.autodiff(Reverse,product_weighted,Active,Const(a),Duplicated(b,db),Const(weights))
        @test db.c ≈ expected_b rtol=100eps(T)
        fill!(da.c,zero(T))
        Enzyme.autodiff(Reverse,product_weighted,Active,Duplicated(a,da),Const(b),Const(weights))
        @test da.c ≈ expected_a rtol=100eps(T)
    end
end

@testset "Selected-coefficient gradients with many parameters" begin
    d = PSDesc(1,6)
    loss = p -> selected_parameter_loss(p,d)
    for n in (3,100,1000)
        p = collect(range(0.0,0.1;length=n))
        a,b = sum(p)/n,sum(abs2,p)/n
        expected = [(1+a^2+2b+4a*pi)/n for pi in p]
        @test Enzyme.gradient(Reverse,loss,p)[1] ≈ expected rtol=1e-12
        fill!(p,0)
        @test Enzyme.gradient(Reverse,loss,p)[1] ≈ fill(1/n,n) rtol=1e-12
    end
end
function product_alias(a,b)
    mul!(a,a,b)
    return element(a,[2]) + 2cst(a)
end
function product_overwrite_left(a,b)
    y = a*b
    PolySeries.zero!(a)
    return element(y,[2]) + 2cst(y)
end
function product_overwrite_right(a,b)
    y = a*b
    PolySeries.zero!(b)
    return element(y,[2]) + 2cst(y)
end

@testset "Mathematical coefficient convolution AD rules" begin
    for T in (Float32,Float64), center in (zero(T),T(0.4))
        d = PSDesc(1,3)
        a = CTPS(center,d)
        b = CTPS(T(2),1,d)
        # a has no nonconstant values, but all coefficient directions matter.
        expected_a = T[4,1,2,0]
        expected_b = T[2center,0,center,0]
        gradients = Enzyme.gradient(Reverse,product_coefficient,a,b)
        @test gradients[1].c ≈ expected_a
        @test gradients[2].c ≈ expected_b
        for fn in (product_then_overwrite,product_alias,product_overwrite_left,product_overwrite_right)
            da,db = Enzyme.make_zero(a),Enzyme.make_zero(b)
            Enzyme.autodiff(Reverse,fn,Active,Duplicated(CTPS(a),da),Duplicated(CTPS(b),db))
            @test da.c ≈ expected_a
            @test db.c ≈ expected_b
        end
        da,db = Enzyme.make_zero(a),Enzyme.make_zero(b)
        da.c .= T[1,2,3,4]
        db.c .= T[4,3,2,1]
        tangent = sum(expected_a.*da.c)+sum(expected_b.*db.c)
        @test Enzyme.autodiff(Forward,product_coefficient,Duplicated(a,da),Duplicated(b,db))[1] ≈ tangent
        @test Enzyme.autodiff(Forward,product_coefficient,Duplicated(a,da),Const(b))[1] ≈ sum(expected_a.*da.c)
        @test Enzyme.autodiff(Forward,product_coefficient,Const(a),Duplicated(b,db))[1] ≈ sum(expected_b.*db.c)
        # Batch lanes have separate seeds and accumulation buffers.
        da2,db2 = Enzyme.make_zero(a),Enzyme.make_zero(b)
        da2.c .= 2da.c
        db2.c .= 2db.c
        batch = Enzyme.autodiff(Forward,product_coefficient,
            BatchDuplicated(a,(da,da2)),BatchDuplicated(b,(db,db2)))[1]
        @test batch[1] ≈ tangent
        @test batch[2] ≈ 2tangent
        for seed in (da,db,da2,db2)
            fill!(seed.c,zero(T))
        end
        ba,bb = BatchDuplicated(a,(da,da2)),BatchDuplicated(b,(db,db2))
        # Split reverse mode accepts an explicit return seed per batch lane.
        # The combined scalar-return convenience API supplies only one seed.
        forward,backward = Enzyme.autodiff_thunk(ReverseSplitWithPrimal,
            Const{typeof(product_coefficient)},Active,typeof(ba),typeof(bb))
        tape,_,_ = forward(Const(product_coefficient),ba,bb)
        backward(Const(product_coefficient),ba,bb,(one(T),T(2)),tape)
        @test da.c ≈ expected_a
        @test da2.c ≈ 2expected_a
        @test db.c ≈ expected_b
        @test db2.c ≈ 2expected_b
        @test Enzyme.gradient(Reverse,product_square,b)[1].c ≈ T[4,2,4,0]
    end
end

@testset "Composition strategy selection preserves zero AD directions" begin
    d = PSDesc(1,3)
    x = CTPS(0.0,1,d)
    fixed = CompositionPlan(x^2 + 2x)
    ordinary(a) = element(compose(x^2+2x,[CTPS(a,1,d)]),[1])
    planned(a) = element(compose(fixed,[CTPS(a,1,d)]),[1])
    for f in (ordinary,planned), a in (0.0,0.3)
        # The closure's source polynomial/plan is fixed; only a is active.
        @test Enzyme.autodiff(Forward,Const(f),Duplicated(a,1.0))[1] ≈ 2
        @test Enzyme.gradient(Reverse,Const(f),a)[1] ≈ 2
    end
    coefficient = p -> element(compose(fixed,[p]),[3])
    p = CTPS(0.0,d)
    fill!(p.c,NaN)
    @test Enzyme.gradient(Reverse,Const(coefficient),p)[1].c ≈ [0,0,0,2]
end

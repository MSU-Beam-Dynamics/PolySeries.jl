using Test, PolySeries, Enzyme

function output_product_loss(a,b)
    y=a*b
    2cst(y)+element(y,[3,2])
end
function output_product_alias_loss(a,b)
    mul!(a,a,b)
    2cst(a)+element(a,[3,2])
end

@testset "Output product AD preserves every coefficient direction" begin
    d=PSDesc(2,6)
    for T in (Float32,Float64), fixture in (:dense,:zero,:gaps)
        a,b=[CTPS(T,d) for _ in 1:2]
        a.c .= T.(sin.(1:d.N)./10);b.c .= T.(cos.(1:d.N)./10)
        if fixture==:zero
            fill!(a.c,zero(T));fill!(b.c,zero(T))
        elseif fixture==:gaps
            for i in 1:d.N
                isodd(d.polymap.map[i,1]) && (a.c[i]=b.c[i]=zero(T))
            end
        end
        PolySeries.update_degree_mask!(a);PolySeries.update_degree_mask!(b)
        expected_a=zeros(T,d.N);expected_b=zeros(T,d.N)
        target=[3,2]
        for i in 1:d.N
            ex=Int.(d.polymap.map[i,2:end])
            if all(ex .<= target)
                expected_a[i]=element(b,target.-ex)
                expected_b[i]=element(a,target.-ex)
            end
        end
        expected_a[1]+=2cst(b);expected_b[1]+=2cst(a)
        for p in (a,b), i in 1:d.N
            p.degree_mask[] >> d.polymap.map[i,1] & 1 == 0 && (p.c[i]=T(NaN))
        end
        grad=Enzyme.gradient(Reverse,output_product_loss,a,b)
        @test grad[1].c ≈ expected_a
        @test grad[2].c ≈ expected_b
        da,db=Enzyme.make_zero(a),Enzyme.make_zero(b)
        Enzyme.autodiff(Reverse,output_product_alias_loss,Active,Duplicated(CTPS(a),da),Duplicated(CTPS(b),db))
        @test da.c ≈ expected_a
        @test db.c ≈ expected_b
        da.c .= T.(cos.(1:d.N));db.c .= T.(sin.(1:d.N))
        expected=sum(da.c.*expected_a)+sum(db.c.*expected_b)
        actual=Enzyme.autodiff(Forward,output_product_loss,Duplicated(a,da),Duplicated(b,db))[1]
        @test actual ≈ expected rtol=100eps(T)
    end
end

output_descriptor_switch(a)=cst(CTPS(a,1)*CTPS(2.0,1))
@testset "Compiled derivatives across descriptors with output plans" begin
    for (nv,n) in ((1,6),(2,12),(1,20),(2,6))
        set_descriptor!(nv,n)
        @test Enzyme.gradient(Reverse,output_descriptor_switch,0.0)[1] ≈ 2
        @test Enzyme.autodiff(Forward,output_descriptor_switch,Duplicated(0.0,1.0))[1] ≈ 2
    end
    clear_descriptor!()
end

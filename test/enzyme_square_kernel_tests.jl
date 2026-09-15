using Test, PolySeries, Enzyme

function square_selected(p)
    y=p*p
    cst(y)+element(y,[2,1,0,0])
end
function square_selected_alias(p)
    mul!(p,p,p)
    cst(p)+element(p,[2,1,0,0])
end

@testset "Squaring AD with independent coefficient tangents" begin
    d=PSDesc(4,6)
    target=[2,1,0,0]
    for T in (Float32,Float64), mask in (UInt64(0),UInt64(5),UInt64(0x7f))
        p=CTPS(T,d)
        p.degree_mask[]=mask
        for i in 1:d.N
            active=(mask >> d.polymap.map[i,1])&1 != 0
            p.c[i]=active ? T(sin(i)/10) : T(NaN)
        end
        expected=zeros(T,d.N)
        for i in 1:d.N
            ei=Int.(d.polymap.map[i,2:end])
            if all(ei .<= target)
                expected[i]=2element(p,target.-ei)
            end
        end
        expected[1]+=2cst(p)
        @test Enzyme.gradient(Reverse,square_selected,p)[1].c ≈ expected
        da=Enzyme.make_zero(p)
        Enzyme.autodiff(Reverse,square_selected_alias,Active,Duplicated(CTPS(p),da))
        @test da.c ≈ expected
        tangent=Enzyme.make_zero(p)
        tangent.c .= T[cos(i) for i in 1:d.N]
        ref=sum(expected.*tangent.c)
        @test Enzyme.autodiff(Forward,square_selected,Duplicated(p,tangent))[1] ≈ ref
        tangent2=Enzyme.make_zero(p)
        tangent2.c .= 2tangent.c
        batch=Enzyme.autodiff(Forward,square_selected,BatchDuplicated(p,(tangent,tangent2)))[1]
        @test batch[1] ≈ ref
        @test batch[2] ≈ 2ref
    end
end

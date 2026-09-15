using Test, PolySeries

square_coefficients(p) = [element(p,Int.(p.desc.polymap.map[i,2:end])) for i in 1:p.desc.N]
function square_reference(p,q=p)
    d=p.desc
    result=zeros(eltype(p.c),d.N)
    for i in 1:d.N, j in 1:d.N
        d.polymap.map[i,1]+Int(d.polymap.map[j,1]) <= d.order || continue
        ei,ej=Int.(d.polymap.map[i,2:end]),Int.(d.polymap.map[j,2:end])
        result[findindex(p,ei.+ej)] += element(p,ei)*element(q,ej)
    end
    result
end

@testset "Dedicated squaring coefficient references" begin
    for T in (Float32,Float64,ComplexF64,BigFloat), nv in (1,4), order in (0,1,6)
        d=PSDesc(nv,order)
        full=typemax(UInt64) >> (63-order)
        for mask in (UInt64(0),UInt64(1),full,full & UInt64(0x25))
            p=CTPS(T,d)
            p.degree_mask[]=mask
            for i in 1:d.N
                active=(mask >> d.polymap.map[i,1])&1 != 0
                p.c[i]=active ? T((-1)^i)/T(i+1) : T(NaN)
            end
            expected=square_reference(p)
            tolerance=100eps(real(one(T)))
            @test square_coefficients(p*p) ≈ expected rtol=tolerance
            q=CTPS(p)
            mul!(q,q,q)
            @test square_coefficients(q) ≈ expected rtol=tolerance
            # Two wrappers can share a vector but have distinct mask objects.
            wrapper=CTPS{T}(p.c,d,Ref(mask))
            @test square_coefficients(p*wrapper) ≈ expected rtol=tolerance
            @test square_coefficients(p*CTPS(p)) ≈ expected rtol=tolerance
        end
    end
    d=PSDesc(1,63)
    x=CTPS(0.0,1,d)
    p=1+x+x^31+x^32+x^63
    @test square_coefficients(p*p) == square_reference(p)
    @test element(p*p,[63]) == 4
end

function square_allocation_checks()
    d=PSDesc(4,6)
    p=CTPS(0.2,1,d)+CTPS(0.3,2,d)^3
    out=CTPS(Float64,d)
    mul!(out,p,p)
    @test @allocated(mul!(out,p,p)) == 0
    mul!(out,out,out)
    @test @allocated(mul!(out,out,out)) == 0
end
@testset "Squaring allocations" begin
    square_allocation_checks()
end

@testset "Output-degree blocks against selected BigFloat coefficients" begin
    # Degree six has 462 monomials here: these schedules cross the 256-column
    # block boundary, unlike the smaller coefficient-reference cases above.
    d=PSDesc(6,12)
    for T in (Float32,Float64)
        a,b,out=[CTPS(T,d) for _ in 1:3]
        a.c .= T[sin(i)/10 for i in 1:d.N]
        b.c .= T[cos(i)/10 for i in 1:d.N]
        PolySeries.update_degree_mask!(a);PolySeries.update_degree_mask!(b)
        for rhs in (a,b)
            mul!(out,a,rhs)
            if rhs === a
                baseline=copy(out.c)
                for schedules in (d.mul,sort(d.mul;by=s -> Int(s.di)+Int(s.dj))), block in (64,256)
                    fill!(out.c,zero(T))
                    PolySeries._square_schedules!(out.c,a.c,schedules,Val(block))
                    @test out.c ≈ baseline rtol=200eps(T)
                end
            end
            for k in unique([1; d.off; round.(Int,range(d.off[end],d.N;length=9))])
                exponent=Int.(d.polymap.map[k,2:end])
                reference=BigFloat(0)
                absolute_sum=BigFloat(0)
                for i in 1:d.N
                    ei=Int.(d.polymap.map[i,2:end])
                    all(ei .<= exponent) || continue
                    j=findindex(a,exponent.-ei)
                    term=BigFloat(a.c[i])*BigFloat(rhs.c[j])
                    reference+=term
                    absolute_sum+=abs(term)
                end
                @test abs(BigFloat(out.c[k])-reference) <= 200eps(T)*absolute_sum
            end
        end
    end
end

@testset "Squaring doubles products without premature overflow" begin
    for T in (Float32,Float64)
        d=PSDesc(4,6)
        large=floatmax(T)*T(0.75)
        small=inv(large)
        x=CTPS(zero(T),1,d)
        p=CTPS(large,d)+small*x+small*x^2
        @test element(p*p,[1,0,0,0]) ≈ 2*(large*small)
    end
end

@testset "Shared coefficients with different masks are not a square" begin
    d=PSDesc(4,6)
    a=CTPS(Float64,d)
    a.c .= [sin(i)/10 for i in 1:d.N]
    PolySeries.update_degree_mask!(a)
    b=CTPS{Float64}(a.c,d,Ref(UInt64(5)))
    expected=square_reference(a,b)
    @test square_coefficients(a*b) ≈ expected
    mul!(a,a,b)
    @test square_coefficients(a) ≈ expected
end

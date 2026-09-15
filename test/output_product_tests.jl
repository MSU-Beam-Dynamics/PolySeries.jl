using Test, PolySeries

output_coefficients(p)=[element(p,Int.(p.desc.polymap.map[i,2:end])) for i in 1:p.desc.N]
function output_reference(a,b)
    d=a.desc
    ref=zeros(BigFloat,d.N)
    absolute=zeros(BigFloat,d.N)
    for i in 1:d.N, j in 1:d.N
        Int(d.polymap.map[i,1])+Int(d.polymap.map[j,1]) <= d.order || continue
        ei,ej=Int.(d.polymap.map[i,2:end]),Int.(d.polymap.map[j,2:end])
        k=findindex(a,ei.+ej)
        v=BigFloat(element(a,ei))*BigFloat(element(b,ej))
        ref[k]+=v
        absolute[k]+=abs(v)
    end
    ref,absolute
end

@testset "Output coefficient product plans" begin
    for (nv,n,enabled) in ((1,5,false),(1,6,true),(1,63,true),(2,5,false),
                            (2,6,true),(2,20,true),(2,21,false),(3,6,false),(6,8,false))
        d=PSDesc(nv,n)
        @test (d.output_product !== nothing)==enabled
    end
    for T in (Float32,Float64), (nv,n) in ((1,6),(1,63),(2,6),(2,12),(2,20))
        d=PSDesc(nv,n)
        a,b,out=[CTPS(T,d) for _ in 1:3]
        a.c .= T.(sin.(1:d.N)./10)
        b.c .= T.(cos.(1:d.N)./10)
        PolySeries.update_degree_mask!(a);PolySeries.update_degree_mask!(b)
        @test PolySeries._output_product_dense(a.c,b.c)
        reference,absolute=output_reference(a,b)
        function check(p)
            actual=BigFloat.(output_coefficients(p))
            @test all(abs.(actual.-reference) .<= 200eps(T).*absolute)
        end
        mul!(out,a,b);check(out)
        q=CTPS(a);mul!(q,q,b);check(q)
        q=CTPS(b);mul!(q,a,q);check(q)
        check(a*b)
        for I in (UInt16,Int32)
            plan=PolySeries.build_output_product_plan(d.N,d.mul,I)
            @test plan.offsets[end]-1==length(plan.left)==length(plan.right)
            @test sum(!iszero,plan.diagonal)+2length(plan.left)==
                sum(Int(s.Ni)*Int(s.Nj)*(s.di==s.dj ? 1 : 2) for s in d.mul)
            fill!(out.c,T(NaN))
            PolySeries._output_product!(out.c,a.c,b.c,plan)
            check(out)
        end
        # Conservative full masks may contain many numerical zero entries.
        fill!(a.c,zero(T));fill!(b.c,zero(T));a.c[end]=one(T);b.c[1]=one(T)
        @test !PolySeries._output_product_dense(a.c,b.c)
        mul!(out,a,b)
        @test out.c==a.c
        # A missing degree can contain poison and must use mask-safe kernels.
        a.degree_mask[]=UInt64(1)<<n
        fill!(a.c,T(NaN));a.c[d.off[end]:end].=one(T)
        b=CTPS(one(T),1,d)
        mul!(out,a,b)
        @test output_coefficients(out)==output_coefficients(a)
    end
    @test_throws ArgumentError PolySeries.build_output_product_plan(65536,PolySeries.MulSchedule2D[],UInt16)
    # The generic builder can represent indices beyond UInt16 for experiments.
    s=PolySeries.MulSchedule2D(reshape(Int32[70000],1,1),Int32(40000),Int32(30000),
                             Int32(70000),Int32(1),Int32(1),UInt8(2),UInt8(1))
    p=PolySeries.build_output_product_plan(70000,[s],Int32)
    @test p.left==[40000] && p.right==[30000] && p.offsets[end]==2
end

function output_allocation_checks()
    d=PSDesc(2,12)
    a,b,out=[CTPS(Float64,d) for _ in 1:3]
    fill!(a.c,0.1);fill!(b.c,0.2)
    PolySeries.update_degree_mask!(a);PolySeries.update_degree_mask!(b)
    mul!(out,a,b)
    @test @allocated(mul!(out,a,b))==0
    mul!(out,out,b)
    @test @allocated(mul!(out,out,b))==0
    @test @inferred(mul!(out,a,b)) === out
end
@testset "Output coefficient product allocations" begin
    output_allocation_checks()
end

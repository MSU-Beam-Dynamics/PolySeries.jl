using Test, PolySeries

# A coefficient-space oracle independent of either multiplication strategy.
function strategy_product_reference(a,b)
    d = a.desc
    out = CTPS(eltype(a.c),d)
    fill!(out.c,zero(eltype(a.c)))
    for i in 1:d.N, j in 1:d.N
        Int(d.polymap.map[i,1])+Int(d.polymap.map[j,1]) <= d.order || continue
        ei,ej = Int.(d.polymap.map[i,2:end]), Int.(d.polymap.map[j,2:end])
        out.c[findindex(out,ei.+ej)] += element(a,ei)*element(b,ej)
    end
    PolySeries.update_degree_mask!(out)
    out
end
strategy_coeffs(p) = [element(p,Int.(p.desc.polymap.map[i,2:end])) for i in 1:p.desc.N]

@testset "Affine multiplication strategies" begin
    for T in (Float32,Float64,ComplexF64,BigFloat), nv in (1,3)
        d = PSDesc(nv,5)
        x = [CTPS(zero(T),i,d) for i in 1:nv]
        for a in (T(0.3)+x[end], x[1]-x[end], T(0.2)+sum(x),
                  T(0.1)+x[end]+T(0.02)*x[1]^2, x[1]^2),
            b in (x[1]^2+x[end]^5, one(T)+x[1]^3)
            expected = strategy_product_reference(a,b)
            # Poison every inactive degree before testing either orientation.
            for p in (a,b), degree in 0:d.order
                p.degree_mask[] >> degree & 1 != 0 && continue
                s = d.off[degree+1]
                p.c[s:s+d.Nd[degree+1]-1] .= T(NaN)
            end
            for (left,right) in ((a,b),(b,a))
                actual = left*right
                @test strategy_coeffs(actual) ≈ strategy_coeffs(expected) rtol=20eps(real(one(T)))
                dest = CTPS(left)
                mul!(dest,dest,right)
                @test strategy_coeffs(dest) ≈ strategy_coeffs(expected) rtol=20eps(real(one(T)))
            end
        end
    end
    d = PSDesc(1,63)
    x = CTPS(0.0,1,d)
    @test element((1+x)*(x^62+x^63),[63]) == 2
end

@testset "Translation and fixed-source composition plans" begin
    for T in (Float32,Float64,ComplexF64,BigFloat), nv in (1,3)
        d = PSDesc(nv,5)
        x = [CTPS(zero(T),i,d) for i in 1:nv]
        f = T(2)+x[1]^2+T(0.4)*x[end]^5
        for p in (f,), degree in 0:d.order
            p.degree_mask[] >> degree & 1 != 0 && continue
            s = d.off[degree+1]
            p.c[s:s+d.Nd[degree+1]-1] .= T(NaN)
        end
        plan = CompositionPlan(f)
        ws = CompositionWorkspace(d,T)
        out = CTPS(T,d)
        for shift in (zero(T),T(0.2),T(-0.5))
            g = [shift+v for v in x]
            expected = T(2)+g[1]^2+T(0.4)*g[end]^5
            @test strategy_coeffs(compose(f,g)) ≈ strategy_coeffs(expected) rtol=100eps(real(one(T)))
            @test strategy_coeffs(compose(plan,g)) ≈ strategy_coeffs(expected) rtol=100eps(real(one(T)))
            compose!(out,plan,g,ws)
            @test strategy_coeffs(out) ≈ strategy_coeffs(expected) rtol=100eps(real(one(T)))
            compose!(out,f,g,ws)
            @test strategy_coeffs(out) ≈ strategy_coeffs(expected) rtol=100eps(real(one(T)))
        end
        g = [T(0.1)+v+T(0.02)*x[1]^2 for v in x]
        expected = T(2)+g[1]^2+T(0.4)*g[end]^5
        @test strategy_coeffs(compose(plan,g)) ≈ strategy_coeffs(expected) rtol=100eps(real(one(T)))
        # The source is a snapshot; mutation cannot silently invalidate support.
        PolySeries.zero!(f)
        @test strategy_coeffs(compose(plan,g)) ≈ strategy_coeffs(expected) rtol=100eps(real(one(T)))
        @test_throws ArgumentError compose!(plan.source,plan,g,ws)
        @test_throws DimensionMismatch compose!(out,plan,g,CompositionWorkspace(PSDesc(nv,4),T))
    end
    d = PSDesc(1,63)
    x = CTPS(0.0,1,d)
    shifted = compose(x^63,[1+x])
    @test element(shifted,[0]) == 1
    @test element(shifted,[1]) == 63
    @test element(shifted,[62]) == 63
    @test element(shifted,[63]) == 1
    for n in (0,1)
        d = PSDesc(2,n)
        f = CTPS(3.0,d)
        @test cst(compose(CompositionPlan(f),[CTPS(2.0,d),CTPS(4.0,d)])) == 3
    end
end

function strategy_allocation_checks()
    d = PSDesc(3,5)
    x = [CTPS(0.0,i,d) for i in 1:3]
    a,b = 0.2+x[1],x[2]^2+x[3]^5
    g = [0.1+v for v in x]
    nonlinear = [0.1+v+0.02x[1]^2 for v in x]
    plan,ws,out = CompositionPlan(b),CompositionWorkspace(d),CTPS(Float64,d)
    other = CompositionWorkspace(d)
    @test ws.first_child === other.first_child === d.comp_plan.first_child
    @test ws.next_sibling === other.next_sibling === d.comp_plan.next_sibling
    @test ws.images[1].c !== other.images[1].c
    mul!(out,a,b); compose!(out,b,g,ws); compose!(out,plan,nonlinear,ws)
    @test @allocated(mul!(out,a,b)) == 0
    @test @allocated(compose!(out,b,g,ws)) == 0
    @test @allocated(compose!(out,plan,nonlinear,ws)) == 0
end
@testset "Strategy allocation checks" begin
    strategy_allocation_checks()
end

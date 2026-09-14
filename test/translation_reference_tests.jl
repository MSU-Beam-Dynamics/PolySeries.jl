using Test,PolySeries

@testset "Translation against independent binomial coefficients" begin
    for T in (Float32,Float64,ComplexF64,BigFloat), nv in (2,3)
        d = PSDesc(nv,5)
        p = CTPS(T,d)
        p.degree_mask[] = UInt64(0x25) # Degrees 0,2,5 with deliberately empty gaps.
        for i in 1:d.N
            degree = Int(d.polymap.map[i,1])
            p.c[i] = p.degree_mask[] >> degree & 1 != 0 ? T((-1)^i)/T(i+7) : T(NaN)
        end
        shifts = T[(-1)^v*v/10 for v in 1:nv]
        T === ComplexF64 && (shifts .*= 1+0.25im)
        g = [CTPS(shifts[v],v,d) for v in 1:nv]
        out = compose(p,g)
        reference = zeros(T,d.N)
        # Each original monomial expands independently by the binomial theorem.
        for i in 1:d.N
            alpha = Int.(d.polymap.map[i,2:end])
            coefficient = element(p,alpha)
            for j in 1:d.N
                beta = Int.(d.polymap.map[j,2:end])
                all(beta .<= alpha) || continue
                term = coefficient
                for v in 1:nv
                    term *= binomial(alpha[v],beta[v])*shifts[v]^(alpha[v]-beta[v])
                end
                reference[j] += term
            end
        end
        actual = [element(out,Int.(d.polymap.map[i,2:end])) for i in 1:d.N]
        @test actual ≈ reference rtol=200eps(real(one(T)))
        identity = compose(p,[CTPS(zero(T),v,d) for v in 1:nv])
        @test identity.degree_mask[] == p.degree_mask[]
    end
end

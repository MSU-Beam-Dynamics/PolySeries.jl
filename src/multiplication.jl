# Keep the established general multiplication and tiny-polynomial kernels.
# This boundary is also used by the real Enzyme primitive: its mathematical
# rules still differentiate both operands, even if their primal arrays alias.
@inline function _mul_full!(out::Vector{T}, a, b, desc, square::Bool) where T
    if square
        _square_schedules!(out, a, desc.mul, Val(0))
    else
        mask = typemax(UInt64) >> (63-desc.order)
        _mul_schedules!(out, a, b, mask, mask, desc.mul, Val(true))
    end
    return nothing
end

# Every yielded degree pair must be active in both orientations. Identical
# square operands guarantee that even when their degree masks contain holes.
# block=0 visits a complete column. Other block sizes and output-degree schedule
# ordering are benchmark candidates, not the default without evidence of gains.
function _square_schedules!(out::Vector{T}, a, schedules, ::Val{block}) where {T,block}
    @inbounds for s in schedules
        ib,jb = Int(s.i_start)-1,Int(s.j_start)-1
        ni,nj = Int(s.Ni),Int(s.Nj)
        k = s.k_local
        if s.di == s.dj
            for i in 1:ni
                ai = a[ib+i]
                _prunable_zero(ai) && continue
                @simd for j in 1:i-1
                    v = ai*a[jb+j]
                    out[k[j,i]] += v+v
                end
                out[k[i,i]] += ai*ai
            end
        else
            for start in 1:(block == 0 ? nj : block):nj
                stop = block == 0 ? nj : min(start+block-1,nj)
                for i in 1:ni
                    ai = a[ib+i]
                    _prunable_zero(ai) && continue
                    # Adding a fixed monomial is injective: j iterations write
                    # distinct output coefficients, so SIMD has no conflicts.
                    @simd for j in start:stop
                        v = ai*a[jb+j]
                        # Double the product, not ai: 2ai can overflow even
                        # when ai*aj and its doubled contribution are finite.
                        out[k[j,i]] += v+v
                    end
                end
            end
        end
    end
    return nothing
end

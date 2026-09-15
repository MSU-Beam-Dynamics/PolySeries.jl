# Dispatch between squaring, selected output plans, and the general kernels.
# This boundary is also used by the real Enzyme primitive: its mathematical
# rules still differentiate both operands, even if their primal arrays alias.
@inline function _mul_full!(out::Vector{T}, a, b, desc, square::Bool) where T
    if square
        _square_schedules!(out, a, desc.mul, Val(0))
    elseif T <: Union{Float32,Float64} && desc.output_product !== nothing &&
           _output_product_dense(a,b)
        _output_product!(out,a,b,desc.output_product)
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

# Both operands must already have fully initialized coefficient storage. This
# is only a performance choice: either path computes the same full product.
# Checking a short prefix avoids a full support scan on each multiplication.
@inline function _output_product_dense(a,b)
    zeros_seen=0
    @inbounds for i in 1:min(16,length(a))
        zeros_seen += iszero(a[i]) && iszero(b[i])
        zeros_seen >= 4 && return false
    end
    return true
end

function build_output_product_plan(N::Int, schedules, ::Type{I}) where {I<:Union{UInt16,Int32}}
    1 <= N <= typemax(I) || throw(ArgumentError("Output product indices do not fit $I"))
    counts=zeros(Int,N)
    diagonal=zeros(I,N)
    for s in schedules
        ib=Int(s.i_start)-1
        @inbounds for i in 1:Int(s.Ni)
            nj=s.di==s.dj ? i-1 : Int(s.Nj)
            for j in 1:nj
                counts[s.k_local[j,i]]+=1
            end
            s.di==s.dj && (diagonal[s.k_local[i,i]]=I(ib+i))
        end
    end
    offsets=Vector{Int}(undef,N+1)
    offsets[1]=1
    for k in 1:N
        offsets[k+1]=offsets[k]+counts[k]
    end
    left=Vector{I}(undef,offsets[end]-1)
    right=similar(left)
    cursor=copy(offsets)
    for s in schedules
        ib,jb=Int(s.i_start)-1,Int(s.j_start)-1
        @inbounds for i in 1:Int(s.Ni)
            nj=s.di==s.dj ? i-1 : Int(s.Nj)
            for j in 1:nj
                k=s.k_local[j,i]
                p=cursor[k]
                cursor[k]+=1
                left[p]=I(ib+i)
                right[p]=I(jb+j)
            end
        end
    end
    return OutputProductPlan(offsets,left,right,diagonal)
end

# Accumulate each output in a scalar and write it once. Off-diagonal pairs
# account for both input orientations; self-products are stored separately.
function _output_product!(out::Vector{T},a,b,plan::OutputProductPlan) where T
    @inbounds for k in eachindex(out)
        value=zero(T)
        for p in plan.offsets[k]:plan.offsets[k+1]-1
            i,j=plan.left[p],plan.right[p]
            value+=a[i]*b[j]+a[j]*b[i]
        end
        i=plan.diagonal[k]
        out[k]=i==0 ? value : value+a[i]*b[i]
    end
    return nothing
end

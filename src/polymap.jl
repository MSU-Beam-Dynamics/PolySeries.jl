# ==============================================================================
# This file is part of the PolySeries.jl Julia package.
#
# Author: Jinyu Wan
# Email: wan@frib.msu.edu
# Version: 1.0
# Created Date: 11-01-2023
# Modified Date: 11-06-2023


# using Zygote

struct PolyMap
    dim::Int
    max_order::Int
    map::Matrix{UInt8}  # UInt8 sufficient: max exponent ≤ max_order < 256

    function PolyMap(dim::Int, order::Int)
        new(dim, order, setindexmap(dim, order))
    end
end
function Base.copy(pm::PolyMap)
    new_pm = PolyMap(pm.dim, pm.max_order)
    return new_pm
end

"""
    decomposite(n::Int, dim::Int) -> Vector{Int}

Decomposes an integer n into a vector of length dim + 1 representing exponents.

# Arguments
- `n::Int`: The integer to decompose.
- `dim::Int`: The number of variables (dimensions).

# Returns
- `Vector{Int}`: A vector of length dim + 1 where the first element is the total degree 
  and the subsequent elements are the exponents for each variable.

# Examples
```jldoctest
julia> decomposite(0, 2)
3-element Vector{Int64}:
 0
 0
 0

julia> decomposite(1, 2)
3-element Vector{Int64}:
 1
 1
 0

julia> decomposite(3, 2)
3-element Vector{Int64}:
 2
 2
 0
```
"""
function decomposite(n::Int, dim::Int)    
    result = zeros(Int, dim + 1)  # Return Vector{Int}, not Vector{Float64}
    itemp = n + 1
    for i in dim:-1:1
        k = i - 1
        while binomial(k, i) < itemp
            k += 1
        end
        itemp -= binomial(k - 1, i)
        result[dim - i + 1] = k - i
    end
    for i in dim+1:-1:2
        result[i] = result[i - 1] - result[i]
    end
    return result
end


function setindexmap(dim::Int, max_order::Int)
    totallength = binomial(max_order + dim, dim)
    # UInt8 is 8× more cache-friendly than Int64; max exponent ≤ max_order ≤ 127 in practice
    map = zeros(UInt8, totallength, dim + 1)
    for i in 0:totallength-1
        map[i + 1, :] = decomposite(i, dim)  # Julia converts Int → UInt8 on assignment
    end
    return map
end

# function getindexmap(p::PolyMap, i::Int)
#     if i < 1 || i > length(p.map)
#         error("index out of range")
#     end
#     return p.map[i]
# end
function getindexmap(p::PolyMap, i::Int)
    if i < 1 || i > size(p.map, 1)  # Check number of rows, not total elements
        error("index out of range")
    end
    return @view p.map[i, :]  # Return view to avoid allocation
end
# z = PolyMap(6, 2)
# println(getindexmap(z, 1))

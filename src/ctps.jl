
# Multiplication schedule - input-major format
# Groups by first operand index to enable loading c1[i] once and reusing across all pairs
struct MulScheduleInputMajor
    ivalues::Vector{Int32}   # Unique i values (first operand indices)
    i_start::Vector{Int32}   # Start index in jidx/kidx for each i value
    i_end::Vector{Int32}     # End index (inclusive) for each i value  
    jidx::Vector{Int32}      # Second operand indices (flat list)
    kidx::Vector{Int32}      # Output indices (flat list, sorted within each i group)
end

# Empty schedule
MulScheduleInputMajor() = MulScheduleInputMajor(Int32[], Int32[], Int32[], Int32[], Int32[])

# Multiplication schedule - output-major format (GPU/thread-safe)
# Groups by output index - each thread/GPU core computes one output coefficient independently
# No race conditions, no atomics needed, perfect for parallel execution
struct MulScheduleOutputMajor
    kvalues::Vector{Int32}   # Unique output indices (sorted)
    k_start::Vector{Int32}   # Start index in iidx/jidx for each k value
    k_end::Vector{Int32}     # End index (inclusive) for each k value
    iidx::Vector{Int32}      # First operand indices (flat list)
    jidx::Vector{Int32}      # Second operand indices (flat list)
end

# Empty schedule
MulScheduleOutputMajor() = MulScheduleOutputMajor(Int32[], Int32[], Int32[], Int32[], Int32[])

# Composition plan for efficient function composition
struct CompPlan
    # Placeholder for composition optimization data
    data::Vector{Int}
end

# TPSA Descriptor - immutable, shared metadata
struct TPSADesc
    nv::Int                       # number of variables
    order::Int                    # maximum order
    N::Int                        # total number of coefficients
    Nd::Vector{Int}               # size per degree
    off::Vector{Int}              # start offset per degree (1-based)
    polymap::PolyMap              # index mapping (index → exponent)
    exp_to_idx::Dict              # reverse map: SVector{nv+1,UInt8} → Int (concrete per instance)
    mul::Vector{MulScheduleInputMajor}    # Input-major multiplication schedules indexed by (di,dj)
    mul_output::Vector{MulScheduleOutputMajor}  # Output-major schedules (GPU/thread-safe)
    comp_plan::CompPlan           # composition build plan
end

# Thread-safe cache for TPSADesc instances
const DESC_CACHE = Dict{Tuple{Int,Int}, TPSADesc}()

# Global default descriptor
const GLOBAL_DESC = Ref{Union{TPSADesc, Nothing}}(nothing)

"""
    set_descriptor!(nv::Int, order::Int)

Set the global default descriptor for all CTPS operations.
This should be called once at the beginning of your program.

# Arguments
- `nv::Int`: Number of variables
- `order::Int`: Maximum order

# Example
```julia
using TPSA
set_descriptor!(3, 4)  # 3 variables, order 4
x = CTPS(0.0, 1)       # Create variable x
y = CTPS(0.0, 2)       # Create variable y
```
"""
function set_descriptor!(nv::Int, order::Int)
    GLOBAL_DESC[] = TPSADesc(nv, order)
    return GLOBAL_DESC[]
end

"""
    get_descriptor()

Get the current global descriptor. Throws an error if not set.
"""
function get_descriptor()
    if GLOBAL_DESC[] === nothing
        error("No global descriptor set. Call set_descriptor!(nv, order) first.")
    end
    return GLOBAL_DESC[]
end

"""
    clear_descriptor!()

Clear the global descriptor.
"""
function clear_descriptor!()
    GLOBAL_DESC[] = nothing
end
const DESC_CACHE_LOCK = ReentrantLock()

# Constructor with caching (thread-safe)
function TPSADesc(nv::Int, order::Int)
    key = (nv, order)
    
    # Thread-safe cache lookup: always acquire lock
    # Note: Base.Dict is NOT safe for concurrent read/write, so we must lock even for reads
    return lock(DESC_CACHE_LOCK) do
        # Check if already cached
        desc = get(DESC_CACHE, key, nothing)
        if desc !== nothing
            return desc
        end
        
        # Compute total number of coefficients
        N = binomial(nv + order, order)
        
        # Compute size per degree (number of monomials at each degree)
        Nd = zeros(Int, order + 1)
        Nd[1] = 1  # degree 0: just constant
        for d in 1:order
            Nd[d + 1] = binomial(nv + d, d) - binomial(nv + d - 1, d - 1)
        end
        
        # Compute start offset per degree (1-based indexing)
        off = zeros(Int, order + 1)
        off[1] = 1
        for d in 1:order
            off[d + 1] = off[d] + Nd[d]
        end
        
        # Create polymap and reverse lookup
        polymap = PolyMap(nv, order)
        
        # Build reverse map: exponent → index (using SVector for type stability)
        # SVector{K,UInt8} is stack-allocated and type-stable for small nv
        K = nv + 1
        exp_to_idx = Dict{SVector{K,UInt8}, Int32}()
        for idx in 1:N
            # Read directly from matrix to avoid allocation (no getindexmap slice)
            exp_svec = SVector{K,UInt8}(UInt8(polymap.map[idx, v]) for v in 1:K)
            exp_to_idx[exp_svec] = Int32(idx)
        end
        
        # Build multiplication schedules for all degree pairs (di, dj)
        mul = MulScheduleInputMajor[]
        mul_output = MulScheduleOutputMajor[]
        for di in 0:order
            for dj in 0:order
                dk = di + dj
                if dk > order
                    push!(mul, MulScheduleInputMajor())  # empty schedule
                    # push!(mul_output, MulScheduleOutputMajor())  # empty schedule
                else
                    schedule = build_mul_schedule(polymap, exp_to_idx, nv, order, di, dj, dk, off, Nd)
                    push!(mul, schedule)
                    
                    # schedule_output = build_mul_schedule_output_major(polymap, exp_to_idx, nv, order, di, dj, dk, off, Nd)
                    # push!(mul_output, schedule_output)
                end
            end
        end
        
        # Create composition plan (placeholder)
        comp_plan = CompPlan(Int[])
        
        desc = TPSADesc(nv, order, N, Nd, off, polymap, exp_to_idx, mul, mul_output, comp_plan)
        DESC_CACHE[key] = desc
        return desc
    end
end

# Build input-major multiplication schedule for degree pair (di, dj) -> dk
# Groups by first operand to maximize reuse: load c1[i] once, use for all j
# Sorts by k within each i group for better cache locality on output writes
# Uses two-pass counting: O(nnz), zero tuple allocations
# NOTE: Sorted k is NOT consecutive! Must store kidx[pos] for each pair.
function build_mul_schedule(polymap::PolyMap, exp_to_idx::Dict,
                            nv::Int, order::Int, 
                            di::Int, dj::Int, dk::Int,
                            off::Vector{Int}, Nd::Vector{Int})
    # Number of terms at each degree
    n_i = Nd[di + 1]
    n_j = Nd[dj + 1]
    n_k = Nd[dk + 1]
    
    # Starting indices for each degree
    i_start = off[di + 1]
    j_start = off[dj + 1]
    k_start_base = off[dk + 1]
    k_end = k_start_base + n_k - 1
    
    K = nv + 1
    
    # Pass 1: Count valid pairs for each i_local
    nnz_i = zeros(Int, n_i)
    
    for i_local in 1:n_i
        i = i_start + i_local - 1
        
        for j_local in 1:n_j
            j = j_start + j_local - 1
            
            # Check degree sum
            degree_sum = polymap.map[i, 1] + polymap.map[j, 1]
            if degree_sum != dk
                continue
            end
            
            # Compute output exponent and check validity
            exp_k_svec = SVector{K,UInt8}(UInt8(polymap.map[i, v] + polymap.map[j, v]) for v in 1:K)
            k = get(exp_to_idx, exp_k_svec, 0)
            
            if k >= k_start_base && k <= k_end
                nnz_i[i_local] += 1
            end
        end
    end
    
    # Compute total nnz and check for empty schedule
    total_nnz = sum(nnz_i)
    if total_nnz == 0
        return MulScheduleInputMajor(Int32[], Int32[], Int32[], Int32[], Int32[])
    end
    
    # Compute prefix sum to get starting positions
    i_offsets = zeros(Int, n_i + 1)
    i_offsets[1] = 1
    for i_local in 1:n_i
        i_offsets[i_local + 1] = i_offsets[i_local] + nnz_i[i_local]
    end
    
    # Allocate output arrays - temporary kidx for sorting, final jidx only
    jidx_temp = Vector{Int}(undef, total_nnz)
    kidx_temp = Vector{Int}(undef, total_nnz)
    
    # Pass 2: Fill jidx/kidx temporarily
    fill_pos = copy(i_offsets[1:n_i])  # Current write position for each i
    
    for i_local in 1:n_i
        i = i_start + i_local - 1
        
        for j_local in 1:n_j
            j = j_start + j_local - 1
            
            # Check degree sum
            degree_sum = polymap.map[i, 1] + polymap.map[j, 1]
            if degree_sum != dk
                continue
            end
            
            # Compute output exponent and check validity
            exp_k_svec = SVector{K,UInt8}(UInt8(polymap.map[i, v] + polymap.map[j, v]) for v in 1:K)
            k = get(exp_to_idx, exp_k_svec, 0)
            
            if k >= k_start_base && k <= k_end
                pos = fill_pos[i_local]
                jidx_temp[pos] = j
                kidx_temp[pos] = k
                fill_pos[i_local] += 1
            end
        end
    end
    
    # Sort by k within each i group to make k sequential
    for i_local in 1:n_i
        if nnz_i[i_local] > 1
            range_start = i_offsets[i_local]
            range_end = i_offsets[i_local + 1] - 1
            
            # Sort by k (stable sort preserves j order for same k)
            perm = sortperm(view(kidx_temp, range_start:range_end))
            jidx_temp[range_start:range_end] = jidx_temp[range_start:range_end][perm]
            kidx_temp[range_start:range_end] = kidx_temp[range_start:range_end][perm]
        end
    end
    
    # Convert to Int32 and build final arrays
    jidx = Vector{Int32}(undef, total_nnz)
    kidx = Vector{Int32}(undef, total_nnz)
    ivalues = Int32[]
    i_start_arr = Int32[]
    i_end_arr = Int32[]
    
    for i_local in 1:n_i
        if nnz_i[i_local] > 0
            range_start = i_offsets[i_local]
            range_end = i_offsets[i_local + 1] - 1
            
            push!(ivalues, Int32(i_start + i_local - 1))
            push!(i_start_arr, Int32(range_start))
            push!(i_end_arr, Int32(range_end))
            
            # Copy jidx and kidx to final arrays with Int32 conversion
            for pos in range_start:range_end
                jidx[pos] = Int32(jidx_temp[pos])
                kidx[pos] = Int32(kidx_temp[pos])
            end
        end
    end
    
    return MulScheduleInputMajor(ivalues, i_start_arr, i_end_arr, jidx, kidx)
end

# Build output-major multiplication schedule for degree pair (di, dj) -> dk
# Groups by output index - each output coefficient is computed independently
# Perfect for GPU/threaded execution: no race conditions, no atomics needed
# Each thread computes one k value by accumulating all contributing (i,j) pairs
function build_mul_schedule_output_major(polymap::PolyMap, exp_to_idx::Dict,
                                          nv::Int, order::Int, 
                                          di::Int, dj::Int, dk::Int,
                                          off::Vector{Int}, Nd::Vector{Int})
    # Number of terms at each degree
    n_i = Nd[di + 1]
    n_j = Nd[dj + 1]
    n_k = Nd[dk + 1]
    
    # Starting indices for each degree
    i_start = off[di + 1]
    j_start = off[dj + 1]
    k_start_base = off[dk + 1]
    k_end = k_start_base + n_k - 1
    
    K = nv + 1
    
    # Pass 1: Count contributions for each k_local
    nnz_k = zeros(Int, n_k)
    
    for i_local in 1:n_i
        i = i_start + i_local - 1
        
        for j_local in 1:n_j
            j = j_start + j_local - 1
            
            # Check degree sum
            degree_sum = polymap.map[i, 1] + polymap.map[j, 1]
            if degree_sum != dk
                continue
            end
            
            # Compute output exponent and check validity
            exp_k_svec = SVector{K,UInt8}(UInt8(polymap.map[i, v] + polymap.map[j, v]) for v in 1:K)
            k = get(exp_to_idx, exp_k_svec, 0)
            
            if k >= k_start_base && k <= k_end
                k_local = k - k_start_base + 1
                nnz_k[k_local] += 1
            end
        end
    end
    
    # Compute total nnz and check for empty schedule
    total_nnz = sum(nnz_k)
    if total_nnz == 0
        return MulScheduleOutputMajor(Int32[], Int32[], Int32[], Int32[], Int32[])
    end
    
    # Compute prefix sum to get starting positions
    k_offsets = zeros(Int, n_k + 1)
    k_offsets[1] = 1
    for k_local in 1:n_k
        k_offsets[k_local + 1] = k_offsets[k_local] + nnz_k[k_local]
    end
    
    # Allocate output arrays
    iidx_temp = Vector{Int}(undef, total_nnz)
    jidx_temp = Vector{Int}(undef, total_nnz)
    
    # Pass 2: Fill iidx/jidx
    fill_pos = copy(k_offsets[1:n_k])  # Current write position for each k
    
    for i_local in 1:n_i
        i = i_start + i_local - 1
        
        for j_local in 1:n_j
            j = j_start + j_local - 1
            
            # Check degree sum
            degree_sum = polymap.map[i, 1] + polymap.map[j, 1]
            if degree_sum != dk
                continue
            end
            
            # Compute output exponent and check validity
            exp_k_svec = SVector{K,UInt8}(UInt8(polymap.map[i, v] + polymap.map[j, v]) for v in 1:K)
            k = get(exp_to_idx, exp_k_svec, 0)
            
            if k >= k_start_base && k <= k_end
                k_local = k - k_start_base + 1
                pos = fill_pos[k_local]
                iidx_temp[pos] = i
                jidx_temp[pos] = j
                fill_pos[k_local] += 1
            end
        end
    end
    
    # Convert to Int32 and build final arrays
    iidx = Vector{Int32}(undef, total_nnz)
    jidx = Vector{Int32}(undef, total_nnz)
    kvalues = Int32[]
    k_start_arr = Int32[]
    k_end_arr = Int32[]
    
    for k_local in 1:n_k
        if nnz_k[k_local] > 0
            range_start = k_offsets[k_local]
            range_end = k_offsets[k_local + 1] - 1
            
            push!(kvalues, Int32(k_start_base + k_local - 1))
            push!(k_start_arr, Int32(range_start))
            push!(k_end_arr, Int32(range_end))
            
            # Copy iidx and jidx to final arrays with Int32 conversion
            for pos in range_start:range_end
                iidx[pos] = Int32(iidx_temp[pos])
                jidx[pos] = Int32(jidx_temp[pos])
            end
        end
    end
    
    return MulScheduleOutputMajor(kvalues, k_start_arr, k_end_arr, iidx, jidx)
end

# CTPS with new design
struct CTPS{T}
    c::Vector{T}      # coefficients, length desc.N
    desc::TPSADesc    # immutable, shared descriptor
    degree_mask::Base.RefValue{UInt64}  # Bit i = 1 if degree i has non-zeros (mutable)
end

# Compute degree mask from coefficients
function compute_degree_mask(c::Vector{T}, desc::TPSADesc) where T
    mask = UInt64(0)
    order = desc.order
    @inbounds for d in 0:order
        d_start = desc.off[d + 1]
        d_end = d_start + desc.Nd[d + 1] - 1
        for i in d_start:d_end
            if !iszero(c[i])
                mask |= (UInt64(1) << d)
                break
            end
        end
    end
    return mask
end

# Update the degree mask after manual coefficient modifications
function update_degree_mask!(ctps::CTPS)
    ctps.degree_mask[] = compute_degree_mask(ctps.c, ctps.desc)
    return ctps
end

# Constructor: zero CTPS of given type, variables, and order
function CTPS(T::Type, nv::Int, order::Int)
    desc = TPSADesc(nv, order)
    c = zeros(T, desc.N)
    return CTPS{T}(c, desc, Ref(UInt64(0)))
end

# Constructor: constant CTPS
function CTPS(a::T, nv::Int, order::Int) where T
    desc = TPSADesc(nv, order)
    c = zeros(T, desc.N)
    c[1] = a
    mask = iszero(a) ? UInt64(0) : UInt64(1)  # Degree 0 has non-zero
    return CTPS{T}(c, desc, Ref(mask))
end

# Constructor: variable CTPS (a + δxₙ)
function CTPS(a::T, n::Int, nv::Int, order::Int) where T
    if n <= nv && n > 0
        desc = TPSADesc(nv, order)
        c = zeros(T, desc.N)
        c[n + 1] = one(T)  # linear term for variable n
        c[1] = a           # constant term
        # Degree 0 and degree 1 have non-zeros
        mask = UInt64(0x3)  # bits 0 and 1 set
        if iszero(a)
            mask = UInt64(0x2)  # only bit 1 set
        end
        return CTPS{T}(c, desc, Ref(mask))
    else
        error("Variable index out of range in CTPS")
    end
end

# Copy constructor
function CTPS(M::CTPS{T}) where T
    c = copy(M.c)
    return CTPS{T}(c, M.desc, Ref(M.degree_mask[]))
end

# ========== Simplified constructors using global descriptor ==========

# Constructor: zero CTPS using global descriptor
function CTPS(T::Type)
    desc = get_descriptor()
    c = zeros(T, desc.N)
    return CTPS{T}(c, desc, Ref(UInt64(0)))
end

# Constructor: constant CTPS using global descriptor
function CTPS(a::T) where T<:Number
    desc = get_descriptor()
    c = zeros(T, desc.N)
    c[1] = a
    mask = iszero(a) ? UInt64(0) : UInt64(1)
    return CTPS{T}(c, desc, Ref(mask))
end

# Constructor: variable CTPS using global descriptor
function CTPS(a::T, n::Int) where T<:Number
    desc = get_descriptor()
    nv = desc.nv
    if n <= nv && n > 0
        c = zeros(T, desc.N)
        c[n + 1] = one(T)  # linear term for variable n
        c[1] = a           # constant term
        mask = UInt64(0x3)  # bits 0 and 1 set
        if iszero(a)
            mask = UInt64(0x2)  # only bit 1 set
        end
        return CTPS{T}(c, desc, Ref(mask))
    else
        error("Variable index $n out of range (must be 1 to $nv)")
    end
end

# ========== End simplified constructors ==========

function cst(ctps::CTPS{T}) where T
    return ctps.c[1]
end

function findindex(ctps::CTPS{T}, indexmap::Vector{Int}) where T
    # find the index of the indexmap in the coefficient vector
    # indexmap is a vector of length nv + 1, e.g. [0, 1, 1] for x1^1 * x2^1
    dim = ctps.desc.nv
    if length(indexmap) == dim
        # Compute total and work with conceptual [total; indexmap]
        total = Base.sum(indexmap)
        # Build cumsum incrementally: cumsum[1]=total, cumsum[i]=cumsum[i-1]-indexmap[i-1]
        # We read conceptual indexmap as: [total; indexmap[1]; indexmap[2]; ...; indexmap[dim]]
        cumsum_val = total
        result = Int(1)
        for i in dim:-1:1
            # We need cumsum[dim - i + 1]
            # cumsum[1] = total (already have)
            # cumsum[2] = total - indexmap[1]
            # cumsum[k] = total - sum(indexmap[1:k-1])
            # For iteration i (dim downto 1), we need cumsum[dim - i + 1]
            # So we need to have subtracted indexmap[1:dim-i]
            # Build cumsum by subtracting as we go backwards
            if cumsum_val == 0
                break
            end
            if cumsum_val < 0
                error("The index map has invalid component")
            end
            result += binomial(cumsum_val - 1 + i, i)
            # Prepare for next iteration: subtract indexmap[dim - i + 1]
            cumsum_val -= indexmap[dim - i + 1]
        end
        return result
    end
    if length(indexmap) != (dim + 1)
        error("Index map does not have correct length")
    end
    # Original pattern: cumsum[1]=indexmap[1], cumsum[i]=cumsum[i-1]-indexmap[i]
    # For i in dim:-1:1, we need cumsum[dim - i + 1]
    # Build cumsum values incrementally without allocating array
    # cumsum[k] = indexmap[1] - sum(indexmap[2:k])
    
    # Start: we'll build cumsum values on the fly
    # For the first iteration (i=dim), we need cumsum[1] = indexmap[1]
    cumsum_val = indexmap[1]
    result = Int(1)
    
    for i in dim:-1:1
        # At loop entry, cumsum_val = cumsum[dim - i + 1]
        if cumsum_val == 0
            break
        end
        if cumsum_val < 0 || indexmap[dim - i + 2] < 0
            error("The index map has invalid component")
        end
        result += binomial(cumsum_val - 1 + i, i)
        # Update cumsum_val for next iteration: cumsum[dim-i+2] = cumsum[dim-i+1] - indexmap[dim-i+2]
        if i > 1  # Only update if there's a next iteration
            cumsum_val -= indexmap[dim - i + 2]
        end
    end
    return result
end

# function findpower(ctps::CTPS{T, TPS_Dim, Max_TPS_Degree}, n::Int) where {T, TPS_Dim, Max_TPS_Degree}
#     if n < ctps.terms
#         return getindexmap(ctps.polymap[], n)
#     else
#         error("The index is out of range")
#     end
# end

# function redegree!(ctps::CTPS{T, TPS_Dim, Max_TPS_Degree}, degree::Int) where {T, TPS_Dim, Max_TPS_Degree}
#     ctps.degree = min(degree, Max_TPS_Degree)
#     ctps.terms = binomial(TPS_Dim + ctps.degree, ctps.degree)
#     new_map = [i <= length(ctps.map) ? ctps.map[i] : zero(T) for i in 1:ctps.terms]
#     ctps.map = new_map
# end
# function redegree(ctps::CTPS{T, TPS_Dim, Max_TPS_Degree}, degree::Int) where {T, TPS_Dim, Max_TPS_Degree}
#     degree = min(degree, Max_TPS_Degree)
#     terms = binomial(TPS_Dim + degree, degree)
#     new_map = zeros(T, terms)
#     for i in 1:ctps.terms
#         new_map[i] = ctps.map[i]
#     end
#     # new_map = [i <= length(ctps.map) ? ctps.map[i] : 0.0 for i in 1:terms]
#     # polymap = getOrCreatePolyMap(TPS_Dim, Max_TPS_Degree)
#     ctps_new = CTPS{T, TPS_Dim, Max_TPS_Degree}(degree, terms, new_map, ctps.polymap)
#     return ctps_new
# end
# function redegree(ctps::CTPS{T, TPS_Dim, Max_TPS_Degree}, degree::Int) where {T, TPS_Dim, Max_TPS_Degree}
#     degree = min(degree, Max_TPS_Degree)
#     terms = binomial(TPS_Dim + degree, degree)
#     new_map = zeros(T, terms)
#     new_map_buffer = Zygote.Buffer(new_map)
#     for i in 1:ctps.terms
#         new_map_buffer[i] = ctps.map[i]
#     end
#     for i in ctps.terms+1:terms
#         new_map_buffer[i] = zero(T)
#     end
#     new_map = copy(new_map_buffer)
#     ctps_new = CTPS{T, TPS_Dim, Max_TPS_Degree}(degree, terms, new_map, PolyMap(TPS_Dim, Max_TPS_Degree))
#     return ctps_new
# end
function assign!(ctps::CTPS{T}, a::T, n_var::Int) where T
    if n_var <= ctps.desc.nv && n_var > 0
        ctps.c[n_var + 1] = one(T)
        ctps.c[1] = a
        return nothing
    else
        error("Variable index out of range in CTPS")
    end
end

function assign!(ctps::CTPS{T}, a::T) where T
    ctps.c[1] = a
    return nothing
end

function reassign!(ctps::CTPS{T}, a::T, n_var::Int) where T
    if n_var <= ctps.desc.nv && n_var > 0
        fill!(ctps.c, zero(T))
        ctps.c[n_var + 1] = one(T)
        ctps.c[1] = a
        return nothing
    else
        error("Variable index out of range in CTPS")
    end
end



@inline function element(ctps::CTPS{T}, ind::Vector{Int}) where T
    result = findindex(ctps, ind)
    return ctps.c[result]
end



# Overloaded operations
import Base: +, -, *, /, sin, cos, tan, sinh, cosh, asin, acos, sqrt, ^, inv, exp, log, copy!


function add!(result::CTPS{T}, ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    @inbounds @simd for i in eachindex(ctps1.c)
        result.c[i] = ctps1.c[i] + ctps2.c[i]
    end
    # Union of masks (result may have non-zeros where either input has them)
    result.degree_mask[] = ctps1.degree_mask[] | ctps2.degree_mask[]
    return nothing
end

function add!(result::CTPS{T}, ctps1::CTPS{T}, a::T) where T
    @inbounds @simd for i in eachindex(ctps1.c)
        result.c[i] = ctps1.c[i]
    end
    result.c[1] += a
    return nothing
end

function addto!(ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    @inbounds @simd for i in eachindex(ctps1.c)
        ctps1.c[i] += ctps2.c[i]
    end
    return nothing
end

function sub!(result::CTPS{T}, ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    @inbounds @simd for i in eachindex(ctps1.c)
        result.c[i] = ctps1.c[i] - ctps2.c[i]
    end
    return nothing
end

function subfrom!(ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    @inbounds @simd for i in eachindex(ctps1.c)
        ctps1.c[i] -= ctps2.c[i]
    end
    return nothing
end

function scale!(ctps::CTPS{T}, a::T) where T
    for i in eachindex(ctps.c)
        ctps.c[i] *= a
    end
    return nothing
end

function copy!(dest::CTPS{T}, src::CTPS{T}) where T
    copyto!(dest.c, src.c)
    return nothing
end

function zero!(ctps::CTPS{T}) where T
    fill!(ctps.c, zero(T))
    ctps.degree_mask[] = UInt64(0)
    return nothing
end

# In-place multiplication: result = ctps1 * ctps2
# Input-major kernel: maximize reuse of c1[i] by processing all j,k pairs that use same i
function mul!(result::CTPS{T}, ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    fill!(result.c, zero(T))
    
    desc = ctps1.desc
    order = desc.order
    
    c1 = ctps1.c
    c2 = ctps2.c
    cr = result.c
    
    # Get sparsity masks
    mask1 = ctps1.degree_mask[]
    mask2 = ctps2.degree_mask[]
    
    # Iterate over all degree combinations
    @inbounds for di in 0:order
        # Skip if degree di has no non-zeros in ctps1
        if (mask1 & (UInt64(1) << di)) == 0
            continue
        end
        
        for dj in 0:order
            # Skip if degree dj has no non-zeros in ctps2
            if (mask2 & (UInt64(1) << dj)) == 0
                continue
            end
            
            dk = di + dj
            if dk > order
                continue
            end
            
            # Get precomputed input-major schedule for (di, dj)
            schedule_idx = di * (order + 1) + dj + 1
            schedule = desc.mul[schedule_idx]
            
            ivalues = schedule.ivalues
            i_starts = schedule.i_start
            i_ends = schedule.i_end
            jidx = schedule.jidx
            kidx = schedule.kidx
            
            # Process each unique i value
            for i_idx in 1:length(ivalues)
                i = ivalues[i_idx]
                ai = c1[i]  # Load once and reuse!
                
                # Skip if zero (sparse optimization)
                if iszero(ai)
                    continue
                end
                
                range_start = i_starts[i_idx]
                range_end = i_ends[i_idx]
                
                # Inner loop: Use actual k indices from schedule
                # kidx is sorted within each i group for better cache locality
                @inbounds for pos in range_start:range_end
                    j = jidx[pos]
                    k = kidx[pos]
                    cr[k] += ai * c2[j]
                end
            end
        end
    end
    
    # Compute degree mask for result
    result.degree_mask[] = compute_degree_mask(cr, desc)
    
    return nothing
end

# Thread-safe multiplication using output-major schedule
# Each thread computes a subset of output coefficients independently
# No race conditions, no atomics needed - perfect for GPU/parallel execution
function mul_threaded!(result::CTPS{T}, ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    fill!(result.c, zero(T))
    
    desc = ctps1.desc
    order = desc.order
    
    c1 = ctps1.c
    c2 = ctps2.c
    cr = result.c
    
    # Get sparsity masks
    mask1 = ctps1.degree_mask[]
    mask2 = ctps2.degree_mask[]
    
    # For the threaded version, we need to merge contributions from all (di,dj) pairs
    # to avoid race conditions. We'll do it sequentially across (di,dj) pairs but
    # parallelize within each pair.
    
    @inbounds for di in 0:order
        # Skip if degree di has no non-zeros in ctps1
        if (mask1 & (UInt64(1) << di)) == 0
            continue
        end
        
        for dj in 0:order
            # Skip if degree dj has no non-zeros in ctps2
            if (mask2 & (UInt64(1) << dj)) == 0
                continue
            end
            
            dk = di + dj
            if dk > order
                continue
            end
            
            # Get precomputed output-major schedule for (di, dj)
            schedule_idx = di * (order + 1) + dj + 1
            schedule = desc.mul_output[schedule_idx]
            
            if isempty(schedule.kvalues)
                continue
            end
            
            kvalues = schedule.kvalues
            k_starts = schedule.k_start
            k_ends = schedule.k_end
            iidx = schedule.iidx
            jidx = schedule.jidx
            
            # Parallelize over output coefficients within this (di,dj) pair
            # Each thread computes one k independently - no race condition!
            Threads.@threads for k_idx in 1:length(kvalues)
                k = kvalues[k_idx]
                accumulator = zero(T)
                
                range_start = k_starts[k_idx]
                range_end = k_ends[k_idx]
                
                # Accumulate all contributions to k from this (di,dj) pair
                @inbounds @simd for pos in range_start:range_end
                    i = iidx[pos]
                    j = jidx[pos]
                    accumulator += c1[i] * c2[j]
                end
                
                # Add contribution to cr[k] (thread-safe since different (di,dj) run sequentially)
                cr[k] += accumulator
            end
        end
    end
    
    # Compute degree mask for result
    result.degree_mask[] = compute_degree_mask(cr, desc)
    
    return nothing
end


# +
function +(ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    # Check descriptor compatibility
    if ctps1.desc !== ctps2.desc
        if ctps1.desc.nv != ctps2.desc.nv || ctps1.desc.order != ctps2.desc.order
            error("Cannot add CTPS with different descriptors: (nv=$(ctps1.desc.nv), order=$(ctps1.desc.order)) vs (nv=$(ctps2.desc.nv), order=$(ctps2.desc.order))")
        end
    end
    ctps_new = CTPS(ctps1)
    for i in eachindex(ctps_new.c)
        ctps_new.c[i] += ctps2.c[i]
    end
    return ctps_new
end

function +(ctps::CTPS{T}, a::Number) where T
    ctps_new = CTPS(ctps)
    ctps_new.c[1] += a
    return ctps_new
end

function +(a::Number, ctps::CTPS{T}) where T
    return ctps + a
end
# -
function -(ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    # Check descriptor compatibility
    if ctps1.desc !== ctps2.desc
        if ctps1.desc.nv != ctps2.desc.nv || ctps1.desc.order != ctps2.desc.order
            error("Cannot subtract CTPS with different descriptors: (nv=$(ctps1.desc.nv), order=$(ctps1.desc.order)) vs (nv=$(ctps2.desc.nv), order=$(ctps2.desc.order))")
        end
    end
    ctps_new = CTPS(ctps1)
    for i in eachindex(ctps_new.c)
        ctps_new.c[i] -= ctps2.c[i]
    end
    return ctps_new
end

function -(ctps::CTPS{T}, a::Number) where T
    ctps_new = CTPS(ctps)
    ctps_new.c[1] -= a
    return ctps_new
end

function -(a::Number, ctps::CTPS{T}) where T
    ctps_new = CTPS(ctps)
    ctps_new.c[1] = a - ctps_new.c[1]
    for i in 2:length(ctps_new.c)
        ctps_new.c[i] = -ctps_new.c[i]
    end
    return ctps_new
end

function -(ctps::CTPS{T}) where T
    ctps_new = CTPS(ctps)
    for i in eachindex(ctps_new.c)
        ctps_new.c[i] = -ctps_new.c[i]
    end
    return ctps_new
end

# * (input-major kernel with maximum reuse)
function *(ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    # Check descriptor compatibility
    if ctps1.desc !== ctps2.desc
        if ctps1.desc.nv != ctps2.desc.nv || ctps1.desc.order != ctps2.desc.order
            error("Cannot multiply CTPS with different descriptors: (nv=$(ctps1.desc.nv), order=$(ctps1.desc.order)) vs (nv=$(ctps2.desc.nv), order=$(ctps2.desc.order))")
        end
    end
    desc = ctps1.desc
    result_c = zeros(T, desc.N)
    
    order = desc.order
    
    c1 = ctps1.c
    c2 = ctps2.c
    
    # Iterate over all degree combinations
    @inbounds for di in 0:order
        for dj in 0:order
            dk = di + dj
            if dk > order
                continue
            end
            
            # Get precomputed input-major schedule for (di, dj)
            schedule_idx = di * (order + 1) + dj + 1
            schedule = desc.mul[schedule_idx]
            
            ivalues = schedule.ivalues
            i_starts = schedule.i_start
            i_ends = schedule.i_end
            jidx = schedule.jidx
            kidx = schedule.kidx
            
            # Process each unique i value
            for i_idx in 1:length(ivalues)
                i = ivalues[i_idx]
                ai = c1[i]  # Load once and reuse!
                
                # Skip if zero (sparse optimization)
                if iszero(ai)
                    continue
                end
                
                range_start = i_starts[i_idx]
                range_end = i_ends[i_idx]
                
                # Inner loop: Use actual k indices from schedule
                @inbounds @simd for pos in range_start:range_end
                    j = jidx[pos]
                    k = kidx[pos]
                    result_c[k] += ai * c2[j]
                end
            end
        end
    end
    
    mask_result = compute_degree_mask(result_c, desc)
    return CTPS{T}(result_c, desc, Ref(mask_result))
end

function *(ctps::CTPS{T}, a::Number) where T
    ctps_new = CTPS(ctps)
    for i in eachindex(ctps_new.c)
        ctps_new.c[i] *= a
    end
    return ctps_new
end

function *(a::Number, ctps::CTPS{T}) where T
    return ctps * a
end

# /
function inv(ctps::CTPS{T}) where T
    if cst(ctps) == zero(T)
        error("Divide by zero in CTPS")
    end
    
    # Preallocate (only allocations in function)
    temp = CTPS(ctps)
    temp.c[1] -= cst(ctps)  # temp = ctps - cst(ctps) in-place
    
    term_by_order = CTPS(T(1.0 / ctps.c[1]), ctps.desc.nv, ctps.desc.order)
    sum = CTPS(term_by_order)
    
    # Preallocate working buffer for multiplication
    term_next = CTPS(T, ctps.desc.nv, ctps.desc.order)
    
    # Precompute -temp / cst(ctps) once
    neg_temp_over_c0 = CTPS(temp)
    scale_factor = -one(T) / cst(ctps)
    scale!(neg_temp_over_c0, scale_factor)
    
    for i in 1:ctps.desc.order
        # term_by_order *= -temp / cst(ctps) using in-place mul!
        mul!(term_next, term_by_order, neg_temp_over_c0)
        copy!(term_by_order, term_next)
        
        # sum += term_by_order (in-place)
        for j in eachindex(sum.c)
            sum.c[j] += term_by_order.c[j]
        end
    end
    return sum
end

function /(ctps1::CTPS{T}, ctps2::CTPS{T}) where T
    if cst(ctps2) == zero(T)
        error("Divide by zero in CTPS")
    end
    return ctps1 * inv(ctps2)
end

function /(ctps::CTPS{T}, a::T) where T
    if a == zero(T)
        error("Divide by zero in CTPS")
    end
    ctps_new = CTPS(ctps)
    for i in eachindex(ctps_new.c)
        ctps_new.c[i] /= a
    end
    return ctps_new
end

function /(a::T, ctps::CTPS{T}) where T
    if cst(ctps) == zero(T)
        error("Divide by zero in CTPS")
    end
    return a * inv(ctps)
end

# exponential (zero loop allocations)
function exp(ctps::CTPS{T}) where T
    a0 = cst(ctps)
    if a0 == zero(T)
        return CTPS(one(T), ctps.desc.nv, ctps.desc.order)
    end
    
    # Preallocate working arrays (only allocations in the function)
    temp = CTPS(ctps)
    temp.c[1] -= a0  # temp = ctps - a0 (in-place)
    
    term = CTPS(one(T), ctps.desc.nv, ctps.desc.order)
    sum = CTPS(one(T), ctps.desc.nv, ctps.desc.order)
    term_next = CTPS(T, ctps.desc.nv, ctps.desc.order)
    
    for i in 1:ctps.desc.order
        # term *= temp (in-place via mul!)
        mul!(term_next, term, temp)
        copy!(term, term_next)
        
        # sum += term / factorial(i) (in-place)
        scale_factor = T(1.0 / factorial(i))
        for j in eachindex(sum.c)
            sum.c[j] += term.c[j] * scale_factor
        end
    end
    
    scale!(sum, T(Base.exp(a0)))
    return sum
end

# logarithm (zero loop allocations)
function log(ctps::CTPS{T}) where T
    a0 = cst(ctps)
    if a0 == zero(T)
        error("Log of zero in CTPS")
    end
    
    # Preallocate (only allocations)
    temp = CTPS(ctps)
    temp.c[1] -= a0
    
    term = temp / a0  # One allocation for initial term
    sum = CTPS(term)
    term_next = CTPS(T, ctps.desc.nv, ctps.desc.order)
    neg_temp_over_a0 = CTPS(temp)
    scale!(neg_temp_over_a0, -one(T) / a0)
    
    for i in 2:ctps.desc.order
        # term *= -temp/a0 (in-place)
        mul!(term_next, term, neg_temp_over_a0)
        copy!(term, term_next)
        
        # sum += term / i (in-place)
        scale_factor = T(1.0 / i)
        for j in eachindex(sum.c)
            sum.c[j] += term.c[j] * scale_factor
        end
    end
    
    sum.c[1] += Base.log(a0)
    return sum
end

# square root (minimal allocations)
function sqrt(ctps::CTPS{T}) where T
    a0_val = cst(ctps)
    if a0_val < zero(T)
        error("Square root of negative number in CTPS")
    end
    a0 = Base.sqrt(a0_val)
    
    # Preallocate
    temp = CTPS(ctps)
    temp.c[1] -= a0_val
    
    term = temp / a0
    sum = CTPS(term)
    scale!(sum, T(0.5))
    
    temp_mul = CTPS(T, ctps.desc.nv, ctps.desc.order)
    neg_temp_over_a0 = CTPS(temp)
    scale!(neg_temp_over_a0, -one(T) / a0_val)
    
    for i in 2:ctps.desc.order
        # term *= -temp / a0_val (use in-place mul! to avoid allocation)
        mul!(temp_mul, term, neg_temp_over_a0)
        copy!(term, temp_mul)
        
        # sum += term * coeff
        coeff = T(doublefactorial(2 * i - 3)) / T(doublefactorial(2 * i))
        for j in eachindex(sum.c)
            sum.c[j] += term.c[j] * coeff
        end
    end
    
    sum.c[1] += a0
    return sum
end

# power
function pow(ctps::CTPS{T}, b::Int) where T
    if b == 1
        return ctps
    elseif b == 0
        return CTPS(one(T), ctps.desc.nv, ctps.desc.order)
    end

    temp = CTPS(ctps)
    index = T(b)
    if cst(ctps) == zero(T)
        if b > 1
            sum = CTPS(ctps)
            for i in 2:b
                sum = sum * ctps
            end
            return sum
        else
            error("Divide by zero, in CTPS::pow")
        end
    end
    temp = temp - cst(temp)
    term_by_order = CTPS(one(T), ctps.desc.nv, ctps.desc.order)
    factor = cst(ctps) ^ b
    sum = CTPS(factor, ctps.desc.nv, ctps.desc.order)

    for i in 1:ctps.desc.order
        factor = factor / cst(ctps) * index / i
        index -= 1.0
        term_by_order = term_by_order * temp
        sum = sum + (term_by_order * factor)
        if index == 0.0
            break
        end
    end
    return sum
end

function ^(ctps::CTPS{T}, b::Int) where T
    return pow(ctps, b)
end

# sin (zero loop allocations)
function sin(ctps::CTPS{T}) where T
    a0 = cst(ctps)
    sin_a0 = Base.sin(a0)
    cos_a0 = Base.cos(a0)
    
    temp = CTPS(ctps)
    temp.c[1] -= a0
    
    term = CTPS(one(T), ctps.desc.nv, ctps.desc.order)
    sum = CTPS(zero(T), ctps.desc.nv, ctps.desc.order)
    term_next = CTPS(T, ctps.desc.nv, ctps.desc.order)

    is_odd = true
    for i in 1:ctps.desc.order
        # term *= temp (in-place)
        mul!(term_next, term, temp)
        copy!(term, term_next)
        
        # Compute coefficient
        coeff = if is_odd
            cos_a0 * T((-1) ^ ((i - 1) ÷ 2)) / T(factorial(i))
        else
            sin_a0 * T((-1) ^ (i ÷ 2)) / T(factorial(i))
        end
        
        # sum += term * coeff (in-place)
        for j in eachindex(sum.c)
            sum.c[j] += term.c[j] * coeff
        end
        
        is_odd = !is_odd
    end
    
    sum.c[1] += sin_a0
    return sum
end

# cos (zero loop allocations)
function cos(ctps::CTPS{T}) where T
    a0 = cst(ctps)
    sin_a0 = Base.sin(a0)
    cos_a0 = Base.cos(a0)
    
    temp = CTPS(ctps)
    temp.c[1] -= a0
    
    term = CTPS(one(T), ctps.desc.nv, ctps.desc.order)
    sum = CTPS(zero(T), ctps.desc.nv, ctps.desc.order)
    term_next = CTPS(T, ctps.desc.nv, ctps.desc.order)

    is_odd = true
    for i in 1:ctps.desc.order
        # term *= temp (in-place)
        mul!(term_next, term, temp)
        copy!(term, term_next)
        
        # Compute coefficient
        coeff = if is_odd
            sin_a0 * T((-1) ^ ((i + 1) ÷ 2)) / T(factorial(i))
        else
            cos_a0 * T((-1) ^ (i ÷ 2)) / T(factorial(i))
        end
        
        # sum += term * coeff (in-place)
        for j in eachindex(sum.c)
            sum.c[j] += term.c[j] * coeff
        end
        
        is_odd = !is_odd
    end
    
    sum.c[1] += cos_a0
    return sum
end
# arcsin
function asin(ctps::CTPS{T}) where T
    temp = CTPS(ctps)
    a0 = cst(ctps)
    arcsin_a0 = asin(a0)
    cos_y0 = sqrt(one(T) - a0 ^ 2)
    temp = temp - a0
    temp1 = CTPS(temp)
    for i in 1:ctps.desc.order + 10
        temp1 = temp1 - sin(temp1) + (temp + a0*(1.0-cos(temp1)))/cos_y0
    end
    return temp1 + arcsin_a0
end

# arccos
function acos(ctps::CTPS{T}) where T
    return T(pi / 2) - asin(ctps)
end

# tangent
function tan(ctps::CTPS{T}) where T
    return sin(ctps) / cos(ctps)
end

# hyperbolic sin (zero loop allocations)
function sinh(ctps::CTPS{T}) where T
    a0 = cst(ctps)
    sinh_a0 = Base.sinh(a0)
    cosh_a0 = Base.cosh(a0)
    
    temp = CTPS(ctps)
    temp.c[1] -= a0
    
    term = CTPS(one(T), ctps.desc.nv, ctps.desc.order)
    sum = CTPS(zero(T), ctps.desc.nv, ctps.desc.order)
    term_next = CTPS(T, ctps.desc.nv, ctps.desc.order)

    is_odd = true
    for i in 1:ctps.desc.order
        # term *= temp (in-place)
        mul!(term_next, term, temp)
        copy!(term, term_next)
        
        coeff = if is_odd
            cosh_a0 / T(factorial(i))
        else
            sinh_a0 / T(factorial(i))
        end
        
        for j in eachindex(sum.c)
            sum.c[j] += term.c[j] * coeff
        end
        
        is_odd = !is_odd
    end
    
    sum.c[1] += sinh_a0
    return sum
end

# hyperbolic cos (zero loop allocations)
function cosh(ctps::CTPS{T}) where T
    a0 = cst(ctps)
    sinh_a0 = Base.sinh(a0)
    cosh_a0 = Base.cosh(a0)
    
    temp = CTPS(ctps)
    temp.c[1] -= a0
    
    term = CTPS(one(T), ctps.desc.nv, ctps.desc.order)
    sum = CTPS(zero(T), ctps.desc.nv, ctps.desc.order)
    term_next = CTPS(T, ctps.desc.nv, ctps.desc.order)

    is_odd = true
    for i in 1:ctps.desc.order
        # term *= temp (in-place)
        mul!(term_next, term, temp)
        copy!(term, term_next)
        
        coeff = if is_odd
            sinh_a0 / T(factorial(i))
        else
            cosh_a0 / T(factorial(i))
        end
        
        for j in eachindex(sum.c)
            sum.c[j] += term.c[j] * coeff
        end
        
        is_odd = !is_odd
    end
    
    sum.c[1] += cosh_a0
    return sum
end
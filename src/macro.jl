# ── @tpsa macro ───────────────────────────────────────────────────────────────
#
# Compiles a CTPS arithmetic assignment into zero-allocation in-place code
# using a PSWorkspace.  Temporaries are borrowed from `ws` and released
# when no longer needed — the minimum number of borrows for the expression.
#
# Usage:
#   @tpsa ws  lhs = expr
#
# `lhs` must be a pre-allocated CTPS (e.g. from CTPS(0.0, i) or borrow!(ws)).
# The result is written directly into `lhs` with no heap allocation.
#
# Supported operations in `expr`:
#   a + b, a - b, a * b, a / b, -a, a^n (integer)
#   sin(a), cos(a), tan(a), exp(a), log(a), sqrt(a), sinh(a), cosh(a), asin(a), acos(a)
#   Scalar (Number) values may appear in +, -, *, / and supported unary calls.
#
# The macro is NOT appropriate for:
#   - Assignments where lhs appears on the rhs (self-referential expressions)
#   - Thread-unsafe concurrent use of the same workspace
#
# Example — 4-variable Henon step rotations with zero allocations:
#   @tpsa ws nx1 = c1*x1 + s1*(x2 + x1^2 - x3^2)
#   @tpsa ws nx2 = c1*(x2 + x1^2 - x3^2) - s1*x1

# ── Runtime dispatch helpers ──────────────────────────────────────────────────
# Called by the generated code; handle scalar/CTPS mixed arguments via dispatch.

@inline function _tpsa_add!(out::CTPS{T}, a::CTPS{T}, b::CTPS{T}) where T
    add!(out, a, b)
end
@inline function _tpsa_add!(out::CTPS{T}, a::CTPS{T}, b::Number) where T
    add!(out, a, T(b))
end
@inline function _tpsa_add!(out::CTPS{T}, a::Number, b::CTPS{T}) where T
    add!(out, b, T(a))
end
@inline function _tpsa_add!(out::CTPS{T}, a::Number, b::Number) where T
    val = T(a) + T(b)
    _zero_active!(out)
    out.c[1] = val
    out.degree_mask[] = _prunable_zero(val) ? UInt64(0) : UInt64(1)
end

@inline function _tpsa_sub!(out::CTPS{T}, a::CTPS{T}, b::CTPS{T}) where T
    sub!(out, a, b)
end
@inline function _tpsa_sub!(out::CTPS{T}, a::CTPS{T}, b::Number) where T
    add!(out, a, -T(b))
end
@inline function _tpsa_sub!(out::CTPS{T}, a::Number, b::CTPS{T}) where T
    # a - b = -(b - a) = scale b by -1, then add scalar a
    scale!(out, b, T(-1))
    m  = out.degree_mask[]
    c0 = (m & UInt64(1) != 0) ? out.c[1] : zero(T)
    out.c[1] = c0 + T(a)
    out.degree_mask[] = (m & ~UInt64(1)) | (_prunable_zero(out.c[1]) ? UInt64(0) : UInt64(1))
end
@inline function _tpsa_sub!(out::CTPS{T}, a::Number, b::Number) where T
    val = T(a) - T(b)
    _zero_active!(out)
    out.c[1] = val
    out.degree_mask[] = _prunable_zero(val) ? UInt64(0) : UInt64(1)
end

@inline function _tpsa_mul!(out::CTPS{T}, a::CTPS{T}, b::CTPS{T}) where T
    mul!(out, a, b)
end
@inline function _tpsa_mul!(out::CTPS{T}, a::CTPS{T}, b::Number) where T
    scale!(out, a, T(b))
end
@inline function _tpsa_mul!(out::CTPS{T}, a::Number, b::CTPS{T}) where T
    scale!(out, b, T(a))
end
@inline function _tpsa_mul!(out::CTPS{T}, a::Number, b::Number) where T
    val = T(a) * T(b)
    _zero_active!(out)
    out.c[1] = val
    out.degree_mask[] = _prunable_zero(val) ? UInt64(0) : UInt64(1)
end

@inline function _tpsa_div!(out::CTPS{T}, a::CTPS{T}, b::CTPS{T}) where T
    div!(out, a, b)
end
@inline function _tpsa_div!(out::CTPS{T}, a::CTPS{T}, b::Number) where T
    scale!(out, a, one(T) / T(b))
end
@inline function _tpsa_div!(out::CTPS{T}, a::Number, b::CTPS{T}) where T
    inv!(out, b)
    scale!(out, T(a))
end
@inline function _tpsa_div!(out::CTPS{T}, a::Number, b::Number) where T
    val = T(a) / T(b)
    _zero_active!(out)
    out.c[1] = val
    out.degree_mask[] = _prunable_zero(val) ? UInt64(0) : UInt64(1)
end

@inline function _tpsa_neg!(out::CTPS{T}, a::CTPS{T}) where T
    scale!(out, a, T(-1))
end
@inline function _tpsa_neg!(out::CTPS{T}, a::Number) where T
    val = -T(a)
    _zero_active!(out)
    out.c[1] = val
    out.degree_mask[] = _prunable_zero(val) ? UInt64(0) : UInt64(1)
end

@inline function _tpsa_pow!(out::CTPS{T}, a::CTPS{T}, b::Int) where T
    pow!(out, a, b)
end
@inline function _tpsa_pow!(out::CTPS{T}, a::Number, b::Int) where T
    val = T(a)^b
    _zero_active!(out)
    out.c[1] = val
    out.degree_mask[] = _prunable_zero(val) ? UInt64(0) : UInt64(1)
end

# Scalar unary calls evaluate in their original scalar type, then store a
# constant polynomial. CTPS arguments keep the existing in-place kernels.
for f in (:sin, :cos, :tan, :exp, :log, :sqrt, :sinh, :cosh, :asin, :acos)
    helper = Symbol("_tpsa_", f, "!")
    kernel = Symbol(f, "!")
    @eval begin
        @inline $helper(out::CTPS{T}, a::CTPS{T}) where T = $kernel(out, a)
        @inline function $helper(out::CTPS{T}, a::Number) where T
            val = T($f(a))
            _zero_active!(out)
            out.c[1] = val
            out.degree_mask[] = _prunable_zero(val) ? UInt64(0) : UInt64(1)
            return out
        end
    end
end

# ── Scalar propagation ────────────────────────────────────────────────────────
#
# An intermediate whose operands are all plain numbers is itself a number: it
# is evaluated here and no workspace slot is borrowed. A slot is taken only
# when a series is involved. This keeps `cos(μ)*x` a scale! rather than a
# series product of a constant, and lets a complex scalar such as `1.0 + 2.0im`
# reach a complex `lhs` intact instead of being forced through a Float64 slot
# (which raised InexactError). Types are concrete at each call site, so the
# choice costs nothing at run time.
@inline _tpsa_scalar(::Val{:+}, a, b) = a + b
@inline _tpsa_scalar(::Val{:-}, a, b) = a - b
@inline _tpsa_scalar(::Val{:*}, a, b) = a * b
@inline _tpsa_scalar(::Val{:/}, a, b) = a / b
@inline _tpsa_scalar(::Val{:^}, a, b) = a ^ b
@inline _tpsa_scalar(::Val{:-}, a)    = -a
for f in (:sin, :cos, :tan, :exp, :log, :sqrt, :sinh, :cosh, :asin, :acos)
    @eval @inline _tpsa_scalar(::Val{$(QuoteNode(f))}, a) = $f(a)
end

# The storage for an intermediate: the evaluated number, or a borrowed slot.
@inline _tpsa_slot(ws, op::Val, a::Number, b::Number) = _tpsa_scalar(op, a, b)
@inline _tpsa_slot(ws, op::Val, a, b)                  = borrow!(ws)
@inline _tpsa_slot(ws, op::Val, a::Number)             = _tpsa_scalar(op, a)
@inline _tpsa_slot(ws, op::Val, a)                     = borrow!(ws)

@inline _tpsa_release!(ws, t::CTPS) = release!(ws, t)
@inline _tpsa_release!(ws, ::Number) = nothing

# A scalar intermediate was already evaluated by _tpsa_slot: nothing to do.
for helper in (:_tpsa_add!, :_tpsa_sub!, :_tpsa_mul!, :_tpsa_div!, :_tpsa_pow!)
    @eval @inline $helper(out::Number, a::Number, b::Number) = out
end
@inline _tpsa_neg!(out::Number, a::Number) = out
for f in (:sin, :cos, :tan, :exp, :log, :sqrt, :sinh, :cosh, :asin, :acos)
    helper = Symbol("_tpsa_", f, "!")
    @eval @inline $helper(out::Number, a::Number) = out
end

# `lhs = leaf`: copy a series, or store a number as a constant polynomial.
@inline _tpsa_assign!(out::CTPS{T}, a::CTPS{T}) where T = copy!(out, a)
@inline function _tpsa_assign!(out::CTPS{T}, a::Number) where T
    val = T(a)
    _zero_active!(out)
    out.c[1] = val
    out.degree_mask[] = _prunable_zero(val) ? UInt64(0) : UInt64(1)
    return out
end

# ── AST helpers ───────────────────────────────────────────────────────────────

# Returns true for expression nodes that should be treated as leaf values
# (user variables, array indices, literals, etc.) rather than TPSA operations.
function _tpsa_is_leaf(ast)
    ast isa Symbol    && return true
    ast isa Number    && return true
    ast isa Bool      && return true
    !(ast isa Expr)   && return true          # QuoteNode, LineNumberNode, etc.
    ast.head == :ref  && return true          # x[1], arr[i]
    ast.head == :.    && return true          # mod.field
    ast.head == :$    && return true          # interpolated value
    return false
end

# Recursively lower a TPSA expression into in-place statements.
#
# Arguments:
#   ast      — the sub-expression to lower
#   ws_sym   — the workspace expression (already esc'd)
#   stmts    — statement list to append generated code to
#   temporaries — maps borrowed symbols to ownership flags (at macro expansion)
#   lhs_sym  — when non-nothing, write the result directly into this expression
#               and return (lhs_sym, false); otherwise borrow a temp and return
#               (temp_sym, true).
#
# Returns: (result_expr, is_borrow::Bool)
#   result_expr  — expression holding the result
#   is_borrow    — true if the caller is responsible for releasing result_expr
function _tpsa_lower_expr(ast, ws_sym, stmts, lhs_sym, temporaries)
    if _tpsa_is_leaf(ast)
        return (esc(ast), false)
    end

    # N-ary + or * → fold left into binary pairs, then lower
    if ast isa Expr && ast.head == :call
        f  = ast.args[1]
        na = length(ast.args) - 1
        if (f == :+ || f == :* || f == :-) && na > 2
            # fold: (a ⊕ b ⊕ c ⊕ d) → ((a ⊕ b) ⊕ c) ⊕ d
            folded = Expr(:call, f, ast.args[2], ast.args[3])
            for i in 4:length(ast.args)
                folded = Expr(:call, f, folded, ast.args[i])
            end
            return _tpsa_lower_expr(folded, ws_sym, stmts, lhs_sym, temporaries)
        end
    end

    if !(ast isa Expr && ast.head == :call)
        # Unknown expr shape: treat as leaf
        return (esc(ast), false)
    end

    f  = ast.args[1]
    na = length(ast.args) - 1

    # Release an operand if it was a borrowed intermediate (a no-op at run time
    # when that intermediate turned out to be a number).
    function maybe_release!(sym, is_tmp)
        if is_tmp
            push!(stmts, :(_tpsa_release!($ws_sym, $sym)))
            push!(stmts, :($(temporaries[sym]) = false))
        end
    end

    # Emit one operation: its storage (lhs, a borrowed slot, or — decided at
    # run time — a plain number), the ownership flag, the in-place call, and
    # the release of any operands that were intermediates.
    function emit(op::Symbol, helper::Symbol, operands, owned)
        if lhs_sym !== nothing
            push!(stmts, :($helper($lhs_sym, $(operands...))))
            for (e, o) in zip(operands, owned)
                maybe_release!(e, o)
            end
            return (lhs_sym, false)
        end
        t    = gensym("tpsa")
        live = gensym("tpsa_live")
        temporaries[t] = live
        push!(stmts, :($t = _tpsa_slot($ws_sym, Val($(QuoteNode(op))), $(operands...))))
        push!(stmts, :($live = true))
        push!(stmts, :($helper($t, $(operands...))))
        for (e, o) in zip(operands, owned)
            maybe_release!(e, o)
        end
        return (t, true)
    end

    lower(arg) = _tpsa_lower_expr(arg, ws_sym, stmts, nothing, temporaries)

    binary = Dict(:+ => :_tpsa_add!, :- => :_tpsa_sub!, :* => :_tpsa_mul!, :/ => :_tpsa_div!)
    unary  = (:sin, :cos, :tan, :exp, :log, :sqrt, :sinh, :cosh, :asin, :acos)

    if na == 2 && haskey(binary, f)
        (ea, ta) = lower(ast.args[2])
        (eb, tb) = lower(ast.args[3])
        return emit(f, binary[f], (ea, eb), (ta, tb))
    elseif f == :- && na == 1
        (ea, ta) = lower(ast.args[2])
        return emit(:-, :_tpsa_neg!, (ea,), (ta,))
    elseif f == :^ && na == 2
        (ea, ta) = lower(ast.args[2])
        n_expr   = esc(ast.args[3])           # exponent: literal or variable, never owned
        return emit(:^, :_tpsa_pow!, (ea, n_expr), (ta, false))
    elseif na == 1 && f in unary
        (ea, ta) = lower(ast.args[2])
        return emit(f, Symbol("_tpsa_", f, "!"), (ea,), (ta,))
    else
        # Unknown function call: treat as an atomic leaf value (evaluated as
        # ordinary, possibly allocating, code).
        return (esc(ast), false)
    end
end

# ── @tpsa macro ───────────────────────────────────────────────────────────────

"""
    @tpsa ws  lhs = expr

Compile a TPSA arithmetic expression into zero-allocation in-place code,
writing the result directly into the pre-allocated CTPS `lhs`.

Temporaries are borrowed from `ws::PSWorkspace` and released automatically
when no longer needed, including when evaluation throws. The output may be
partially written on failure. The number of simultaneous borrows equals the peak
number of live intermediates in `expr`.

# Supported operations
`+`, `-`, `*`, `/`, unary `-`, `^n` (Int), `sin`, `cos`, `tan`, `exp`, `log`,
`sqrt`, `sinh`, `cosh`, `asin`, `acos`. Scalar (`Number`) values may appear as
either operand to `+`, `-`, `*`, `/`, and as arguments to the supported unary
functions; a sub-expression whose operands are all numbers is evaluated as a
number and borrows no slot, so `cos(μ)*x` is a single `scale!` and complex
scalars reach a complex `lhs` intact. Any other call is evaluated as an
ordinary (allocating) expression.

# Example
```julia
desc = set_descriptor!(3, 4)
ws = PSWorkspace(desc, 16)
x1 = CTPS(0.0, 1); x2 = CTPS(0.0, 2); x3 = CTPS(0.0, 3)
nx1 = CTPS(Float64, desc)
μ = 2π * 0.205
@tpsa ws nx1 = cos(μ)*x1 + sin(μ)*(x2 + x1^2 - x3^2)
```
"""
macro tpsa(ws_expr, assign_expr)
    if !(assign_expr isa Expr && assign_expr.head == :(=))
        error("@tpsa: second argument must be an assignment `lhs = rhs`, got: $assign_expr")
    end
    lhs = assign_expr.args[1]
    rhs = assign_expr.args[2]

    ws_sym  = gensym("tpsa_ws")
    lhs_sym = esc(lhs)

    stmts = Expr[]
    temporaries = Dict{Symbol, Symbol}()
    (result, is_borrow) = _tpsa_lower_expr(rhs, ws_sym, stmts, lhs_sym, temporaries)

    # If _tpsa_lower_expr didn't write directly into lhs (shouldn't happen when
    # lhs_sym is passed, but guard just in case):
    if result !== lhs_sym
        push!(stmts, :(_tpsa_assign!($lhs_sym, $result)))
        if is_borrow
            push!(stmts, :(_tpsa_release!($ws_sym, $result)))
            push!(stmts, :($(temporaries[result]) = false))
        end
    end

    # Track ownership with local booleans, not a runtime collection. Early
    # releases retain the normal peak slot count; finally releases only slots
    # still owned by this invocation, including after a failed borrow or call.
    initializers = [:(local $live = false) for live in values(temporaries)]
    declarations = [:(local $temp) for temp in keys(temporaries)]
    cleanup = [:($live && _tpsa_release!($ws_sym, $temp)) for (temp, live) in temporaries]
    return quote
        local $ws_sym = $(esc(ws_expr))
        $(declarations...)
        $(initializers...)
        try
            $(stmts...)
        finally
            $(cleanup...)
        end
    end
end

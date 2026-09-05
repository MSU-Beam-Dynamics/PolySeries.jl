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
#   a + b, a - b, a * b, -a, a^n (integer)
#   sin(a), cos(a), exp(a), log(a), sqrt(a), sinh(a), cosh(a)
#   Scalar (Real) values may appear in +, -, * and supported unary calls.
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
@inline function _tpsa_add!(out::CTPS{T}, a::CTPS{T}, b::Real) where T
    add!(out, a, T(b))
end
@inline function _tpsa_add!(out::CTPS{T}, a::Real, b::CTPS{T}) where T
    add!(out, b, T(a))
end
@inline function _tpsa_add!(out::CTPS{T}, a::Real, b::Real) where T
    val = T(a) + T(b)
    _zero_active!(out)
    out.c[1] = val
    out.degree_mask[] = _prunable_zero(val) ? UInt64(0) : UInt64(1)
end

@inline function _tpsa_sub!(out::CTPS{T}, a::CTPS{T}, b::CTPS{T}) where T
    sub!(out, a, b)
end
@inline function _tpsa_sub!(out::CTPS{T}, a::CTPS{T}, b::Real) where T
    add!(out, a, -T(b))
end
@inline function _tpsa_sub!(out::CTPS{T}, a::Real, b::CTPS{T}) where T
    # a - b = -(b - a) = scale b by -1, then add scalar a
    scale!(out, b, T(-1))
    m  = out.degree_mask[]
    c0 = (m & UInt64(1) != 0) ? out.c[1] : zero(T)
    out.c[1] = c0 + T(a)
    out.degree_mask[] = (m & ~UInt64(1)) | (_prunable_zero(out.c[1]) ? UInt64(0) : UInt64(1))
end
@inline function _tpsa_sub!(out::CTPS{T}, a::Real, b::Real) where T
    val = T(a) - T(b)
    _zero_active!(out)
    out.c[1] = val
    out.degree_mask[] = _prunable_zero(val) ? UInt64(0) : UInt64(1)
end

@inline function _tpsa_mul!(out::CTPS{T}, a::CTPS{T}, b::CTPS{T}) where T
    mul!(out, a, b)
end
@inline function _tpsa_mul!(out::CTPS{T}, a::CTPS{T}, b::Real) where T
    scale!(out, a, T(b))
end
@inline function _tpsa_mul!(out::CTPS{T}, a::Real, b::CTPS{T}) where T
    scale!(out, b, T(a))
end
@inline function _tpsa_mul!(out::CTPS{T}, a::Real, b::Real) where T
    val = T(a) * T(b)
    _zero_active!(out)
    out.c[1] = val
    out.degree_mask[] = _prunable_zero(val) ? UInt64(0) : UInt64(1)
end

@inline function _tpsa_neg!(out::CTPS{T}, a::CTPS{T}) where T
    scale!(out, a, T(-1))
end
@inline function _tpsa_neg!(out::CTPS{T}, a::Real) where T
    val = -T(a)
    _zero_active!(out)
    out.c[1] = val
    out.degree_mask[] = _prunable_zero(val) ? UInt64(0) : UInt64(1)
end

@inline function _tpsa_pow!(out::CTPS{T}, a::CTPS{T}, b::Int) where T
    pow!(out, a, b)
end
@inline function _tpsa_pow!(out::CTPS{T}, a::Real, b::Int) where T
    val = T(a)^b
    _zero_active!(out)
    out.c[1] = val
    out.degree_mask[] = _prunable_zero(val) ? UInt64(0) : UInt64(1)
end

# Scalar unary calls evaluate in their original scalar type, then store a
# constant polynomial. CTPS arguments keep the existing in-place kernels.
for f in (:sin, :cos, :exp, :log, :sqrt, :sinh, :cosh)
    helper = Symbol("_tpsa_", f, "!")
    kernel = Symbol(f, "!")
    @eval begin
        @inline $helper(out::CTPS{T}, a::CTPS{T}) where T = $kernel(out, a)
        @inline function $helper(out::CTPS{T}, a::Real) where T
            val = T($f(a))
            _zero_active!(out)
            out.c[1] = val
            out.degree_mask[] = _prunable_zero(val) ? UInt64(0) : UInt64(1)
            return out
        end
    end
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

    # Helper: allocate the output slot (lhs or a new borrow)
    function get_out()
        if lhs_sym !== nothing
            return (lhs_sym, false)
        else
            t = gensym("tpsa")
            live = gensym("tpsa_live")
            temporaries[t] = live
            push!(stmts, :($t = borrow!($ws_sym)))
            push!(stmts, :($live = true))
            return (t, true)
        end
    end

    # Helper: release a result if it was borrowed
    function maybe_release!(sym, is_tmp)
        if is_tmp
            push!(stmts, :(release!($ws_sym, $sym)))
            push!(stmts, :($(temporaries[sym]) = false))
        end
    end

    if f == :+ && na == 2
        (ea, ta) = _tpsa_lower_expr(ast.args[2], ws_sym, stmts, nothing, temporaries)
        (eb, tb) = _tpsa_lower_expr(ast.args[3], ws_sym, stmts, nothing, temporaries)
        (out, tout) = get_out()
        push!(stmts, :(_tpsa_add!($out, $ea, $eb)))
        maybe_release!(ea, ta);  maybe_release!(eb, tb)
        return (out, tout)

    elseif f == :- && na == 2
        (ea, ta) = _tpsa_lower_expr(ast.args[2], ws_sym, stmts, nothing, temporaries)
        (eb, tb) = _tpsa_lower_expr(ast.args[3], ws_sym, stmts, nothing, temporaries)
        (out, tout) = get_out()
        push!(stmts, :(_tpsa_sub!($out, $ea, $eb)))
        maybe_release!(ea, ta);  maybe_release!(eb, tb)
        return (out, tout)

    elseif f == :- && na == 1
        (ea, ta) = _tpsa_lower_expr(ast.args[2], ws_sym, stmts, nothing, temporaries)
        (out, tout) = get_out()
        push!(stmts, :(_tpsa_neg!($out, $ea)))
        maybe_release!(ea, ta)
        return (out, tout)

    elseif f == :* && na == 2
        (ea, ta) = _tpsa_lower_expr(ast.args[2], ws_sym, stmts, nothing, temporaries)
        (eb, tb) = _tpsa_lower_expr(ast.args[3], ws_sym, stmts, nothing, temporaries)
        (out, tout) = get_out()
        push!(stmts, :(_tpsa_mul!($out, $ea, $eb)))
        maybe_release!(ea, ta);  maybe_release!(eb, tb)
        return (out, tout)

    elseif f == :^ && na == 2
        (ea, ta) = _tpsa_lower_expr(ast.args[2], ws_sym, stmts, nothing, temporaries)
        n_expr   = esc(ast.args[3])   # exponent: can be literal or variable
        (out, tout) = get_out()
        push!(stmts, :(_tpsa_pow!($out, $ea, $n_expr)))
        maybe_release!(ea, ta)
        return (out, tout)

    elseif na == 1 && f in (:sin, :cos, :exp, :log, :sqrt, :sinh, :cosh)
        (ea, ta) = _tpsa_lower_expr(ast.args[2], ws_sym, stmts, nothing, temporaries)
        f_bang   = Symbol("_tpsa_", f, "!")
        (out, tout) = get_out()
        push!(stmts, :($f_bang($out, $ea)))
        maybe_release!(ea, ta)
        return (out, tout)

    else
        # Unknown function call: treat as atomic leaf value
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
`+`, `-`, `*`, unary `-`, `^n` (Int), `sin`, `cos`, `exp`, `log`, `sqrt`,
`sinh`, `cosh`.  Scalar (Real) values may appear as either operand to `+`,
`-`, `*`, and as arguments to the supported unary functions.

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
        push!(stmts, :(copy!($lhs_sym, $result)))
        if is_borrow
            push!(stmts, :(release!($ws_sym, $result)))
            push!(stmts, :($(temporaries[result]) = false))
        end
    end

    # Track ownership with local booleans, not a runtime collection. Early
    # releases retain the normal peak slot count; finally releases only slots
    # still owned by this invocation, including after a failed borrow or call.
    initializers = [:(local $live = false) for live in values(temporaries)]
    declarations = [:(local $temp) for temp in keys(temporaries)]
    cleanup = [:($live && release!($ws_sym, $temp)) for (temp, live) in temporaries]
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

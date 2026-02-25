# TPSA.jl Documentation

Official package documentation for TPSA.jl (Truncated Power Series Algebra).

## Documentation Files

### Main Documentation
- **[index.md](index.md)**: Complete user guide covering installation, basic concepts, API overview, and examples
- **[api.md](api.md)**: Detailed API reference with all types, functions, and methods

## Online Documentation

[Link to online documentation - to be added]

## Building Documentation Locally

If you want to build the documentation locally:

```julia
using Pkg
Pkg.activate("docs")
Pkg.instantiate()
include("docs/make.jl")
```

## Quick Links

- **Getting Started**: See [index.md](index.md#quick-start)
- **API Reference**: See [api.md](api.md)
- **Examples**: See `../examples/` directory
- **Tests**: See `../test/` directory

## Topics Covered

1. **Installation and Setup**
2. **Basic Concepts** - Understanding TPSA, variables, and descriptors
3. **Core Types** - CTPS, TPSADesc, PolyMap
4. **Operations** - Arithmetic, comparison, mathematical functions
5. **Examples** - Practical usage examples
6. **Performance** - Tips for efficient usage
7. **Advanced Topics** - Index mapping, schedules, composition

## For Package Users

Start with [index.md](index.md) for a comprehensive introduction to TPSA.jl.

## For Developers

Development documentation can be found in:
- **`../agent/`**: Codebase architecture summary
- **`../dev_scripts/`**: Development notes and implementation history
- **`../benchmarks/`**: Performance analysis and comparisons

## Contributing to Documentation

To improve documentation:
1. Edit the markdown files in this directory
2. Submit a pull request
3. Ensure examples are tested and working

Documentation should be:
- Clear and concise
- Well-organized with proper headers
- Include working code examples
- Focus on user needs, not implementation details

---

*For questions about the documentation, please open an issue on GitHub.*

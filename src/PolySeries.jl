# ==============================================================================
# This file is part of the PolySeries.jl Julia package.
#
# Author: Jinyu Wan, initial version
#          Yue Hao and Kelly Anderson
# Version: 1.0
# Created Date: 11-01-2023
# Modified Date: 03-01-2026


module PolySeries
using StaticArrays
include("mathfunc.jl")
include("polymap.jl")
include("ctps.jl")
export CTPS, PSDesc, pow, cst, element, findindex # assign!, reassign!
export add!, addto!, sub!, subfrom!, scale!, scaleadd!, copy!, zero!, mul!
export set_descriptor!, get_descriptor, clear_descriptor!
export PSWorkspace, borrow!, release!
export sin!, cos!, exp!, log!, sqrt!, sinh!, cosh!, asin!, acos!
export pow!
include("macro.jl")
export @tpsa

end
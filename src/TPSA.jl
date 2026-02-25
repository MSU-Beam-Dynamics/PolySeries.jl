# ==============================================================================
# This file is part of the TPSA (Truncated Power Series Algebra) Julia package.
#
# Author: Jinyu Wan
# Email: wan@frib.msu.edu
# Version: 1.0
# Created Date: 11-01-2023
# Modified Date: 11-13-2023


module TPSA
include("mathfunc.jl")
include("polymap.jl")
include("ctps.jl")
using StaticArrays
export CTPS, TPSADesc, pow, cst, element, findindex, assign!, reassign!
export add!, addto!, sub!, subfrom!, scale!, copy!, zero!, mul!
export set_descriptor!, get_descriptor, clear_descriptor!

end
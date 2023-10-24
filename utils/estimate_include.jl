include("../estimate/estimate_MICP.jl")
# include("../estimate/estimate_MICP_noconic.jl")
# include("../estimate/estimate_MICP_quartile.jl")
# include("../estimate/estimate_MICP_lazyconstr.jl")
include("../estimate/estimate_MICP_polyapprox.jl")
include("../estimate/estimate_MICP_IGA.jl")
include("../estimate/estimate_MICP_IGA_polyapprox.jl")
# include("../estimate/estimate_MICP_IGA_polyapprox_flexible.jl")
include("../estimate/estimate_MICP_socapprox.jl")
include("../estimate/estimate_MICP_lpm.jl")
include("../estimate/estimate_MICP_maxcount.jl")



include("initial.jl")
include("update_list.jl")
include("export_table.jl")
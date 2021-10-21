module UpdatableQRFactorizations

using LinearAlgebra
using LinearAlgebra: Givens

export AbstractQR, GivensQ, GivensQR, UpdatableGivensQR, UpdatableQR, UQR,
        add_column!, remove_column!

const AbstractMatOrFac{T} = Union{AbstractMatrix{T}, Factorization{T}}

include("abstract.jl")
include("givensQ.jl")
include("givensQR.jl")
include("updatableQR.jl")

end # module

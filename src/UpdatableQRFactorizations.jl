module UpdatableQRFactorizations

using LinearAlgebra
using LinearAlgebra: Givens, AbstractQ

export AbstractQR, GivensQ, GivensQR, UpdatableGivensQR, UpdatableQR, UQR,
        add_column!, remove_column!

const AbstractMatOrFac{T} = Union{AbstractMatrix{T}, Factorization{T}}
const AdjointQ = isdefined(LinearAlgebra, :AdjointQ) ? LinearAlgebra.AdjointQ : Adjoint

# IDEA: could have efficient version for sparse matrices
include("abstract.jl")
include("givensQ.jl")
include("givensQR.jl")
include("updatableQR.jl")

end # module

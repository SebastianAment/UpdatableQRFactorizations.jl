# Givens representation of Q matrix, memory efficient compared to dense representation
struct GivensQ{T, QT<:AbstractVector{<:Givens{T}}} <: Factorization{T}
    rotations::QT
    n::Int
    m::Int
end

# r controls how many columns of Q we instantiate
function Base.Matrix(F::GivensQ, r::Int = size(F, 2))
    n, m = size(F)
    Q = Matrix(one(eltype(F))*I, n, r)
    return lmul!(F, Q)
end
Base.size(F::GivensQ) = (F.n, F.n) # could be square but also (n, m)?
Base.size(F::GivensQ, i::Int) = i > 2 ? 1 : size(F)[i]
Base.copy(F::GivensQ) = GivensQ(copy(F.rotations), F.n, F.m)
Base.adjoint(F::GivensQ) = Adjoint(F)

# IDEA: can have special version that adapts to size of x (only applies relevant rotations)
function LinearAlgebra.lmul!(F::GivensQ, X::AbstractVecOrMat)
    m = size(X, 1)
    for G in Iterators.reverse(F.rotations) # the adjoint of a GivensQ reverses the order of the rotations
        lmul!(G', X) # ... and takes their adjoint
    end
    return X
end
function LinearAlgebra.lmul!(A::Adjoint{<:Any, <:GivensQ}, X::AbstractVecOrMat)
    F = A.parent
    for G in F.rotations # the adjoint of a GivensQ reverses the order of the rotations
        lmul!(G, X) # ... and takes their adjoint
    end
    return X
end
LinearAlgebra.rmul!(X::AbstractVecOrMat, F::GivensQ) = lmul!(F', X')'
LinearAlgebra.rmul!(X::AbstractVecOrMat, F::Adjoint{<:Any, <:GivensQ}) = lmul!(F', X')'
# IDEA: 5-arg mul?
# function LinearAlgebra.mul!(y::AbstractVector, F::GivensQ, x::AbstractVector)
#     @. y = x
#     lmul!(F, y)
# end

lmul(F, X) = lmul!(F, copy(X))
rmul(X, F) = rmul!(copy(X), F)

Base.:*(F::GivensQ, X::AbstractVector) = lmul(F, X)
Base.:*(F::GivensQ, X::AbstractMatrix) = lmul(F, X)
Base.:*(F::Adjoint{<:Any, <:GivensQ}, X::AbstractVector) = lmul(F, X)
Base.:*(F::Adjoint{<:Any, <:GivensQ}, X::AbstractMatrix) = lmul(F, X)

Base.:*(X::AbstractMatrix, F::GivensQ) = rmul(X, F)
Base.:*(X::AbstractMatrix, F::Adjoint{<:Any, <:GivensQ}) = rmul(X, F)

function Base.:*(F::GivensQ, G::Adjoint{<:Any, <:GivensQ})
    T = promote_type(eltype(F), eltype(G))
    F === G' ? (one(T) * I)(F.n) : F * Matrix(G)
end
function Base.:*(F::Adjoint{<:Any, <:GivensQ}, G::GivensQ)
    T = promote_type(eltype(F), eltype(G))
    F' === G ? (one(T) * I)(G.n) : F * Matrix(G)
end
function Base.:(==)(F::GivensQ, G::GivensQ)
    F.rotations == G.rotation && F.n == G.n && F.m == G.m
end

LinearAlgebra.ldiv!(F::GivensQ, x::AbstractVector) = lmul!(F', x)
LinearAlgebra.ldiv!(F::Adjoint{<:Any, <:GivensQ}, x::AbstractVector) = lmul!(F', x)
LinearAlgebra.ldiv!(F::GivensQ, X::AbstractMatrix) = lmul!(F', X)
LinearAlgebra.ldiv!(F::Adjoint{<:Any, <:GivensQ}, X::AbstractMatrix) = lmul!(F', X)

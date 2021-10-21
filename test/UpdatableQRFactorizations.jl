module TestUpdatableQRFactorizations
using LinearAlgebra
using Test
using UpdatableQRFactorizations

@testset "UpdatableQRFactorizations" begin

    n, m = 32, 8
    A = randn(n, m)
    F = GivensQR(A)

    @testset "GivensQ" begin
        Q = Matrix(F.Q)

        @test Q'Q ≈ I(n)
        @test Q*Q' ≈ I(n)

        @test F.Q'*F.Q == I(n)
        @test F.Q'*F.Q == I(n)

        x = randn(n)
        Qx = Q*x
        @test Qx ≈ F.Q*x
        @test F.Q \ Qx ≈ x
        Qtx = Q'*x
        @test Qtx ≈ F.Q'*x
        @test F.Q' \ Qtx ≈ x

        r = 4
        X = randn(r, n)
        @test Q*X' ≈ F.Q*X'
        @test X*Q ≈ X*F.Q
        @test Q'*X' ≈ F.Q'*X'
        @test X*Q' ≈ X*F.Q'
        @test Q \ (Q*X') ≈ X'
        @test Q' \ (Q'*X') ≈ X'
    end

    @testset "GivensQR" begin
        QA = F.Q'*A
        @test UpperTriangular(QA[1:m, 1:m]) ≈ F.R
        @test Matrix(F) ≈ A

        x = randn(m)
        Ax = A*x
        @test F \ (A*x) ≈ x
        r = 4
        X = randn(m, r)
        @test F \ (A*X) ≈ X
    end

    @testset "UpdatableQR" begin
        F = UpdatableQR(A)
        @test Matrix(F) ≈ A

        x = randn(m)
        Ax = A*x
        @test F \ (A*x) ≈ x
        r = 4
        X = randn(m, r)
        @test F \ (A*X) ≈ X

        # testing column addition
        F = UpdatableQR(A)
        b = randn(n)
        add_column!(F, b)
        @test Matrix(F) ≈ [A b]

        # testing column addition at particular index
        x = randn(m+1)
        X = randn(m+1, r)
        for i in 1:m
            F = UpdatableQR(A)
            add_column!(F, b, i)
            B = [A[:, 1:i-1] b A[:, i:end]]
            @test B ≈ Matrix(F)
            Q = Matrix(F.Q)
            @test Q'Q ≈ I(n) # maintains orthogonality

            # testing solves
            @test F \ (B*x) ≈ x
            r = 4
            @test F \ (B*X) ≈ X
        end

        # testing column addition
        F = UpdatableQR(A)
        remove_column!(F)
        @test Matrix(F) ≈ A[:, 1:m-1]

        # testing column removal
        for i in 1:m
            F = UpdatableQR(A)
            remove_column!(F, i)
            B = @view A[:, 1:m .!= i]
            @test B ≈ Matrix(F)
            Q = Matrix(F.Q)
            @test Q'Q ≈ I(n) # maintains orthogonality
        end

        # testing empty initialization
        F = UpdatableQR(n, m)
        @test size(F) == (n, 0)
        @test Matrix(F) ≈ zeros(n, 0)

        order = reverse(1:m) # adding columns of A in reverse order (but to front of factorization)
        for i in order
            add_column!(F, A[:, i], 1)
        end
        @test Matrix(F) ≈ A
    end

    # multiply is very efficient!
    # @time Q*x
    # @time Q*x
    # @time F.Q*x
    # @time F.Q*x

    # IDEA: testing overdetermined system, irrelevant for now
    # n, m = 128, 64
    # A = randn(m, n)
    # @time F = GivensQR(A)

    # A = randn(n, m)
    # rotations = allocate_rotations(A)
    # F = GivensQR()
    # for i in 1:m
    #     Ai = @view A[:, 1:i]
    #     append_column!()
    # end
end

end # module

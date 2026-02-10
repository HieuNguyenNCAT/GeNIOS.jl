@setup_workload begin
    Random.seed!(1)
    m, n = 20, 40
    A = randn(m, n)
    A .-= sum(A, dims=1) ./ m
    normalize!.(eachcol(A))
    xstar = sprandn(n, 0.1)
    b = A*xstar + 1e-3*randn(m)
    λ = 0.05*norm(A'*b, Inf)

    ## MLSolver interface
    f(x) = 0.5*x^2 
    fconj(x) = 0.5*x^2


    ## QPSolver interface
    P = blockdiag(sparse(A'*A), spzeros(n, n))
    q = vcat(-A'*b, λ*ones(n))
    M = [
        sparse(I, n, n)     sparse(I, n, n);
        sparse(I, n, n)     -sparse(I, n, n)
    ]
    l = [zeros(n); -Inf*ones(n)]
    u = [Inf*ones(n); zeros(n)]


    ## Generic interface
    struct HessianLasso{T, S <: AbstractMatrix{T}} <: HessianOperator
        A::S
        vm::Vector{T}
    end
    function LinearAlgebra.mul!(y, H::HessianLasso, x)
        mul!(H.vm, H.A, x)
        mul!(y, H.A', H.vm)
        return nothing
    end

    function update!(::HessianLasso, ::Solver)
        return nothing
    end

    params = (; A=A, b=b, tmp=zeros(m), λ=λ)
    function f(x, p)
        A, b, tmp = p.A, p.b, p.tmp
        mul!(tmp, A, x)
        @. tmp -= b
        return 0.5 * sum(w->w^2, tmp)
    end

    function grad_f!(g, x, p)
        A, b, tmp = p.A, p.b, p.tmp
        mul!(tmp, A, x)
        @. tmp -= b
        mul!(g, A', tmp)
        return nothing
    end

    Hf = HessianLasso(A, zeros(m))
    g(z, p) = p.λ*sum(x->abs(x), z)

    function prox_g!(v, z, ρ, p)
        λ = p.λ
        @inline soft_threshold(x::T, κ::T) where {T <: Real} = sign(x) * max(zero(T), abs(x) - κ)
        v .= soft_threshold.(z, λ/ρ)
    end

    # Finally, we can solve the problem.
    @compile_workload begin 

    end
end

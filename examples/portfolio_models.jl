# J assets whose prices follow geometric Brownian motions, observed
# at times t = 0, ..., T. Holdings u are committed before the return
# R is seen, and wealth moves as x' = x + R ⋅ u. The terminal loss is
# exp(-ρ x_T) / ρ, or a few of its tangents floored at 0. The model
# has no recourse decision.

using SDDP
import LinearAlgebra
using FastGaussQuadrature: gausshermite

const MU = [0.08]        # drift per year
const SIGMA = [0.2]      # volatility per year
const CORR = ones(1, 1)  # correlation of the Brownian motions
const DT = 0.25          # years per period
const RHO = 1.0          # risk aversion
const WEALTH_0 = 0.0     # initial wealth w
const U_MAX = 20.0       # box on holdings, never binding at the optimum
"Points u_i where the loss exp(ρu)/ρ is replaced by its tangent."
const TANGENTS = [-1.0, -0.5, 0.0, 0.5, 1.0]

"Gauss-Hermite returns and probabilities of one period, with n nodes per assset."
function gbm_returns(
    n::Int;
    μ::Vector{Float64} = MU,
    σ::Vector{Float64} = SIGMA,
    C::Matrix{Float64} = CORR,
    Δt::Float64 = DT,
)
    # N(0, 1) nodes z with probability p.
    z, p = gausshermite(n; normalize = true)
    L = LinearAlgebra.cholesky(C).L
    Ω, P = Vector{Float64}[], Float64[]
    # Looping over the tensor grid.
    for idx in Iterators.product(ntuple(_ -> 1:n, length(μ))...)
        Z = L * [z[i] for i in idx] # Correlation.
        # R_j per asset.
        push!(Ω, @. exp((μ - σ^2 / 2) * Δt + σ * sqrt(Δt) * Z) - 1)
        push!(P, prod(p[i] for i in idx))
    end
    return Ω, P
end

"Minimize E[exp(-ρ R ⋅ U)] by Newton's method."
function one_period_optimum(
    Ω::Vector{Vector{Float64}},
    P::Vector{Float64};
    ρ::Float64 = RHO
)
    f(U) = sum(p * exp(-ρ * (R' * U)) for (R, p) in zip(Ω, P))
    U = zeros(length(first(Ω)))
    for _ in 1:50
        # Each scenario's weighted contribution.
        e = [p * exp(-ρ * (R' * U)) for (R, p) in zip(Ω, P)]
        g = -ρ * sum(e .* Ω) # Gradient.
        if LinearAlgebra.norm(g) < 1e-12
            return sum(e), U # Return the mimimum and the minimizer.
        end
        H = ρ^2 * sum(e[k] * Ω[k] * Ω[k]' for k in eachindex(Ω)) # Hessian.
        Δ = -(H \ g) # Solution of the linear system H Δ = g.
        s = 1.0
        while f(U + s * Δ) > (1 + 1e-14) * sum(e)
            s /= 2 # Linesearch.
        end
        U += s * Δ
    end
    return error("Newton's method did not converge.")
end

"Optimal value of the T-period problem whose one-period minimum is m."
function exact_value(T::Int, m::Float64; ρ::Float64 = RHO, w::Float64 = WEALTH_0)
    return m^T * exp(-ρ * w) / ρ
end

"Tangents of exp(ρ u) / ρ at the points `tangents`, floored at 0."
function tangent_loss(u::Float64; ρ::Float64 = RHO, tangents::Vector{Float64} = TANGENTS)
    return max(0.0, maximum(exp(ρ * a) * (1 / ρ + u - a) for a in tangents))
end

"Kinks of tangent loss: the zero of each tangent and where two tangents cross."
function tangent_kinks(; ρ::Float64 = RHO, tangents::Vector{Float64} = TANGENTS)
    kinks = [a - 1 / ρ for a in tangents] # Crossing zero.
    for a in tangents, b in tangents
        if a < b
            ea, eb = exp(ρ * a), exp(ρ * b)
            push!(kinks, (b * eb - a * ea) / (eb - ea) - 1 / ρ) # Pairwise crossing.
        end
    end
    return kinks
end

"Minimize E[tangent_loss(-X - R U)] over |U| <= U_max for one asset."
function one_period_lp_optimum(
    Ω::Vector{Vector{Float64}},
    P::Vector{Float64},
    X::Float64;
    ρ::Float64 = RHO,
    tangents::Vector{Float64} = TANGENTS,
)
    # Convex and piecewise linear on [-U_MAX, U_MAX].
    f(U) = sum(p * tangent_loss(-X - only(R) * U; ρ, tangents) for (R, p) in zip(Ω, P))
    candidates = [-U_MAX, U_MAX]
    for b in tangent_kinks(; ρ, tangents), R in Ω
        U = -(X + b) / only(R) # Hitting a kink `b``.
        if abs(U) <= U_MAX
            push!(candidates, U)
        end
    end
    value, i = findmin(f, candidates)
    return value, candidates[i]
end

# J assets whose prices follow geometric Brownian motions, observed
# at times t = 0, ..., T. Holdings u are committed before the return
# R is seen, and wealth moves as x' = x + R ⋅ u. The terminal loss is
# exp(-ρ x_T) / ρ, or a few of its tangents floored at 0. The model
# has no recourse decision.

using SDDP
import LinearAlgebra
using FastGaussQuadrature: gausshermite
import HiGHS

const MU = [0.08]        # drift per year
const SIGMA = [0.2]      # volatility per year
const CORR = ones(1, 1)  # correlation of the Brownian motions
const DT = 0.25          # years per period
const RHO = 1.0          # risk aversion
const WEALTH_0 = 0.0     # initial wealth w
const U_MAX = 20.0       # box on holdings, never binding at the optimum
"Points u_i where the loss exp(ρu)/ρ is replaced by its tangent."
const TANGENTS = [-1.0, -0.5, 0.0, 0.5, 1.0]

"Gauss-Hermite returns and probabilities of one period, with n nodes per asset."
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
        U = -(X + b) / only(R) # Hitting a kink `b`.
        if abs(U) <= U_MAX
            push!(candidates, U)
        end
    end
    value, i = findmin(f, candidates)
    return value, candidates[i]
end

"Add the terminal loss of x.out as a stage cost at node T."
function _terminal_cost!(sp, x, ρ, tangents)
    if tangents === nothing # Exponential loss.
        @variable(sp, z) # z ≥ exp(-ρ X_T) / ρ
        @constraint(sp, [-ρ * x.out, 1.0, ρ * z] in MOI.ExponentialCone())
    else # Tangent loss.
        @variable(sp, z >= 0)
        for a in tangents
            @constraint(sp, z >= exp(ρ * a) * (1 / ρ - x.out - a))
        end
    end
    @stageobjective(sp, z)
    return
end

"DHD model in which node t commits the holdings u before the return R of t is seen."
function portfolio_dhd(;
    T::Int = 3,
    n::Int = 5,
    returns = gbm_returns(n),
    ρ::Float64 = RHO,
    w::Float64 = WEALTH_0,
    tangents::Union{Nothing,Vector{Float64}} = TANGENTS,
    optimizer = HiGHS.Optimizer
)
    Ω, P = returns
    J = length(first(Ω))
    return SDDP.LinearPolicyGraph(;
        stages = T,
        sense = :Min,
        lower_bound = 0.0,
        optimizer,
    ) do sp, t
        @variable(sp, x, SDDP.State, initial_value = w)
        @variable(sp, -U_MAX <= u[1:J] <= U_MAX, SDDP.HereAndNow)
        @constraint(sp, wealth, x.out == x.in + sum(u))
        SDDP.parameterize(sp, Ω, P) do R
            for j in 1:J
                # x.out - x.in - ∑_j R_j u_j = 0.
                set_normalized_coefficient(wealth, u[j], -R[j])
            end
            return
        end
        # The loss is charged once at the end.
        if t == T
            _terminal_cost!(sp, x, ρ, tangents)
        else
            @stageobjective(sp, 0.0)
        end
        return
    end
end

"DH-aug model: the holdings u are a state variable commited one node before."
function portfolio_dh_aug(;
    T::Int = 3,
    n::Int = 5,
    returns = gbm_returns(n),
    ρ::Float64 = RHO,
    w::Float64 = WEALTH_0,
    tangents::Union{Nothing,Vector{Float64}} = TANGENTS,
    optimizer = HiGHS.Optimizer,
)
    Ω, P = returns
    J = length(first(Ω))
    return SDDP.LinearPolicyGraph(;
        stages = T + 1,
        sense = :Min,
        lower_bound = 0.0,
        optimizer,
    ) do sp, node
        @variable(sp, x, SDDP.State, initial_value = w)
        @variable(sp, -U_MAX <= u[1:J] <= U_MAX, SDDP.State, initial_value = 0.0)
        if node == 1
            @constraint(sp, x.out == x.in)
            @stageobjective(sp, 0.0)
        else
            @constraint(sp, wealth, x.out == x.in + sum(u[j].in for j in 1:J))
            SDDP.parameterize(sp, Ω, P) do R
                for j in 1:J
                    set_normalized_coefficient(wealth, u[j].in, -R[j])
                end
                return
            end
            if node == T + 1
                _terminal_cost!(sp, x, ρ, tangents)
            else
                @stageobjective(sp, 0.0)
            end
        end
        return
    end
end

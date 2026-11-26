module TestDHDValidation

using SDDP
using Test
import HiGHS

module Hydro
include(joinpath(@__DIR__, "..", "..", "examples", "hydro_models.jl"))
end

module Batteries
include(joinpath(@__DIR__, "..", "..", "examples", "batteries_models.jl"))
end

module Portfolio
include(joinpath(@__DIR__, "..", "..", "examples", "portfolio_models.jl"))
end

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

"Exact optimum of `model` from its deterministic equivalent."
function _exact(model)
    det = SDDP.deterministic_equivalent(model, HiGHS.Optimizer)
    set_silent(det)
    optimize!(det)
    return objective_value(det)
end

"Expected cost of the policy in `model` over every path of its acyclic graph."
function _policy_cost(model::SDDP.PolicyGraph{T}) where {T}
    return _path_cost(model, model.root_children, Tuple{T,Any}[], 1.0)
end

"Expected cost of the paths that continue `path`, which has `probability`."
function _path_cost(model, children, path, probability)
    total = 0.0
    for child in children
        node = model[child.term]
        for noise in node.noise_terms
            p = probability * child.probability * noise.probability
            next_path = vcat(path, [(child.term, noise.term)])
            if isempty(node.children)
                scheme = SDDP.Historical(next_path)
                sim = only(SDDP.simulate(model, 1; sampling_scheme = scheme))
                total += p * sum(s[:stage_objective] for s in sim)
            else
                total += _path_cost(model, node.children, next_path, p)
            end
        end
    end
    return total
end

"Check that `dhd()` matches `reference()` and that training finds the optimum."
function _validate(dhd, reference; iteration_limit, kwargs...)
    exact = _exact(reference())
    @test _exact(dhd()) ≈ exact rtol = 1e-6
    model = dhd()
    SDDP.train(model; iteration_limit, print_level = 0, kwargs...)
    @test SDDP.calculate_bound(model) ≈ exact rtol = 1e-6
    @test _policy_cost(model) ≈ exact rtol = 1e-6
    return exact
end

function test_simple_hydro()
    exact = _validate(
        () -> Hydro.simple_hydro_dhd(),
        () -> Hydro.simple_hydro_dh_aug();
        iteration_limit = 10,
    )
    @test exact ≈ 410 # Documented optimum from the tutorial.
    return
end

function test_seasonal_hydro()
    years = 2
    _validate(
        () -> Hydro.seasonal_hydro_dhd(; years),
        () -> Hydro.seasonal_hydro_dh_aug(; years);
        iteration_limit = 150,
    )
    return
end

function test_batteries()
    T, Ω = 6, [-4.0, 0.0, 4.0]
    _validate(
        () -> Batteries.batteries_dhd(; T, Ω),
        () -> Batteries.batteries_dh_aug(; T, Ω);
        iteration_limit = 300,
    )
    return
end

function test_markovian_hydro()
    # Train with and without sharing cuts between with the same children;
    # by default, cut sharing is on.
    for refine_at_similar_nodes in (true, false)
        exact = _validate(
            () -> Hydro.markovian_hydro_dhd(),
            () -> Hydro.markovian_hydro_split();
            iteration_limit = 30,
            refine_at_similar_nodes,
        )
        @test exact ≈ 285
    end
    return
end

function test_markovian_hydro_cut_sharing()
    # Shared cuts give every node of a stage its exact cut in one iteration.
    model = Hydro.markovian_hydro_dhd()
    SDDP.train(model; iteration_limit = 1, print_level = 0)
    @test SDDP.calculate_bound(model) ≈ 285 rtol = 1e-6
    return
end

function test_portfolio_quadrature()
    # Probabilities sum to one and mean return matches exp(μ Δt) - 1.
    Ω, P = Portfolio.gbm_returns(10)
    μ, Δt = only(Portfolio.MU), Portfolio.DT
    @test sum(P) ≈ 1
    @test sum(P .* first.(Ω)) ≈ exp(μ * Δt) - 1 rtol = 1e-10
    # No arbitrage.
    Ω, _ = Portfolio.gbm_returns(2)
    @test minimum(first.(Ω)) < 0 < maximum(first.(Ω))
    # Log-returns of two correlated assets have the exact mean and covariance.
    μ, σ, C = [0.08, 0.05], [0.2, 0.15], [1 0.5; 0.5 1]
    Ω, P = Portfolio.gbm_returns(3; μ, σ, C)
    Y = [log.(1 .+ R) for R in Ω]
    ȳ = sum(P .* Y)
    cov = sum(P[k] * (Y[k] - ȳ) * (Y[k] - ȳ)' for k in eachindex(P))
    @test length(Ω) == 9
    @test ȳ ≈ (μ .- σ .^2 ./ 2) .* Δt atol = 1e-12
    @test cov ≈ Δt .* σ .* C .* σ' atol = 1e-12
end

function test_portfolio_one_period_references()
    ρ, U_MAX = Portfolio.RHO, Portfolio.U_MAX
    # Newton matches the closed form for two equally likely returns.
    Ω, P = Portfolio.gbm_returns(2)
    R_m, R_p = only.(Ω)
    _, U = Portfolio.one_period_optimum(Ω, P)
    @test only(U) ≈ log(-R_p / R_m) / (ρ * (R_p - R_m)) rtol = 1e-10
    # Newton's minimum matches a fine grid search, and investing beats U = 0.
    Ω, P = Portfolio.gbm_returns(5)
    m, U = Portfolio.one_period_optimum(Ω, P)
    f(U) = sum(p * exp(-ρ * only(R) * U) for (R, p) in zip(Ω, P))
    grid = range(-U_MAX, U_MAX; length = 400_001)
    @test minimum(f, grid) ≈ m rtol = 1e-8
    @test m < 1
    # Two correlated assets: Newton's minimizer is stable.
    Ω2, P2 = Portfolio.gbm_returns(3; μ = [0.08, 0.05], σ = [0.2, 0.15], C = [1 0.5; 0.5 1])
    m2, U2 = Portfolio.one_period_optimum(Ω2, P2)
    f2(U) = sum(p * exp(-ρ * (R' * U)) for (R, p) in zip(Ω2, P2))
    @test all(f2(U2 + d) >= m2 for d in ([1e-3, 0], [-1e-3, 0], [0, 1e-3], [0, -1e-3]))
    # The tangent loss lies below exp(ρ u) / ρ and touches it at each tangent point.
    @test all(Portfolio.tangent_loss(u) <= exp(ρ * u) / ρ + 1e-12 for u in -3:0.01:3)
    @test all(Portfolio.tangent_loss(a) ≈ exp(ρ * a) / ρ for a in Portfolio.TANGENTS)
    for X in -1.0:0.5:1.0
        v, U_lp = Portfolio.one_period_lp_optimum(Ω, P, X)
        f_lp(U) = sum(p * Portfolio.tangent_loss(-X - only(R) * U) for (R, p) in zip(Ω, P))
        # The breakpoint minimum is exact, so a grid can only match it from above.
        @test f_lp(U_lp) ≈ v
        @test minimum(f_lp, grid) >= v - 1e-12
        @test minimum(f_lp, grid) ≈ v atol = 1e-4
        # The tangent loss is below the exponential loss, so its optimum is too.
        @test v <= m * exp(-ρ * X) / ρ
    end
    return
end

function test_portfolio_lp_last_node()
    # With T = 1, the joint subproblem is the whole one-period problem, with no cuts.
    Ω, P = Portfolio.gbm_returns(5)
    model = Portfolio.portfolio_dhd(; T = 1, returns = (Ω, P))
    node = model[1]
    @test length(node.dhd.recourse_copies) == length(Ω)
    v(X) = first(Portfolio.one_period_lp_optimum(Ω, P, X))
    for X in -1.0:0.5:1.0
        joint = SDDP.solve_joint_subproblem(
            model,
            node,
            Dict(:x => X);
            duality_handler = SDDP.ContinuousConicDuality(),
        )
        # The joint value is the exact one-period minimum.
        @test joint.objective ≈ v(X) rtol = 1e-8
        # The commited holding attains that minimum.
        u = only(values(joint.here_and_now)) # Might not be unique here.
        f_lp(U) = sum(p * Portfolio.tangent_loss(-X - only(R) * U) for (R, p) in zip(Ω, P))
        @test f_lp(u) ≈ v(X) rtol = 1e-8
        @test abs(u) < Portfolio.U_MAX
        # The dual of x.in lies between the left and right slopes of v, i.e. subgradient.
        h = 1e-6
        @test (v(X) - v(X - h)) / h - 1e-6 <= joint.duals[:x] <= (v(X + h) - v(X)) / h + 1e-6
    end
    @test SDDP.calculate_bound(model) ≈ v(Portfolio.WEALTH_0) rtol = 1e-8
    return
end

function test_portfolio_lp_dh_aug()
    Ω, P = Portfolio.gbm_returns(5)
    optima = Float64[]
    for T in 1:3
        dhd = _exact(Portfolio.portfolio_dhd(; T, returns = (Ω, P)))
        dh_aug = _exact(Portfolio.portfolio_dh_aug(; T, returns = (Ω, P)))
        @test dhd ≈ dh_aug rtol = 1e-6
        push!(optima, dhd)
    end
    # With one period, both match the breakpoint reference.
    @test optima[1] ≈ first(Portfolio.one_period_lp_optimum(Ω, P, Portfolio.WEALTH_0))
    # More periods can never hurt.
    @test optima[3] <= optima[2] <= optima[1]
    return
end

function test_portfolio_lp()
    T, n = 3, 3
    _validate(
        () -> Portfolio.portfolio_dhd(; T, n),
        () -> Portfolio.portfolio_dh_aug(; T, n);
        iteration_limit = 200,
    )
    return
end

function test_portfolio_lp_below_exp()
    T, n = 3, 5
    Ω, P = Portfolio.gbm_returns(n)
    m, _ = Portfolio.one_period_optimum(Ω, P)
    exp_value = Portfolio.exact_value(T, m)
    # Each tangent set contains the previous one, so the tangent loss can only rise.
    tangent_sets = [Portfolio.TANGENTS, collect(-3:0.25:3), collect(-3:0.05:3)]
    optima = [
        # Deterministic equivalent of DHD model.
        _exact(Portfolio.portfolio_dhd(; T, returns = (Ω, P), tangents))
        for tangents in tangent_sets
    ]
    @test optima[1] <= optima[2] <= optima[3]
    @test optima[3] <= exp_value # Tangents are upper bounded by exact function.
    @test optima[3] ≈ exp_value rtol = 1e-3 # Close enough with fine tangents.
    return
end

end # module

TestDHDValidation.runtests()

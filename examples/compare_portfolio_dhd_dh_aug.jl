# Compare how fast SDDP converges on the DHD and DH-aug formulations of a
# portfolio with three correlated assets and the exponential loss, solved
# with MOSEK. The optimum has a closed form, so for every seed we record
# the iteration and time at which the lower bound first comes within a
# relative gap TOL of it.

import MosekTools
import Printf
import Random
import SDDP
using Statistics: median

module Portfolio
include(joinpath(@__DIR__, "portfolio_models.jl"))
end

# Number of assets, Gauss-Hermite nodes per asset and periods.
const J, N, T = 3, 3, 3
"Relative gap to the optimum at which a run counts as converged."
const TOL = 1e-4
const SEEDS = 1:5
const ITERATION_LIMIT = 2000
"Seconds after which the gaps of both formulations are compared."
const TIME_BUDGETS = [5.0, 100.0]
"Drifts, volatilities and pairwise correlation of the assets, as in compare_dhd_dh_aug.jl."
const ASSET_MU = [0.08, 0.05, 0.03]
const ASSET_SIGMA = [0.2, 0.15, 0.1]
const ASSET_CORR = 0.5

"Train a new model from `build` and return it."
function train_model(build; seed, iteration_limit)
    Random.seed!(seed)
    model = build()
    SDDP.train(
        model;
        iteration_limit,
        print_level = 0,
        log_file = joinpath(@__DIR__, "compare_portfolio_dhd_dh_aug.log"),
    )
    return model
end

"Median of `x`, or NaN if no run reached the gap."
median_or_nan(x) = isempty(x) ? NaN : median(x)

"Relative gap between the bound in `l` and `exact`."
relative_gap(l, exact) = (exact - l.bound) / abs(exact)

"Maximum of `x`, or NaN if no run reached the gap."
maximum_or_nan(x) = isempty(x) ? NaN : maximum(x)

"Committed holding: a here-and-now value in DHD, an outgoing state in DH-aug."
holding(u::Real) = u
holding(u::SDDP.State) = u.out

"Holdings that `model` commits at its first node, before any return is seen."
first_holdings(model) = holding.(only(SDDP.simulate(model, 1, [:u]))[1][:u])

"Portfolio with J assets and the exponential loss, whose optimum has a closed form."
C = [i == j ? 1.0 : ASSET_CORR for i in 1:J, j in 1:J]
returns = Portfolio.gbm_returns(N; μ = ASSET_MU, σ = ASSET_SIGMA, C)
m, U = Portfolio.one_period_optimum(returns...)
exact = Portfolio.exact_value(T, m)
optimizer = SDDP.JuMP.optimizer_with_attributes(
    MosekTools.Optimizer,
    "MSK_IPAR_NUM_THREADS" => 1, # Single thread.
)
kwargs = (; T, returns, tangents = nothing, optimizer)

println(
    "J = $J, n = $N, T = $T, exact optimum = $exact, tol = $TOL, ",
    "$(length(SEEDS)) seeds of $ITERATION_LIMIT iterations",
)

for (formulation, build) in [
    ("DHD", () -> Portfolio.portfolio_dhd(; kwargs...)),
    ("DH-aug", () -> Portfolio.portfolio_dh_aug(; kwargs...)),
]
    model = build()
    states, stages = length(model[1].states), length(model.nodes)
    # Warm up so that compilation does not count towards the timings.
    println("Warming up...")
    train_model(build; seed = 0, iteration_limit = 2)
    iterations, seconds, ms_to_tol = Int[], Float64[], Float64[]
    solves_per_iteration, ms_per_iteration = Float64[], Float64[]
    final_gaps, holding_errors = Float64[], Float64[]
    budget_gaps = [Float64[] for _ in TIME_BUDGETS]
    for seed in SEEDS
        println("Training with seed $seed...")
        trained = train_model(build; seed, iteration_limit = ITERATION_LIMIT)
        log = trained.most_recent_training_results.log
        push!(solves_per_iteration, log[end].total_solves / length(log))
        push!(ms_per_iteration, 1000 * log[end].time / length(log))
        i = findfirst(l -> relative_gap(l, exact) <= TOL, log)
        if i !== nothing
            push!(iterations, i)
            push!(seconds, log[i].time)
            push!(ms_to_tol, 1000 * log[i].time / i)
        end
        push!(final_gaps, relative_gap(log[end], exact))
        for (k, budget) in enumerate(TIME_BUDGETS)
            # The last bound available when the budget runs out.
            k_last = findlast(l -> l.time <= budget, log)
            push!(budget_gaps[k], k_last === nothing ? NaN : relative_gap(log[k_last], exact))
        end
        push!(holding_errors, maximum(abs.(first_holdings(trained) .- U)))
        println("... Done!")

    end
    println("$formulation: $states states, $stages stages")
    Printf.@printf(
        "  per iteration: %.0f solves, %.1f ms over the run, %.1f ms until tol\n",
        median(solves_per_iteration),
        median(ms_per_iteration),
        median_or_nan(ms_to_tol),
    )
    Printf.@printf(
        "  tol reached by %d/%d seeds: median %g it and %.1f s, worst %g it and %.1f s\n",
        length(iterations),
        length(SEEDS),
        median_or_nan(iterations),
        median_or_nan(seconds),
        maximum_or_nan(iterations),
        maximum_or_nan(seconds),
    )
    Printf.@printf(
        "  gap after %d iterations: median %.1e, worst %.1e\n",
        ITERATION_LIMIT,
        median(final_gaps),
        maximum(final_gaps),
    )
    for (k, budget) in enumerate(TIME_BUDGETS)
        Printf.@printf("  gap after %g s: median %.1e\n", budget, median(budget_gaps[k]))
    end
    Printf.@printf(
        "  first-stage holding error: median %.1e, worst %.1e\n",
        median(holding_errors),
        maximum(holding_errors),
    )
end

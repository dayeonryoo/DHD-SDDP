# Compare how fast SDDP converges on the DHD and DH-aug formulations of the
# same problems. Both formulations have the same exact optimum, so here we
# record, for each instance and seed, the iteration and time at which the
# lower bound first comes within a relative gap of that optimum.

# Outputs are:
#   compare_dhd_dh_aug.csv  the bound after every iteration of every run
#   compare_dhd_dh_aug.png  the relative gap to the optimum for every run

import Clarabel
import HiGHS
import Plots
import Printf
import Random
import SDDP
using Statistics: median

module Hydro
include(joinpath(@__DIR__, "hydro_models.jl"))
end

module Batteries
include(joinpath(@__DIR__, "batteries_models.jl"))
end

module Portfolio
include(joinpath(@__DIR__, "portfolio_models.jl"))
end

const SEEDS = 1:5
"Relative gaps at which the iterations and times are reported."
const GAPS = [1e-2, 1e-4, 1e-6]

"Drift, volatilities and pairwise correlation of the portfolio's assets."
const ASSET_MU = [0.08, 0.05, 0.03]
const ASSET_SIGMA = [0.2, 0.15, 0.1]
const ASSET_CORR = 0.5

"Portfolio with J assets and the exponential loss, whose optimum has a closed form."
function portfolio_instance(J; T = 3, n = 2, iteration_limit)
    C = [i == j ? 1.0 : ASSET_CORR for i in 1:J, j in 1:J]
    returns = Portfolio.gbm_returns(n; μ = ASSET_MU[1:J], σ = ASSET_SIGMA[1:J], C)
    m, _ = Portfolio.one_period_optimum(returns...)
    kwargs = (; T, returns, tangents = nothing, optimizer = Clarabel.Optimizer)
    return (
        "portfolio, J = $J, T = $T, n = $n",
        () -> Portfolio.portfolio_dhd(; kwargs...),
        () -> Portfolio.portfolio_dh_aug(; kwargs...),
        iteration_limit,
        () -> Portfolio.exact_value(T, m),
    )
end

"Instances: name, DHD builder, DH-aug builder, iteration limit and exact optimum."
const INSTANCES = [
    (
        "simple hydro, T = 4",
        () -> Hydro.simple_hydro_dhd(),
        () -> Hydro.simple_hydro_dh_aug(),
        50,
        () -> exact_optimum(Hydro.simple_hydro_dh_aug()),
    ),
    (
        "seasonal hydro, 2 years",
        () -> Hydro.seasonal_hydro_dhd(; years = 2),
        () -> Hydro.seasonal_hydro_dh_aug(; years = 2),
        200,
        () -> exact_optimum(Hydro.seasonal_hydro_dh_aug(; years = 2)),
    ),
    (
        "batteries, T = 6, 3 deviations",
        () -> Batteries.batteries_dhd(; T = 6, Ω = [-4.0, 0.0, 4.0]),
        () -> Batteries.batteries_dh_aug(; T = 6, Ω = [-4.0, 0.0, 4.0]),
        400,
        () -> exact_optimum(Batteries.batteries_dh_aug(; T = 6, Ω = [-4.0, 0.0, 4.0])),
    ),
    portfolio_instance(1; iteration_limit = 200),
    portfolio_instance(2; iteration_limit = 400),
    portfolio_instance(3; iteration_limit = 600),
]

"Exact optimum of `model` from its deterministic equivalent."
function exact_optimum(model)
    det = SDDP.deterministic_equivalent(model, HiGHS.Optimizer)
    SDDP.JuMP.set_silent(det)
    SDDP.JuMP.optimize!(det)
    return SDDP.JuMP.objective_value(det)
end

"Train a new model from `build` and return its log."
function train_log(build; seed, iteration_limit)
    Random.seed!(seed)
    model = build()
    SDDP.train(
        model;
        iteration_limit,
        print_level = 0,
        log_file = joinpath(@__DIR__, "compare_dhd_dh_aug.log"),
    )
    return model.most_recent_training_results.log
end

"Relative gap between the bound in `l` and `exact`."
relative_gap(l, exact) = (exact - l.bound) / abs(exact)

"Median of `x`, or NaN if no run reached the gap."
median_or_nan(x) = isempty(x) ? NaN : median(x)

rows = String["instance,formulation,seed,iteration,bound,time,total_solves"]
plots = Any[]
println(
    "instance | formulation | states | stages | solves per iteration | " *
    "ms per iteration | median iterations and seconds to a gap of ",
    join(GAPS, ", "),
)

for (name, build_dhd, build_dh_aug, iteration_limit, optimum) in INSTANCES
    exact = optimum()
    plt, plt_time = [
        Plots.plot(;
            title = name,
            xlabel,
            ylabel = (xlabel == "iteration") ? "relative gap" : "",
            yaxis = :log10,
            ylims = (1e-9, 2),
            legend = :topright,
        ) for xlabel in ("iteration", "seconds")
    ]
    Plots.hline!(plt, GAPS; color = :gray, linestyle = :dash, label = "")
    for (formulation, build, color) in
        [("DHD", build_dhd, :blue), ("DH-aug", build_dh_aug, :red)]
        model = build()
        states, stages = length(model[1].states), length(model.nodes)
        # Warm up so that compilation does not count towards the timings.
        train_log(build; seed = 0, iteration_limit = 2)
        iterations = [Int[] for _ in GAPS]
        seconds = [Float64[] for _ in GAPS]
        solves_per_iteration, ms_per_iteration = Float64[], Float64[]
        for seed in SEEDS
            log = train_log(build; seed, iteration_limit)
            for l in log
                push!(
                    rows,
                    "\"$name\",$formulation,$seed,$(l.iteration),$(l.bound)," *
                    "$(l.time),$(l.total_solves)",
                )
            end
            push!(solves_per_iteration, log[end].total_solves / length(log))
            push!(ms_per_iteration, 1000 * log[end].time / length(log))
            for (k, gap) in enumerate(GAPS)
                i = findfirst(l -> relative_gap(l, exact) <= gap, log)
                if i !== nothing
                    push!(iterations[k], i)
                    push!(seconds[k], log[i].time)
                end
            end
            # A gap of zero cannot be drawn on a log scale, so floor it.
            gap = [max(relative_gap(l, exact), 1e-9) for l in log]
            label = (seed == first(SEEDS)) ? formulation : ""
            Plots.plot!(plt, gap; color, alpha = 0.4, label)
            Plots.plot!(plt_time, [l.time for l in log], gap; color, alpha = 0.4, label)
        end
        reached = [
            Printf.@sprintf(
                "%g it, %.3f s",
                median_or_nan(iterations[k]),
                median_or_nan(seconds[k]),
            ) for k in eachindex(GAPS)
        ]
        println(
            "$name | $formulation | $states | $stages | ",
            Printf.@sprintf("%.0f", median(solves_per_iteration)), " | ",
            Printf.@sprintf("%.2f", median(ms_per_iteration)), " | ",
            join(reached, " | "),
        )
    end
    for p in (plt, plt_time)
        Plots.hline!(p, GAPS; color = :gray, linestyle = :dash, label = "")
    end
    push!(plots, plt, plt_time)
end

write(joinpath(@__DIR__, "compare_dhd_dh_aug.csv"), join(rows, "\n") * "\n")
Plots.savefig(
    Plots.plot(
        plots...;
        layout = (length(INSTANCES), 2),
        size = (1200, 300 * length(INSTANCES)),
        left_margin = 10Plots.mm,
    ),
    joinpath(@__DIR__, "compare_dhd_dh_aug.png"),
)

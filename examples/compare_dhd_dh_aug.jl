# Compare how fast SDDP converges on the DHD and DH-aug formulations of the
# same problems. Both formulations have the same exact optimum, so here we
# record, for each instance and seed, the iteration and time at which the
# lower bound first comes within a relative gap of that optimum.

# Outputs are:
#   compare_dhd_dh_aug.csv  the bound after every iteration of every run
#   compare_dhd_dh_aug.png  the relative gap to the optimum for every run

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

const SEEDS = 1:5
"Relative gaps at which the iterations and times are reported."
const GAPS = [1e-2, 1e-4, 1e-6]

"Instances: name, DHD builder, DH-aug builder and iteration limit."
const INSTANCES = [
    (
        "simple hydro, T = 4",
        () -> Hydro.simple_hydro_dhd(),
        () -> Hydro.simple_hydro_dh_aug(),
        50,
    ),
    (
        "seasonal hydro, 2 years",
        () -> Hydro.seasonal_hydro_dhd(; years = 2),
        () -> Hydro.seasonal_hydro_dh_aug(; years = 2),
        200,
    ),
    (
        "batteries, T = 6, 3 deviations",
        () -> Batteries.batteries_dhd(; T = 6, Ω = [-4.0, 0.0, 4.0]),
        () -> Batteries.batteries_dh_aug(; T = 6, Ω = [-4.0, 0.0, 4.0]),
        400,
    ),
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

for (name, build_dhd, build_dh_aug, iteration_limit) in INSTANCES
    exact = exact_optimum(build_dh_aug())
    plt = Plots.plot(;
        title = name,
        xlabel = "iteration",
        ylabel = "relative gap",
        yaxis = :log10,
        ylims = (1e-9, 2),
        legend = :topright,
    )
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
            Plots.plot!(
                plt,
                gap;
                color,
                alpha = 0.4,
                label = seed == first(SEEDS) ? formulation : "",
            )
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
    push!(plots, plt)
end

write(joinpath(@__DIR__, "compare_dhd_dh_aug.csv"), join(rows, "\n") * "\n")
Plots.savefig(
    Plots.plot(plots...; layout = (length(plots), 1), size = (800, 900)),
    joinpath(@__DIR__, "compare_dhd_dh_aug.png"),
)

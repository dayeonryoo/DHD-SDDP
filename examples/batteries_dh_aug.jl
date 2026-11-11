#  Derived from the SDDP.jl tutorials "Example: batteries" and "Here-and-now
#  and hazard-decision". Source: https://github.com/odow/SDDP.jl
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Compare the hazard-decision (HD) battery model with the augmented-state
# decision-hazard (DH) version, in which thermal output is committed one hour
# before the load is observed.
#
# Part 1 solves a small instance with SDDP and checks both formulations against
#        the exact optimum of the deterministic equivalent.
# Part 2 trains both formulations on the full 24-hour instance and evaluates
#        the two policies on the same sampled load paths.
#
# Run from the repository root:
#     julia --project=examples examples/batteries_dh_aug.jl
# or, in the REPL started with --project=examples:
#     include("examples/batteries_dh_aug.jl")

include(joinpath(@__DIR__, "batteries_models.jl"))

import Printf
import Random

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

const T_SMALL = 6                   # hours in the small instance
const Ω_SMALL = [-4.0, 0.0, 4.0]    # 3^6 = 729 scenarios
const ITERATIONS_SMALL = 200
const ITERATIONS_FULL = 500         # same as the SDDP.jl tutorial
const N_SIMULATIONS = 1_000
const TRAIN_SEED = 1234
const SIMULATION_SEED = 2026

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"Exact optimal value from the deterministic equivalent (small instances only)."
function exact_value(model)
    det = SDDP.deterministic_equivalent(model, HiGHS.Optimizer)
    set_silent(det)
    optimize!(det)
    if !is_solved_and_feasible(det)
        error("Deterministic equivalent not solved: ", termination_status(det))
    end
    return objective_value(det)
end

"Train in serial with a fixed seed, so runs are reproducible. Nothing is printed or logged."
function train_and_bound!(model; iteration_limit::Int)
    Random.seed!(TRAIN_SEED)
    SDDP.train(
        model;
        iteration_limit = iteration_limit,
        print_level = 0,
        log_file = joinpath(@__DIR__, "SDDP.log"),
    )
    return SDDP.calculate_bound(model)
end

"Sample `N` load paths, each a vector of `T` deviations drawn uniformly from `Ω`."
function sample_load_paths(T::Int, Ω::Vector{Float64}, N::Int; seed::Int)
    rng = Random.Xoshiro(seed)
    return [rand(rng, Ω, T) for _ in 1:N]
end

# The same load path written as a scenario for each formulation.
# HD: node t is hour t.
hd_scenario(path) = [(t, path[t]) for t in eachindex(path)]
# DH: node 1 is the deterministic hour 0 (no noise); node t + 1 is hour t.
function dh_scenario(path)
    return Tuple{Int,Any}[(1, nothing); [(t + 1, path[t]) for t in eachindex(path)]]
end

"Total cost of the trained policy on each of the given scenarios, in order."
function evaluate_policy(model, scenarios)
    simulations = SDDP.simulate(
        model,
        length(scenarios);
        sampling_scheme = SDDP.Historical(scenarios),
    )
    return [sum(stage[:stage_objective] for stage in sim) for sim in simulations]
end

"Sample mean and the half-width of a 95% confidence interval."
function mean_ci(x::Vector{Float64})
    n = length(x)
    m = sum(x) / n
    s = sqrt(sum(abs2, x .- m) / (n - 1))
    return m, 1.96 * s / sqrt(n)
end

const CHECKS = Pair{String,Bool}[]

function check(label::String, ok::Bool)
    println(ok ? "  [pass] " : "  [FAIL] ", label)
    push!(CHECKS, label => ok)
    return ok
end

fmt(x) = Printf.@sprintf("%.6e", x)

# ---------------------------------------------------------------------------
# Part 1: small instance, checked against the exact optimum
# ---------------------------------------------------------------------------

n_scenarios = length(Ω_SMALL)^T_SMALL
println("\nPart 1: T = $T_SMALL hours, |Ω| = $(length(Ω_SMALL)), ",
        "$n_scenarios scenarios")

hd_exact = exact_value(batteries_hd(; T = T_SMALL, Ω = Ω_SMALL))
dh_exact = exact_value(batteries_dh_aug(; T = T_SMALL, Ω = Ω_SMALL))
hd_small = train_and_bound!(
    batteries_hd(; T = T_SMALL, Ω = Ω_SMALL);
    iteration_limit = ITERATIONS_SMALL,
)
dh_small = train_and_bound!(
    batteries_dh_aug(; T = T_SMALL, Ω = Ω_SMALL);
    iteration_limit = ITERATIONS_SMALL,
)

println("                  exact optimum    SDDP bound")
println("  HD              ", fmt(hd_exact), "    ", fmt(hd_small))
println("  DH (augmented)  ", fmt(dh_exact), "    ", fmt(dh_small))
check("HD: SDDP bound equals the exact optimum",
      isapprox(hd_small, hd_exact; rtol = 1e-6))
check("DH: SDDP bound equals the exact optimum",
      isapprox(dh_small, dh_exact; rtol = 1e-6))
check("DH optimum >= HD optimum (less information cannot cost less)",
      dh_exact >= hd_exact - 1e-6)

# With no uncertainty, committing one hour ahead loses nothing, so the two
# formulations must have the same optimum. This tests the reformulation itself
# (timing of the ramp, first stage, initial values), not the SDDP solver.
hd_no_noise = exact_value(batteries_hd(; Ω = [0.0]))
dh_no_noise = exact_value(batteries_dh_aug(; Ω = [0.0]))
println("  No uncertainty, T = 24: HD ", fmt(hd_no_noise), ", DH ", fmt(dh_no_noise))
check("Without uncertainty, HD and DH have the same optimum",
      isapprox(hd_no_noise, dh_no_noise; rtol = 1e-8))

# ---------------------------------------------------------------------------
# Part 2: full 24-hour instance
# ---------------------------------------------------------------------------

println("\nPart 2: T = 24 hours, |Ω| = $(length(LOAD_NOISE)), ",
        "$ITERATIONS_FULL iterations each")

hd = batteries_hd()
dh = batteries_dh_aug()
t_hd = @elapsed hd_bound = train_and_bound!(hd; iteration_limit = ITERATIONS_FULL)
t_dh = @elapsed dh_bound = train_and_bound!(dh; iteration_limit = ITERATIONS_FULL)

# Evaluate both policies on the same load paths, so the difference between
# them is not blurred by sampling noise.
paths = sample_load_paths(24, LOAD_NOISE, N_SIMULATIONS; seed = SIMULATION_SEED)
hd_costs = evaluate_policy(hd, [hd_scenario(p) for p in paths])
dh_costs = evaluate_policy(dh, [dh_scenario(p) for p in paths])
hd_mean, hd_ci = mean_ci(hd_costs)
dh_mean, dh_ci = mean_ci(dh_costs)
diff_mean, diff_ci = mean_ci(dh_costs .- hd_costs)

println("                  lower bound      simulated cost (95% CI)        train time")
println("  HD              ", fmt(hd_bound), "     ", fmt(hd_mean), " ± ",
        Printf.@sprintf("%.1f", hd_ci), "      ", Printf.@sprintf("%.1f s", t_hd))
println("  DH (augmented)  ", fmt(dh_bound), "     ", fmt(dh_mean), " ± ",
        Printf.@sprintf("%.1f", dh_ci), "      ", Printf.@sprintf("%.1f s", t_dh))
println("  Cost of committing thermal output one hour ahead, same load paths: ",
        Printf.@sprintf("%.1f ± %.1f", diff_mean, diff_ci))
println("  (The SDDP.jl documentation reports an HD bound of 5.712852e+04.)")

check("HD bound agrees with the SDDP.jl documentation (5.712852e+04)",
      isapprox(hd_bound, 5.712852e4; rtol = 1e-4))
check("DH bound >= HD bound", dh_bound >= hd_bound)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

n_failed = count(!last, CHECKS)
println("\n", n_failed == 0 ? "All $(length(CHECKS)) checks passed." :
        "$n_failed of $(length(CHECKS)) checks FAILED.")

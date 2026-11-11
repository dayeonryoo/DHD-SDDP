#  Derived from the SDDP.jl tutorial "Here-and-now and hazard-decision" and
#  the SDDP.jl example "Hydro-thermal scheduling".
#  Source: https://github.com/odow/SDDP.jl
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Compare hazard-decision (HD) hydro-thermal models with their decision-hazard
# (DH) versions, in which u_thermal is added as a state variable so thermal
# output is committed one stage ahead: the HD state is x_storage alone
# and the augmented DH state is (x_storage, u_thermal).
#
# Part 1 reproduces the documented costs of the "Here-and-now and
#        hazard-decision" tutorial ($400 HD, $410 DH) and checks SDDP against
#        the exact optimum.
# Part 2 checks the seasonal model over 2 years against the exact optimum of
#        the deterministic equivalent.
# Part 3 trains the seasonal model over 10 years and looks at the DH
#        cost-to-go as a function of the augmented state.
#
# Run from the repository root:
#     julia --project=examples examples/hydro_dh_aug.jl
# or, in the REPL started with --project=examples:
#     include("examples/hydro_dh_aug.jl")

include(joinpath(@__DIR__, "hydro_models.jl"))

import Printf
import Random

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

const ITERATIONS = 200
const TRAIN_SEED = 1234
const YEARS_SMALL = 2                  # 6 seasons, 3^6 = 729 scenarios
const YEARS_LARGE = SEASONAL_YEARS     # 30 seasons, too many to enumerate
const SIMPLE_DOCUMENTED_HD = 400.0     # tutorial, hazard_decision
const SIMPLE_DOCUMENTED_DH = 410.0     # tutorial, decision_hazard_2

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
function train_and_bound!(model; iteration_limit::Int = ITERATIONS)
    Random.seed!(TRAIN_SEED)
    SDDP.train(
        model;
        iteration_limit = iteration_limit,
        print_level = 0,
        log_file = joinpath(@__DIR__, "SDDP.log"),
    )
    return SDDP.calculate_bound(model)
end

const CHECKS = Pair{String,Bool}[]

function check(label::String, ok::Bool)
    println(ok ? "  [pass] " : "  [FAIL] ", label)
    push!(CHECKS, label => ok)
    return ok
end

fmt(x) = Printf.@sprintf("%.6e", x)

function print_table(hd_exact, hd_bound, dh_exact, dh_bound)
    println("                  exact optimum    SDDP bound")
    println("  HD              ", fmt(hd_exact), "    ", fmt(hd_bound))
    println("  DH (augmented)  ", fmt(dh_exact), "    ", fmt(dh_bound))
    return
end

# ---------------------------------------------------------------------------
# Part 1: simple hydro, from the "Here-and-now and hazard-decision" tutorial
# ---------------------------------------------------------------------------

println("\nPart 1: simple hydro (tutorial), 4 stages, ",
        "$(length(SIMPLE_INFLOW)^4) scenarios")

hd_exact = exact_value(simple_hydro_hd())
dh_exact = exact_value(simple_hydro_dh_aug())
hd_bound = train_and_bound!(simple_hydro_hd())
dh_bound = train_and_bound!(simple_hydro_dh_aug())
print_table(hd_exact, hd_bound, dh_exact, dh_bound)
check("HD exact optimum equals the documented \$400",
      isapprox(hd_exact, SIMPLE_DOCUMENTED_HD; rtol = 1e-8))
check("DH exact optimum equals the documented \$410",
      isapprox(dh_exact, SIMPLE_DOCUMENTED_DH; rtol = 1e-8))
check("HD: SDDP bound equals the exact optimum",
      isapprox(hd_bound, hd_exact; rtol = 1e-6))
check("DH: SDDP bound equals the exact optimum",
      isapprox(dh_bound, dh_exact; rtol = 1e-6))

# Without uncertainty, committing one stage ahead loses nothing.
hd_no_noise = exact_value(simple_hydro_hd(; Ω = [2.5]))
dh_no_noise = exact_value(simple_hydro_dh_aug(; Ω = [2.5]))
check("Without uncertainty, HD and DH have the same optimum",
      isapprox(hd_no_noise, dh_no_noise; rtol = 1e-8))

# ---------------------------------------------------------------------------
# Part 2: seasonal hydro over 2 years, checked against the exact optimum
# ---------------------------------------------------------------------------

n_scenarios = length(SEASONAL_NOISE)^(N_SEASONS * YEARS_SMALL)
println("\nPart 2: seasonal hydro, $YEARS_SMALL years ",
        "($(N_SEASONS * YEARS_SMALL) seasons), $n_scenarios scenarios")

hd_exact = exact_value(seasonal_hydro_hd(; years = YEARS_SMALL))
dh_exact = exact_value(seasonal_hydro_dh_aug(; years = YEARS_SMALL))
hd_bound = train_and_bound!(seasonal_hydro_hd(; years = YEARS_SMALL))
dh_bound = train_and_bound!(seasonal_hydro_dh_aug(; years = YEARS_SMALL))
print_table(hd_exact, hd_bound, dh_exact, dh_bound)
check("HD: SDDP bound equals the exact optimum",
      isapprox(hd_bound, hd_exact; rtol = 1e-6))
check("DH: SDDP bound equals the exact optimum",
      isapprox(dh_bound, dh_exact; rtol = 1e-6))
check("DH optimum >= HD optimum (less information cannot cost less)",
      dh_exact >= hd_exact - 1e-6)

hd_no_noise = exact_value(seasonal_hydro_hd(; years = YEARS_SMALL, Ω = [(3.0, 5.0)]))
dh_no_noise = exact_value(seasonal_hydro_dh_aug(; years = YEARS_SMALL, Ω = [(3.0, 5.0)]))
check("Without uncertainty, HD and DH have the same optimum",
      isapprox(hd_no_noise, dh_no_noise; rtol = 1e-8))

# ---------------------------------------------------------------------------
# Part 3: seasonal hydro over 10 years
# ---------------------------------------------------------------------------

println("\nPart 3: seasonal hydro, $YEARS_LARGE years ",
        "($(N_SEASONS * YEARS_LARGE) seasons), $ITERATIONS iterations each")

hd = seasonal_hydro_hd(; years = YEARS_LARGE)
dh = seasonal_hydro_dh_aug(; years = YEARS_LARGE)
t_hd = @elapsed hd_bound = train_and_bound!(hd)
t_dh = @elapsed dh_bound = train_and_bound!(dh)
println("                  lower bound      state variables        train time")
println("  HD              ", fmt(hd_bound), "     x_storage              ",
        Printf.@sprintf("%.1f s", t_hd))
println("  DH (augmented)  ", fmt(dh_bound), "     x_storage, u_thermal   ",
        Printf.@sprintf("%.1f s", t_dh))
check("DH bound >= HD bound", dh_bound >= hd_bound)

# The cuts stored at node 1 (the commit stage) approximate the cost from the
# first season onward as a function of the outgoing state: V(x_storage,
# u_thermal). Node 1 costs nothing and keeps x_storage at its initial value,
# so the DH bound is min over g of V(STORAGE_0, g). That minimisation over the
# commitment is exactly what a DHD stage would do internally, giving a
# cost-to-go in x_storage alone.
V = SDDP.ValueFunction(dh[1])
cost_at(g) = SDDP.evaluate(V; x_storage = STORAGE_0, u_thermal = g)[1]
grid = collect(0.0:1.5:THERMAL_MAX)
println("\n  DH cost-to-go after the commit stage, V(x_storage = $STORAGE_0, u_thermal = g):")
for g in grid
    println("    g = ", Printf.@sprintf("%4.1f", g), "   V = ",
            Printf.@sprintf("%.2f", cost_at(g)))
end
g_star = SDDP.simulate(dh, 1, [:u_thermal])[1][1][:u_thermal].out
println("  Commitment chosen by the policy: g* = ", Printf.@sprintf("%.3f", g_star),
        ",  V = ", Printf.@sprintf("%.2f", cost_at(g_star)),
        ",  DH bound = ", Printf.@sprintf("%.2f", dh_bound))
check("DH bound = V(x_storage_0, g*): the bound is the minimum over commitments",
      isapprox(cost_at(g_star), dh_bound; rtol = 1e-6))
check("No commitment on the grid beats g*",
      all(cost_at(g) >= dh_bound - 1e-6 for g in grid))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

n_failed = count(!last, CHECKS)
println("\n", n_failed == 0 ? "All $(length(CHECKS)) checks passed." :
        "$n_failed of $(length(CHECKS)) checks FAILED.")

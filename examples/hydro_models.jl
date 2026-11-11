#  Derived from the SDDP.jl tutorial "Here-and-now and hazard-decision" and
#  the SDDP.jl example "Hydro-thermal scheduling".
#  Source: https://github.com/odow/SDDP.jl
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Shared data and model builders for two hydro-thermal examples. In both, the
# decision-hazard (DH) version commits thermal output one stage ahead by adding
# u_thermal as a state variable. Unlike the battery example, this genuinely
# augments the state:
#
#   hazard-decision (HD) state:  x_storage              (1 dimension)
#   augmented DH state:          x_storage, u_thermal   (2 dimensions)
#   DHD state (for comparison):  x_storage              (1 dimension)
#
# 1. simple_hydro_*   : the 4-stage model from the "Here-and-now and
#                       hazard-decision" tutorial (documented costs: HD $400,
#                       DH $410).
# 2. seasonal_hydro_* : the 3-season model from the "Hydro-thermal scheduling"
#                       example, on a finite horizon of `years` years. The
#                       example is infinite-horizon with a discount factor of
#                       0.95 per year; here the same discount multiplies each
#                       year's costs instead, so every model is a linear
#                       policy graph.
#
# Changes from the documented models, needed so that every subproblem is feasible
# for every committed thermal output, and chosen so that the HD optimum is unchanged:
#   - simple hydro: add a free spill variable u_spill, and bound u_thermal by
#     the demand (both are redundant in the HD model).
#   - seasonal hydro: add unmet demand (cost VOLL) and surplus generation
#     (cost equal to the spill cost), and bound u_thermal by the largest
#     demand (all redundant in the HD model).

using SDDP
import HiGHS

# ===========================================================================
# 1. Simple hydro (tutorial "Here-and-now and hazard-decision")
# ===========================================================================

const SIMPLE_DEMAND = 9.0
const SIMPLE_STORAGE_MAX = 8.0
const SIMPLE_STORAGE_0 = 6.0
const SIMPLE_THERMAL_COST = 20.0
const SIMPLE_VOLL = 500.0
"Equally likely inflows, independent across stages."
const SIMPLE_INFLOW = [2.0, 3.0]

"""
    simple_hydro_hd(; T = 4, Ω = SIMPLE_INFLOW)

Hazard-decision model: all generation is decided after the inflow is seen.
Node `t` is stage `t`.
"""
function simple_hydro_hd(; T::Int = 4, Ω::Vector{Float64} = SIMPLE_INFLOW)
    return SDDP.LinearPolicyGraph(;
        stages = T,
        sense = :Min,
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer,
    ) do sp, t
        @variable(
            sp,
            0 <= x_storage <= SIMPLE_STORAGE_MAX,
            SDDP.State,
            initial_value = SIMPLE_STORAGE_0,
        )
        @variable(sp, 0 <= u_thermal <= SIMPLE_DEMAND)
        @variable(sp, u_hydro >= 0)
        @variable(sp, u_spill >= 0)
        @variable(sp, u_unmet_demand >= 0)
        @variable(sp, w_inflow)
        SDDP.parameterize(ω -> fix(w_inflow, ω), sp, Ω)
        @constraint(sp, u_thermal + u_hydro == SIMPLE_DEMAND - u_unmet_demand)
        @constraint(
            sp,
            x_storage.out == x_storage.in - u_hydro - u_spill + w_inflow,
        )
        @stageobjective(
            sp,
            SIMPLE_VOLL * u_unmet_demand + SIMPLE_THERMAL_COST * u_thermal,
        )
        return
    end
end

"""
    simple_hydro_dh_aug(; T = 4, Ω = SIMPLE_INFLOW)

Decision-hazard model for thermal output, using the tutorial's trick:
u_thermal becomes a state variable. Has `T + 1` stages: node 1 is a
deterministic stage that commits the stage-1 thermal output, and node `t + 1`
is stage `t`, in which
    u_thermal.in  = thermal output in stage t (committed one node earlier)
    u_thermal.out = thermal output committed for stage t + 1
"""
function simple_hydro_dh_aug(; T::Int = 4, Ω::Vector{Float64} = SIMPLE_INFLOW)
    return SDDP.LinearPolicyGraph(;
        stages = T + 1,
        sense = :Min,
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer,
    ) do sp, node
        @variable(
            sp,
            0 <= x_storage <= SIMPLE_STORAGE_MAX,
            SDDP.State,
            initial_value = SIMPLE_STORAGE_0,
        )
        # The augmented state. Its initial value is never used: node 1 has no
        # demand constraint.
        @variable(
            sp,
            0 <= u_thermal <= SIMPLE_DEMAND,
            SDDP.State,
            initial_value = 0.0,
        )
        if node == 1
            @constraint(sp, x_storage.out == x_storage.in)
            @stageobjective(sp, 0.0)
        else
            @variable(sp, u_hydro >= 0)
            @variable(sp, u_spill >= 0)
            @variable(sp, u_unmet_demand >= 0)
            @variable(sp, w_inflow)
            SDDP.parameterize(ω -> fix(w_inflow, ω), sp, Ω)
            @constraint(
                sp,
                u_thermal.in + u_hydro == SIMPLE_DEMAND - u_unmet_demand,
            )
            @constraint(
                sp,
                x_storage.out == x_storage.in - u_hydro - u_spill + w_inflow,
            )
            @stageobjective(
                sp,
                SIMPLE_VOLL * u_unmet_demand +
                SIMPLE_THERMAL_COST * u_thermal.in,
            )
        end
        return
    end
end

# ===========================================================================
# 2. Seasonal hydro (example "Hydro-thermal scheduling"), finite horizon
# ===========================================================================

const N_SEASONS = 3            # summer-fall, winter, spring
const SEASONAL_YEARS = 10      # default horizon
const DISCOUNT = 0.95          # per year
const STORAGE_MIN = 5.0
const STORAGE_MAX = 15.0
const STORAGE_0 = 10.0
const SPILL_COST = 1.0
const SURPLUS_COST = SPILL_COST  # dumping hydro must not be cheaper than spill
const VOLL = 500.0
"Equally likely (inflow, demand) pairs, independent across seasons."
const SEASONAL_NOISE = [(0.0, 7.5), (3.0, 5.0), (10.0, 2.5)]
"Thermal output never needs to exceed the largest demand."
const THERMAL_MAX = maximum(last, SEASONAL_NOISE)

thermal_cost(season::Int) = Float64(season)  # $1, $2, $3 per unit

"Season (1, 2, 3) and year (1, 2, ...) of the `t`-th season of the horizon."
season_and_year(t::Int) = (mod1(t, N_SEASONS), cld(t, N_SEASONS))

"Discount applied to all costs incurred in `year`."
discount_factor(year::Int) = DISCOUNT^(year - 1)

"""
    seasonal_hydro_hd(; years = SEASONAL_YEARS, Ω = SEASONAL_NOISE)

Hazard-decision seasonal model with `3 * years` stages. Node `t` is the `t`-th
season of the horizon.
"""
function seasonal_hydro_hd(;
    years::Int = SEASONAL_YEARS,
    Ω::Vector{Tuple{Float64,Float64}} = SEASONAL_NOISE,
)
    return SDDP.LinearPolicyGraph(;
        stages = N_SEASONS * years,
        sense = :Min,
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer,
    ) do sp, t
        season, year = season_and_year(t)
        @variable(
            sp,
            STORAGE_MIN <= x_storage <= STORAGE_MAX,
            SDDP.State,
            initial_value = STORAGE_0,
        )
        @variable(sp, 0 <= u_thermal <= THERMAL_MAX)
        @variable(sp, u_hydro >= 0)
        @variable(sp, u_spill >= 0)
        @variable(sp, u_unmet_demand >= 0)
        @variable(sp, u_surplus >= 0)
        @variable(sp, w_inflow)
        @variable(sp, w_demand)
        SDDP.parameterize(sp, Ω) do ω
            fix(w_inflow, ω[1])
            fix(w_demand, ω[2])
            return
        end
        @constraint(
            sp,
            x_storage.out == x_storage.in - u_hydro - u_spill + w_inflow,
        )
        @constraint(
            sp,
            u_hydro + u_thermal + u_unmet_demand == w_demand + u_surplus,
        )
        @stageobjective(
            sp,
            discount_factor(year) * (
                SPILL_COST * u_spill +
                thermal_cost(season) * u_thermal +
                VOLL * u_unmet_demand +
                SURPLUS_COST * u_surplus
            ),
        )
        return
    end
end

"""
    seasonal_hydro_dh_aug(; years = SEASONAL_YEARS, Ω = SEASONAL_NOISE)

Seasonal model in which thermal output is committed one season ahead, with
u_thermal as a state variable. Has `3 * years + 1` stages: node 1 is a
deterministic stage that commits the first season's output, and node `t + 1`
is the `t`-th season of the horizon, in which
    u_thermal.in  = thermal output this season (committed one node earlier)
    u_thermal.out = thermal output committed for the next season
"""
function seasonal_hydro_dh_aug(;
    years::Int = SEASONAL_YEARS,
    Ω::Vector{Tuple{Float64,Float64}} = SEASONAL_NOISE,
)
    return SDDP.LinearPolicyGraph(;
        stages = N_SEASONS * years + 1,
        sense = :Min,
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer,
    ) do sp, node
        @variable(
            sp,
            STORAGE_MIN <= x_storage <= STORAGE_MAX,
            SDDP.State,
            initial_value = STORAGE_0,
        )
        @variable(
            sp,
            0 <= u_thermal <= THERMAL_MAX,
            SDDP.State,
            initial_value = 0.0,
        )
        if node == 1
            @constraint(sp, x_storage.out == x_storage.in)
            @stageobjective(sp, 0.0)
        else
            season, year = season_and_year(node - 1)
            @variable(sp, u_hydro >= 0)
            @variable(sp, u_spill >= 0)
            @variable(sp, u_unmet_demand >= 0)
            @variable(sp, u_surplus >= 0)
            @variable(sp, w_inflow)
            @variable(sp, w_demand)
            SDDP.parameterize(sp, Ω) do ω
                fix(w_inflow, ω[1])
                fix(w_demand, ω[2])
                return
            end
            @constraint(
                sp,
                x_storage.out == x_storage.in - u_hydro - u_spill + w_inflow,
            )
            @constraint(
                sp,
                u_hydro + u_thermal.in + u_unmet_demand == w_demand + u_surplus,
            )
            @stageobjective(
                sp,
                discount_factor(year) * (
                    SPILL_COST * u_spill +
                    thermal_cost(season) * u_thermal.in +
                    VOLL * u_unmet_demand +
                    SURPLUS_COST * u_surplus
                ),
            )
        end
        return
    end
end

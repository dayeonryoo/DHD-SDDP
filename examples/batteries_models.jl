#  Derived from the SDDP.jl tutorials "Example: batteries" (model originally by
#  Andy Philpott, Winter School Kvitfjell 2025) and "Here-and-now and
#  hazard-decision". Source: https://github.com/odow/SDDP.jl
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Shared data and model builders for the battery example.
#
# Every formulation of the problem is built from the constants below, so the
# hazard-decision (HD) model, the augmented-state decision-hazard (DH) model
# and, later, the DHD-SDDP model are guaranteed to describe the same problem.

using SDDP
import HiGHS

# ---------------------------------------------------------------------------
# Problem data
# ---------------------------------------------------------------------------

"Expected net load in hours 1-24 (the 'duck curve')."
const NET_LOAD = vcat(
    [40, 41, 42, 43, 35, 40, 40, 25, 10, 8, 6, 5],  # hours 01-12
    [5, 6, 8, 10, 20, 30, 55, 72, 75, 70, 64, 60],  # hours 13-24
)

"Equally likely deviations of the load, independent across hours."
const LOAD_NOISE = [-4.0, -2.0, 0.0, 2.0, 4.0]

const SOC_0 = 4.0         # initial state of charge
const SOC_MAX = 30.0      # battery capacity
const CHARGE_MAX = 15.0   # max charge and discharge per hour
const EFFICIENCY = 0.8    # charging efficiency
const THERMAL_0 = 35.0    # thermal output in hour 0
const THERMAL_MAX = 70.0  # thermal capacity
const RAMP_UP = 10.0      # max increase in thermal output between hours
const FUEL_COST = 70.0    # cost per unit of thermal output
const VOLL = 500.0        # value of lost load

function _check_inputs(T, Ω)
    if !(1 <= T <= length(NET_LOAD))
        error("T must be between 1 and $(length(NET_LOAD)); got $T")
    end
    if isempty(Ω)
        error("Ω must contain at least one load deviation")
    end
    return
end

# ---------------------------------------------------------------------------
# Hazard-decision (HD): all decisions in hour t are made after observing the
# load in hour t. This is the model from the SDDP.jl tutorial.
# ---------------------------------------------------------------------------

"""
    batteries_hd(; T = 24, Ω = LOAD_NOISE)

Hazard-decision battery model with `T` hourly stages. Node `t` is hour `t`.
"""
function batteries_hd(; T::Int = 24, Ω::Vector{Float64} = LOAD_NOISE)
    _check_inputs(T, Ω)
    return SDDP.LinearPolicyGraph(;
        stages = T,
        sense = :Min,
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer,
    ) do sp, t
        # State variables
        @variable(sp, 0 <= x_soc <= SOC_MAX, SDDP.State, initial_value = SOC_0)
        # x_thermal.out is the thermal output in hour t, chosen after the load
        # in hour t is observed.
        @variable(
            sp,
            0 <= x_thermal <= THERMAL_MAX,
            SDDP.State,
            initial_value = THERMAL_0,
        )
        # Control variables
        @variable(sp, 0 <= u_charge <= CHARGE_MAX)
        @variable(sp, 0 <= u_discharge <= CHARGE_MAX)
        @variable(sp, u_slack >= 0)
        @variable(sp, u_surplus >= 0)
        # Random variable
        @variable(sp, w_load)
        SDDP.parameterize(ω -> fix(w_load, NET_LOAD[t] + ω), sp, Ω)
        # Objective and constraints
        @stageobjective(sp, FUEL_COST * x_thermal.out + VOLL * u_slack)
        @constraint(
            sp,
            x_soc.out == x_soc.in + EFFICIENCY * u_charge - u_discharge,
        )
        @constraint(sp, x_thermal.out - x_thermal.in <= RAMP_UP)
        @constraint(
            sp,
            λ,
            x_thermal.out + u_discharge - u_charge + u_slack ==
            w_load + u_surplus,
        )
        return
    end
end

# ---------------------------------------------------------------------------
# Decision-hazard for thermal output, written with the state-augmentation trick
# from the SDDP.jl "Here-and-now and hazard-decision" tutorial.
#
# Thermal output for hour t is committed BEFORE the load in hour t is
# observed; the battery and the slack/surplus variables remain recourse
# decisions made AFTER it. This is a DHD problem in augmented-state form.
#
# Node 1 is a deterministic "hour 0" stage that commits the hour-1 output.
# Node t + 1 is hour t:
#   x_thermal.in  = output in hour t (committed at the previous node)
#   x_thermal.out = output committed for hour t + 1
# ---------------------------------------------------------------------------

"""
    batteries_dh_aug(; T = 24, Ω = LOAD_NOISE)

Battery model in which thermal output is a here-and-now decision, written with
an augmented state. Has `T + 1` stages: node 1 is a deterministic hour 0, and
node `t + 1` is hour `t`.
"""
function batteries_dh_aug(; T::Int = 24, Ω::Vector{Float64} = LOAD_NOISE)
    _check_inputs(T, Ω)
    return SDDP.LinearPolicyGraph(;
        stages = T + 1,
        sense = :Min,
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer,
    ) do sp, node
        @variable(sp, 0 <= x_soc <= SOC_MAX, SDDP.State, initial_value = SOC_0)
        @variable(
            sp,
            0 <= x_thermal <= THERMAL_MAX,
            SDDP.State,
            initial_value = THERMAL_0,
        )
        # Ramp limit between the current hour and the hour being committed.
        # In node 1 this is hour 0 -> hour 1, matching the HD model.
        @constraint(sp, x_thermal.out - x_thermal.in <= RAMP_UP)
        if node == 1
            # Hour 0: only commit the hour-1 output. Nothing else happens.
            @constraint(sp, x_soc.out == x_soc.in)
            @stageobjective(sp, 0.0)
        else
            t = node - 1
            @variable(sp, 0 <= u_charge <= CHARGE_MAX)
            @variable(sp, 0 <= u_discharge <= CHARGE_MAX)
            @variable(sp, u_slack >= 0)
            @variable(sp, u_surplus >= 0)
            @variable(sp, w_load)
            SDDP.parameterize(ω -> fix(w_load, NET_LOAD[t] + ω), sp, Ω)
            # Cost and energy balance use the output committed one stage
            # earlier, x_thermal.in.
            @stageobjective(sp, FUEL_COST * x_thermal.in + VOLL * u_slack)
            @constraint(
                sp,
                x_soc.out == x_soc.in + EFFICIENCY * u_charge - u_discharge,
            )
            @constraint(
                sp,
                λ,
                x_thermal.in + u_discharge - u_charge + u_slack ==
                w_load + u_surplus,
            )
        end
        return
    end
end

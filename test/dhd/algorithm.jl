module TestDHDAlgorithm

using SDDP
using Test
import HiGHS

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

"Simple hydro with two DHD stages; `sense = :Max` negates the costs."
function _two_stage_hydro(; sense = :Min)
    s = sense == :Min ? 1 : -1
    return SDDP.LinearPolicyGraph(;
        stages = 2,
        sense = sense,
        lower_bound = sense == :Min ? 0.0 : -Inf,
        upper_bound = sense == :Min ? Inf : 0.0,
        optimizer = HiGHS.Optimizer,
    ) do sp, t
        @variable(sp, 0 <= x_storage <= 8, SDDP.State, initial_value = 6)
        @variable(sp, 0 <= u_thermal <= 9, SDDP.HereAndNow)
        @variable(sp, u_hydro >= 0)
        @variable(sp, u_spill >= 0)
        @variable(sp, u_unmet_demand >= 0)
        @variable(sp, w_inflow)
        SDDP.parameterize(ω -> fix(w_inflow, ω), sp, [2.0, 3.0])
        @constraint(sp, u_thermal + u_hydro == 9 - u_unmet_demand)
        @constraint(
            sp,
            x_storage.out == x_storage.in - u_hydro - u_spill + w_inflow
        )
        @stageobjective(sp, s * (500 * u_unmet_demand + 20 * u_thermal))
    end
end

"Add the exact stage 2 cost-to-go as a cut at node 1."
function _add_exact_cut(model; sense = :Min)
    s = sense == :Min ? 1 : -1
    SDDP._add_cut(
        model[1].bellman_function.global_theta,
        s * 140.0,
        Dict(:x_storage => -s * 20.0),
        Dict(:x_storage => 0.0),
        nothing,
        nothing,
    )
    return
end

function test_joint_optimizer_attached()
    model = _two_stage_hydro()
    node = model[1]
    joint = node.dhd.joint
    @test MOI.Utilities.state(backend(joint)) == MOI.Utilities.NO_OPTIMIZER
    SDDP._initialize_solver(node; throw_error = false)
    @test solver_name(joint) == "HiGHS"
    SDDP._uninitialize_solver(model; throw_error = false)
    @test MOI.Utilities.state(backend(joint)) == MOI.Utilities.NO_OPTIMIZER
    return
end

function test_solve_joint_with_exact_cut()
    model = _two_stage_hydro()
    _add_exact_cut(model)
    node = model[1]
    result = SDDP.solve_joint_subproblem(
        model,
        node,
        Dict(:x_storage => 6.0);
        duality_handler = SDDP.ContinuousConicDuality(),
    )
    @test result.objective ≈ 150
    u = result.here_and_now[node.subproblem[:u_thermal]]
    @test 1 - 1e-6 <= u <= 7 + 1e-6
    @test result.duals[:x_storage] ≈ -20
    return
end

function test_solve_joint_leaf()
    model = _two_stage_hydro()
    node = model[2]
    handler = SDDP.ContinuousConicDuality()
    # Below 7, one more unit of storage saves 20 of thermal.
    result = SDDP.solve_joint_subproblem(
        model,
        node,
        Dict(:x_storage => 6.0);
        duality_handler = handler,
    )
    @test result.objective ≈ 20
    @test result.duals[:x_storage] ≈ -20
    # Above 7, no thermal is needed.
    result = SDDP.solve_joint_subproblem(
        model,
        node,
        Dict(:x_storage => 8.0);
        duality_handler = handler,
    )
    @test result.objective ≈ 0 atol = 1e-6
    @test result.duals[:x_storage] ≈ 0 atol = 1e-6
    return
end

function test_solve_joint_without_duals()
    model = _two_stage_hydro()
    _add_exact_cut(model)
    result = SDDP.solve_joint_subproblem(
        model,
        model[1],
        Dict(:x_storage => 6.0);
        duality_handler = nothing,
    )
    @test result.objective ≈ 150
    @test isempty(result.duals)
    return
end

function test_solve_joint_max_dual_sign()
    model = _two_stage_hydro(; sense = :Max)
    _add_exact_cut(model; sense = :Max)
    result = SDDP.solve_joint_subproblem(
        model,
        model[1],
        Dict(:x_storage => 6.0);
        duality_handler = SDDP.ContinuousConicDuality(),
    )
    @test result.objective ≈ -150
    @test result.duals[:x_storage] ≈ 20
    return
end

function test_solve_joint_counts_solves()
    model = _two_stage_hydro()
    before = get(model.ext, :total_solves, 0)
    SDDP.solve_joint_subproblem(
        model,
        model[2],
        Dict(:x_storage => 6.0);
        duality_handler = nothing,
    )
    @test model.ext[:total_solves] == before + 1
    return
end

function test_solve_joint_infeasible_error()
    model = _two_stage_hydro()
    # Storage cannot stay non-negative when it starts at -10.
    @test_throws(
        ErrorException,
        SDDP.solve_joint_subproblem(
            model,
            model[2],
            Dict(:x_storage => -10.0);
            duality_handler = nothing,
        ),
    )
    @test model.ext[:numerical_issue]
    return
end

function test_solve_joint_unsupported_duality_handler()
    model = _two_stage_hydro()
    @test_throws(
        ErrorException,
        SDDP.solve_joint_subproblem(
            model,
            model[2],
            Dict(:x_storage => 6.0);
            duality_handler = SDDP.LagrangianDuality(),
        ),
    )
    return
end

function test_recourse_fixes_here_and_now()
    model = _two_stage_hydro()
    _add_exact_cut(model)
    node = model[1]
    # x = 6, h = 9 - u, x' = x - h + w, θ = 140 - 20x'
    u = node.subproblem[:u_thermal]
    path = Tuple{Int,Any}[(1, 2.0)]
    # Dry (w = 2):
    #   1 ≤ u ≤ 8, x' = u - 1, stage cost = 20u, θ = 160 - 20u, objective = 160
    result = SDDP.solve_recourse_subproblem(
        model,
        node,
        Dict(:x_storage => 6.0),
        Dict(u => 4.0),
        2.0,
        path,
    )
    @test is_fixed(u)
    @test !has_lower_bound(u) && !has_upper_bound(u)
    @test value(u) ≈ 4
    @test result.state[:x_storage] ≈ 3
    @test result.stage_objective ≈ 80
    @test result.objective ≈ 160
    # Wet (w = 3):
    #   0 ≤ u ≤ 7, x' = u, stage cost = 20u, θ = 140 - 20u, objective = 140
    result = SDDP.solve_recourse_subproblem(
        model,
        node,
        Dict(:x_storage => 6.0),
        Dict(u => 6.0),
        3.0,
        path,
    )
    @test value(u) ≈ 6
    @test result.state[:x_storage] ≈ 6
    @test result.stage_objective ≈ 120
    @test result.objective ≈ 140
    return
end

function test_solve_dhd_node()
    model = _two_stage_hydro()
    _add_exact_cut(model)
    node = model[1]
    for (w, objective) in [(2.0, 160.0), (3.0, 140.0)]
        result = SDDP.solve_dhd_node(
            model,
            node,
            Dict(:x_storage => 6.0),
            w,
            Tuple{Int,Any}[(1, w)],
        )
        # The recourse uses the commitment from the joint solve.
        u = value(node.subproblem[:u_thermal])
        @test u ≈ value(node.dhd.first_stage[node.subproblem[:u_thermal]])
        @test 1 - 1e-6 <= u <= 7 + 1e-6
        @test result.state[:x_storage] ≈ u + w - 3
        @test result.stage_objective ≈ 20 * u
        @test result.objective ≈ objective
    end
    return
end

function test_solve_dhd_node_counts_solves()
    model = _two_stage_hydro()
    before = get(model.ext, :total_solves, 0)
    SDDP.solve_dhd_node(
        model,
        model[2],
        Dict(:x_storage => 6.0),
        2.0,
        Tuple{Int,Any}[(2, 2.0)],
    )
    # One joint solve and one recourse solve.
    @test model.ext[:total_solves] == before + 2
    return
end

function test_calculate_bound_without_cuts()
    model = _two_stage_hydro()
    # With only θ ≥ 0, node 1 commits u = 1 to cover the dry case.
    @test SDDP.calculate_bound(model) ≈ 20
    return
end

function test_calculate_bound_with_exact_cuts()
    model = _two_stage_hydro()
    _add_exact_cut(model)
    @test SDDP.calculate_bound(model) ≈ 150
    # V_1(x) = 270 - 20x near x = 6 (initial storage)
    @test SDDP.calculate_bound(model, Dict(:x_storage => 4.0)) ≈ 190
    return
end

function test_calculate_bound_max()
    model = _two_stage_hydro(; sense = :Max)
    _add_exact_cut(model; sense = :Max)
    @test SDDP.calculate_bound(model) ≈ -150
    return
end

function test_calculate_bound_one_solve_per_root_child()
    model = _two_stage_hydro()
    before = get(model.ext, :total_solves, 0)
    SDDP.calculate_bound(model)
    # One joint solve; an HD node would be solved once per noise.
    @test model.ext[:total_solves] == before + 1
    return
end

function test_calculate_bound_weights_root_children()
    graph = SDDP.Graph(0, [1, 2], [(0 => 1, 0.3), (0 => 2, 0.7)])
    model = SDDP.PolicyGraph(
        graph;
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer,
    ) do sp, node
        @variable(sp, 0 <= x_storage <= 8, SDDP.State, initial_value = 6.0)
        @variable(sp, 0 <= u_thermal <= 9, SDDP.HereAndNow)
        @variable(sp, u_hydro >= 0)
        @variable(sp, u_spill >= 0)
        @variable(sp, u_unmet_demand >= 0)
        @variable(sp, w_inflow)
        SDDP.parameterize(ω -> fix(w_inflow, ω), sp, [2.0, 3.0])
        @constraint(sp, u_thermal + u_hydro == 9 - u_unmet_demand)
        @constraint(
            sp,
            x_storage.out == x_storage.in - u_hydro - u_spill + w_inflow
        )
        # Thermal costs 20 at node 1 and 40 at node 2.
        @stageobjective(sp, 500 * u_unmet_demand + 20 * node * u_thermal)
    end
    # Both nodes are leaved (θ = 0) and commit u = 1 at x = 6.
    @test SDDP.calculate_bound(model) ≈ 34 # = 0.3 * 20 + 0.7 * 40.
    return
end

function test_forward_pass_path_cost()
    model = _two_stage_hydro()
    _add_exact_cut(model)
    options = SDDP.Options(model, model.initial_root_state)
    for _ in 1:10
        result = SDDP.forward_pass(model, options, SDDP.DefaultForwardPass())
        @test length(result.sampled_states) == 2
        w = result.scenario_path[1][2]
        u = value(model[1].subproblem[:u_thermal])
        @test 1 - 1e-6 <= u <= 7 + 1e-6
        @test result.sampled_states[1][:x_storage] ≈ u + w - 3
        # Thermal covers 18 - (6 + w + 2) over both stages at 20 per unit.
        @test result.cumulative_value ≈ 20 * (10 - w)
    end
    return
end

"Run one forward pass in which both stages see the inflow `w`."
function _forward_pass(model, w)
    options = SDDP.Options(
        model,
        model.initial_root_state;
        sampling_scheme = SDDP.Historical([(1, w), (2, w)]),
    )
    return SDDP.forward_pass(model, options, SDDP.DefaultForwardPass())
end

function test_forward_pass_commits_before_noise()
    model = _two_stage_hydro()
    _add_exact_cut(model)
    _forward_pass(model, 2.0)
    u_dry = value(model[1].subproblem[:u_thermal])
    _forward_pass(model, 3.0)
    u_wet = value(model[1].subproblem[:u_thermal])
    # The commitment is made before the inflow is known.
    @test u_dry ≈ u_wet
    return
end

function test_forward_pass_without_cuts()
    model = _two_stage_hydro()
    result = _forward_pass(model, 2.0)
    # With only θ ≥ 0, stage 1 commits u1 = 1 and uses all 8 units of water.
    @test value(model[1].subproblem[:u_thermal]) ≈ 1
    @test result.sampled_states[1][:x_storage] ≈ 0 atol = 1e-6
    # Stage 2 then commits u2 = 7 at 20 per unit.
    @test value(model[2].subproblem[:u_thermal]) ≈ 7
    @test result.cumulative_value ≈ 160 # = 20u1 + 20u2 = path cost.
    return
end

function test_forward_pass_counts_solve()
    model = _two_stage_hydro()
    options = SDDP.Options(model, model.initial_root_state)
    before = get(model.ext, :total_solves, 0)
    SDDP.forward_pass(model, options, SDDP.DefaultForwardPass())
    # Two nodes, each with a joint solve and a recourse solve.
    @test model.ext[:total_solves] == before + 4
    return
end

"Run the backward pass at node 1 with the trial point `x`."
function _backward_pass(model, x; kwargs...)
    # `train` sets single cuts; the default of a BellmanFunction is multi-cut.
    for node in values(model.nodes)
        node.bellman_function.cut_type = SDDP.SINGLE_CUT
    end
    options = SDDP.Options(model, model.initial_root_state; kwargs...)
    return SDDP.backward_pass(
        model,
        options,
        Tuple{Int,Float64}[(1, 2.0)],
        [Dict(:x_storage => x)],
        NTuple{0,Float64}[],
        Tuple{Int,Dict{Int,Float64}}[]
    )
end

function test_backward_pass_builds_exact_cut()
    model = _two_stage_hydro()
    _backward_pass(model, 6.0)
    # θ ≥ 140 - 20x'.
    cut = only(model[1].bellman_function.global_theta.cuts)
    @test cut.intercept ≈ 140
    @test cut.coefficients[:x_storage] ≈ -20
    @test length(model[1].dhd.cut_constraints[cut]) == 2
    @test SDDP.calculate_bound(model) ≈ 150
    return
end

function test_backward_pass_flat_cut()
    model = _two_stage_hydro()
    _backward_pass(model, 8.0)
    # With storage > 7, V is flat at 0, so the cut is θ ≥ 0.
    cut = only(model[1].bellman_function.global_theta.cuts)
    @test cut.intercept ≈ 0 atol = 1e-6
    @test cut.coefficients[:x_storage] ≈ 0 atol = 1e-6
    @test SDDP.calculate_bound(model) ≈ 20
    return
end

function test_backward_pass_max()
    model = _two_stage_hydro(; sense = :Max)
    _backward_pass(model, 6.0)
    # θ ≤ -140 + 20x'.
    cut = only(model[1].bellman_function.global_theta.cuts)
    @test cut.intercept ≈ -140
    @test cut.coefficients[:x_storage] ≈ 20
    @test length(model[1].dhd.cut_constraints[cut]) == 2
    @test SDDP.calculate_bound(model) ≈ -150
    return
end

function test_backward_pass_one_solve_per_child()
    model = _two_stage_hydro()
    before = get(model.ext, :total_solves, 0)
    _backward_pass(model, 6.0)
    # One joint solve; an HD child would be solved once per noise.
    @test model.ext[:total_solves] == before + 1
    return
end

function test_backward_pass_weights_children()
    graph = SDDP.Graph(
        0,
        [1, 2, 3],
        [(0 => 1, 1.0), (1 => 2, 0.3), (1 => 3, 0.7)],
    )
    model = SDDP.PolicyGraph(
        graph;
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer,
    ) do sp, node
        @variable(sp, 0 <= x_storage <= 8, SDDP.State, initial_value = 6)
        @variable(sp, 0 <= u_thermal <= 9, SDDP.HereAndNow)
        @variable(sp, u_hydro >= 0)
        @variable(sp, u_spill >= 0)
        @variable(sp, u_unmet_demand >= 0)
        @variable(sp, w_inflow)
        SDDP.parameterize(ω -> fix(w_inflow, ω), sp, [2.0, 3.0])
        @constraint(sp, u_thermal + u_hydro == 9 - u_unmet_demand)
        @constraint(
            sp,
            x_storage.out == x_storage.in - u_hydro - u_spill + w_inflow
        )
        # Thermal costs 40 at node 3 and 20 elsewhere.
        cost = node === 3 ? 40 : 20
        @stageobjective(sp, 500 * u_unmet_demand + cost * u_thermal)
    end
    _backward_pass(model, 6.0)
    # Children: V = 20, λ = -20 (p = 0.3) and V = 40, λ = -40 (p = 0.7).
    # Expectation: θ = 34 and β = -34 at x = 6, so α = 34 + 34 * 6 = 238.
    cut = only(model[1].bellman_function.global_theta.cuts)
    @test cut.intercept ≈ 238
    @test cut.coefficients[:x_storage] ≈ -34
    return
end

function test_backward_pass_monte_carlo_error()
    model = _two_stage_hydro()
    @test_throws(
        ErrorException,
        _backward_pass(
            model,
            6.0;
            backward_sampling_scheme = SDDP.MonteCarloSampler(1),
        ),
    )
    return
end

function test_simulate_dhd()
    model = _two_stage_hydro()
    _add_exact_cut(model)
    sims = SDDP.simulate(model, 10, [:u_thermal, :x_storage])
    for sim in sims
        @test length(sim) == 2
        w = sim[1][:noise_term]
        # The here-and-now values are recorded from the fixed template.
        @test 1 - 1e-6 <= sim[1][:u_thermal] <= 7 + 1e-6
        x = sim[1][:x_storage].out
        @test sim[2][:u_thermal] ≈ 7 - x
        # θ = 140 - 20x' at stage 1, and 0 at stage 2 (leaf).
        @test sim[1][:bellman_term] ≈ 140 - 20 * x
        @test sim[2][:bellman_term] ≈ 0 atol = 1e-6
        @test sum(s[:stage_objective] for s in sim) ≈ 20 * (10 - w)
    end
    return
end

function test_simulate_dhd_duality_handler_error()
    model= _two_stage_hydro()
    @test_throws(
        ErrorException(
            "`SDDP.simulate` does not support a `duality_handler` for DHD nodes yet."
        ),
        SDDP.simulate(
            model,
            1;
            duality_handler = SDDP.ContinuousConicDuality(),
        )
    )
    return
end

function test_decision_rule_dhd()
    model = _two_stage_hydro()
    rule = SDDP.DecisionRule(model; node = 2)
    for w in (2.0, 3.0)
        result = SDDP.evaluate(
            rule;
            incoming_state = Dict(:x_storage => 3.0),
            noise = w,
            controls_to_record = [:u_thermal, :u_hydro],
        )
        # The commitment is u_thermal = max(0, 7 - x) = 4 at x = 3 for both noises.
        # u_hydro = 9 - u_thermal = 5, and the stage cost is 20 * u_thermal.
        @test result.controls[:u_thermal] ≈ 4
        @test result.controls[:u_hydro] ≈ 5
        @test result.stage_objective ≈ 80
    end
    return
end

end # module

TestDHDAlgorithm.runtests()

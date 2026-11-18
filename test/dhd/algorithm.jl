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

end # module

TestDHDAlgorithm.runtests()

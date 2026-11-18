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

end # module

TestDHDAlgorithm.runtests()

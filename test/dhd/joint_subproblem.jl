module TestJointSubproblem

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

"Small DHD model with one state, one here-and-now variable, and three noise terms."
function _small_model(; Ω = [1.0, 2.0, 3.0])
    return SDDP.LinearPolicyGraph(; stages = 1, lower_bound = 0.0) do sp, t
        @variable(sp, 0 <= x <= 10, SDDP.State, initial_value = 5.0)
        @variable(sp, 0 <= u_b <= 4, SDDP.HereAndNow)
        @variable(sp, u_a >= 0)
        @variable(sp, w)
        @constraint(sp, u_b <= x.in)
        @constraint(sp, x.out == x.in + u_a + u_b - w)
        SDDP.parameterize(ω -> fix(w, ω), sp, Ω)
        @stageobjective(sp, 2 * u_b + 3 * u_a)
    end
end

function test_HD_node_not_expanded()
    model = SDDP.LinearPolicyGraph(; stages = 1, lower_bound = 0.0) do sp, t
        @variable(sp, x >= 0, SDDP.State, initial_value = 0.0)
    end
    @test model[1].dhd === nothing
    @test !SDDP.is_dhd(model[1])
    return
end

function test_expansion_sizes()
    model = _small_model()
    node = model[1]
    @test SDDP.is_dhd(node)
    dhd = node.dhd
    @test length(dhd.first_stage) == 2 # x.in and u_b
    @test length(dhd.recourse_copies) == 3 # number of noises
    @test num_variables(dhd.joint) == 2 + 3 * 3 # first stage + (x.out, u_a, noise) x #noise
    @test [c.noise.term for c in dhd.recourse_copies] == [1.0, 2.0, 3.0]
    @test all(c.noise.probability == 1/3 for c in dhd.recourse_copies)
    return
end

function test_first_stage_constraints_added_once()
    node = _small_model()[1]
    joint = node.dhd.joint
    u_b = node.dhd.first_stage[node.subproblem[:u_b]]
    # `u <= x.in` once; `x.out == ...` once per copy.
    @test num_constraints(joint, AffExpr, MOI.LessThan{Float64}) == 1
    @test num_constraints(joint, AffExpr, MOI.EqualTo{Float64}) == 3
    @test lower_bound(u_b) == 0.0
    @test upper_bound(u_b) == 4.0
    return
end

function test_copies_use_shared_first_stage()
    node = _small_model()[1]
    x_in = node.dhd.first_stage[node.subproblem[:x].in]
    u_b = node.dhd.first_stage[node.subproblem[:u_b]]
    for c in node.dhd.recourse_copies
        @test c.variables[node.subproblem[:x].in] === x_in
        @test c.variables[node.subproblem[:u_b]] === u_b
        @test fix_value(c.variables[node.subproblem[:w]]) == c.noise.term
    end
    u_a = [c.variables[node.subproblem[:u_a]] for c in node.dhd.recourse_copies]
    @test length(unique(u_a)) == 3 # u_a is copied for each noise
    return
end

function test_stage_objective_per_copy()
    model = SDDP.LinearPolicyGraph(; stages = 1, lower_bound = 0.0) do sp, t
        @variable(sp, x >= 0, SDDP.State, initial_value = 0.0)
        @variable(sp, u_b >= 0, SDDP.HereAndNow)
        @variable(sp, u_a >= 0)
        SDDP.parameterize(sp, [10.0, 20.0]) do ω
            @stageobjective(sp, u_b + ω * u_a)
            return
        end
    end
    node = model[1]
    u_b = node.dhd.first_stage[node.subproblem[:u_b]]
    # Expectation: 0.5 * (u_b + 10 u_a1) + 0.5 * (u_b + 20 u_a2).
    obj = objective_function(node.dhd.joint)
    @test coefficient(obj, u_b) ≈ 1.0
    for c in node.dhd.recourse_copies
        u_a = c.variables[node.subproblem[:u_a]]
        @test coefficient(c.stage_objective, u_b) == 1.0
        @test coefficient(c.stage_objective, u_a) == c.noise.term
        @test coefficient(obj, u_a) ≈ c.noise.probability * c.noise.term
    end
    return
end

function test_single_noise()
    model = SDDP.LinearPolicyGraph(; stages = 1, lower_bound = 0.0) do sp, t
        @variable(sp, x >= 0, SDDP.State, initial_value = 0.0)
        @variable(sp, u_b >= 0, SDDP.HereAndNow)
        @stageobjective(sp, u_b)
    end
    copies = model[1].dhd.recourse_copies
    @test length(copies) == 1
    @test copies[1].noise.probability == 1.0
    return
end

function test_parameterize_modifies_first_stage_error()
    builder = function (sp, t)
        @variable(sp, x >= 0, SDDP.State, initial_value = 0.0)
        @variable(sp, u >= 0, SDDP.HereAndNow)
        SDDP.parameterize(ω -> set_upper_bound(u, ω), sp, [1.0, 2.0])
        @stageobjective(sp, u)
    end
    @test_throws ErrorException SDDP.LinearPolicyGraph(builder; stages = 1, lower_bound = 0.0)
    return
end

function test_objective_state_error()
    builder = function (sp, t)
        SDDP.add_objective_state(
            (u_a, ω) -> u_a + ω,
            sp;
            initial_value = 0.0,
            lower_bound = 0.0,
            upper_bound = 1.0,
            lipschitz = 1.0,
        )
        @variable(sp, x >= 0, SDDP.State, initial_value = 0.0)
        @variable(sp, u >= 0, SDDP.HereAndNow)
        @stageobjective(sp, u)
    end
    @test_throws ErrorException SDDP.LinearPolicyGraph(builder; stages = 1, lower_bound = 0.0)
    return
end

function test_belief_state_error()
    graph = SDDP.LinearGraph(2)
    SDDP.add_ambiguity_set(graph, [1, 2])
    builder = function (sp, t)
        @variable(sp, x >= 0, SDDP.State, initial_value = 0.0)
        @variable(sp, u >= 0, SDDP.HereAndNow)
        @stageobjective(sp, u)
    end
    @test_throws ErrorException SDDP.PolicyGraph(builder, graph; lower_bound = 0.0)
    return
end

function test_set_optimizer()
    model = _small_model()
    optimizer = HiGHS.Optimizer
    set_optimizer(model, optimizer)
    @test solver_name(model[1].dhd.joint) == "HiGHS"
    return
end

"Simple hydro from examples/hydro_models.jl, one stage."
function _simple_hydro(; dhd::Bool)
    return SDDP.LinearPolicyGraph(;
        stages = dhd ? 1 : 2,
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer
    ) do sp, node
        @variable(sp, 0 <= x_storage <= 8, SDDP.State, initial_value = 6)
        if dhd
            @variable(sp, 0 <= u_thermal <= 9, SDDP.HereAndNow)
            thermal = u_thermal
        else
            @variable(sp, 0 <= u_thermal <= 9, SDDP.State, initial_value = 0.0)
            thermal = u_thermal.in
            if node == 1
                @constraint(sp, x_storage.out == x_storage.in)
                @stageobjective(sp, 0.0)
                return
            end
        end
        @variable(sp, u_hydro >= 0)
        @variable(sp, u_spill >= 0)
        @variable(sp, u_unmet_demand >= 0)
        @variable(sp, w_inflow)
        SDDP.parameterize(ω -> fix(w_inflow, ω), sp, [2.0, 3.0])
        @constraint(sp, thermal + u_hydro == 9 - u_unmet_demand)
        @constraint(sp, x_storage.out == x_storage.in - u_hydro - u_spill + w_inflow)
        @stageobjective(sp, 500 * u_unmet_demand + 20 * thermal)
    end
end

function test_simple_hydro_one_stage_value()
    node = _simple_hydro(dhd = true)[1]
    joint = node.dhd.joint
    fix(node.dhd.first_stage[node.subproblem[:x_storage].in], 6.0)
    set_optimizer(joint, HiGHS.Optimizer)
    set_silent(joint)
    optimize!(joint)
    @test objective_value(joint) ≈ 20
    @test value(node.dhd.first_stage[node.subproblem[:u_thermal]]) ≈ 1
    det = SDDP.deterministic_equivalent(_simple_hydro(dhd = false), HiGHS.Optimizer)
    set_silent(det)
    optimize!(det)
    @test objective_value(det) ≈ objective_value(joint)
    return
end

end # module

TestJointSubproblem.runtests()
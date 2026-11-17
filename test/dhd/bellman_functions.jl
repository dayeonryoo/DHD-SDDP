module TestDHDBellmanFunctions

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

"Two-stage DHD model with one state, one here-and-now variable, one recourse variable, and two noise terms."
function _two_stage_model(;
    sense = :Min,
    lower_bound = 0.0,
    upper_bound = Inf,
    P = [0.2, 0.8]
)
    return SDDP.LinearPolicyGraph(;
        stages = 2,
        sense = sense,
        lower_bound = lower_bound,
        upper_bound = upper_bound,
    ) do sp, t
        @variable(sp, 0 <= x <= 10, SDDP.State, initial_value = 5.0)
        @variable(sp, 0 <= ub <= 4, SDDP.HereAndNow)
        @variable(sp, ua >= 0)
        @variable(sp, w)
        @constraint(sp, x.out == x.in + ua + ub - w)
        SDDP.parameterize(ω -> fix(w, ω), sp, [1.0, 2.0], P)
        @stageobjective(sp, 2 * ub + 3 * ua)
    end
end

function test_one_theta_per_copy()
    node = _two_stage_model()[1]
    joint = node.dhd.joint
    thetas = [c.theta for c in node.dhd.recourse_copies]
    @test length(unique(thetas)) == 2
    @test all(owner_model(theta) === joint for theta in thetas)
    @test owner_model(SDDP.bellman_term(node.bellman_function)) === node.subproblem
    # First stage: x.in and ub. Each copy: x.out, ua, w and theta.
    @test num_variables(joint) == 2 + 2 * 4
    return
end

function test_theta_bounds_min()
    node = _two_stage_model()[1]
    for c in node.dhd.recourse_copies
        @test lower_bound(c.theta) == 0.0
        @test !has_upper_bound(c.theta)
    end
    return
end

function test_theta_bounds_max()
    node = _two_stage_model(;
        sense = :Max,
        lower_bound = -Inf,
        upper_bound = 10.0
    )[1]
    for c in node.dhd.recourse_copies
        @test !has_lower_bound(c.theta)
        @test upper_bound(c.theta) == 10.0
    end
    return
end

function test_leaf_theta_zero()
    node = _two_stage_model()[2]
    for c in node.dhd.recourse_copies
        @test lower_bound(c.theta) == 0.0
        @test upper_bound(c.theta) == 0.0
    end
    return
end

function test_joint_objective()
    node = _two_stage_model()[1]
    obj = objective_function(node.dhd.joint)
    ub = node.dhd.first_stage[node.subproblem[:ub]]
    @test coefficient(obj, ub) ≈ 2.0
    for c in node.dhd.recourse_copies
        ua = c.variables[node.subproblem[:ua]]
        @test coefficient(obj, c.theta) ≈ c.noise.probability
        @test coefficient(obj, ua) ≈ 3 * c.noise.probability
    end
    return
end

"Add the cut θ ≥ height - slope (x' - point) to `node` and return it."
function _add_test_cut(node; height = 10.0, slope = -2.0, point = 3.0)
    V = node.bellman_function.global_theta
    SDDP._add_cut(V, height, Dict(:x => slope), Dict(:x => point), nothing, nothing)
    return last(V.cuts)
end

function test_cut_added_to_template_and_copies()
    node = _two_stage_model()[1]
    F, S = AffExpr, MOI.GreaterThan{Float64}
    @test num_constraints(node.subproblem, F, S) == 0
    @test num_constraints(node.dhd.joint, F, S) == 0
    cut = _add_test_cut(node)
    @test num_constraints(node.subproblem, F, S) == 1 # one cut for template
    @test num_constraints(node.dhd.joint, F, S) == 2  # one cut for each recourse copy
    @test length(node.dhd.cut_constraints[cut]) == 2  # one row per recourse copy
    return
end

function test_cut_rows_use_copy_variables()
    node = _two_stage_model()[1]
    cut = _add_test_cut(node)
    x_out = node.subproblem[:x].out
    rows = node.dhd.cut_constraints[cut]
    for (c, row) in zip(node.dhd.recourse_copies, rows)
        f = constraint_object(row).func # θ + 2x′ ≥ 16.
        @test length(f.terms) == 2
        @test coefficient(f, c.theta) == 1.0
        @test coefficient(f, c.variables[x_out]) == 2.0
        @test normalized_rhs(row) == 16.0
    end
    return
end

function test_cut_sense_max()
    node = _two_stage_model(;
        sense = :Max,
        lower_bound = -Inf,
        upper_bound = 10.0
    )[1]
    _add_test_cut(node)
    joint = node.dhd.joint
    @test num_constraints(joint, AffExpr, MOI.LessThan{Float64}) == 2
    @test num_constraints(joint, AffExpr, MOI.GreaterThan{Float64}) == 0
    return
end

function test_multi_cut_error()
    node = _two_stage_model()[1]
    SDDP._add_locals_if_necessary(node, node.bellman_function, 2)
    V = node.bellman_function.local_thetas[1]
    @test_throws(
        ErrorException("Multi-cut is not supported for DHD nodes."),
        SDDP._add_cut(V, 10.0, Dict(:x => -2.0), Dict(:x => 3.0), nothing, nothing)
    )
    return
end

function test_cut_deleted_from_copies()
    node = _two_stage_model()[1]
    joint = node.dhd.joint
    F, S = AffExpr, MOI.GreaterThan{Float64}
    A = _add_test_cut(node) # θ ≥ 16 − 2x'
    rows_A = node.dhd.cut_constraints[A]
    B = _add_test_cut(node; height = 20.0) # θ ≥ 26 − 2x'
    # B is higher than A at x' = 3
    @test A.constraint_ref === nothing
    @test num_constraints(node.subproblem, F, S) == 1
    @test num_constraints(joint, F, S) == 2
    @test !any(is_valid(joint, row) for row in rows_A)
    @test !haskey(node.dhd.cut_constraints, A)
    @test haskey(node.dhd.cut_constraints, B)
    return
end

function test_deletion_minimum()
    node = _two_stage_model()[1]
    joint = node.dhd.joint
    F, S = AffExpr, MOI.GreaterThan{Float64}
    node.bellman_function.global_theta.deletion_minimum = 2
    A = _add_test_cut(node)
    B = _add_test_cut(node; height = 20.0)
    # Only A is dominated, so nothing is deleted yet.
    @test num_constraints(node.subproblem, F, S) == 2
    @test num_constraints(joint, F, S) == 4
    C = _add_test_cut(node; height = 30.0)
    # A and B are dominated, so both are deleted.
    @test num_constraints(node.subproblem, F, S) == 1
    @test num_constraints(joint, F, S) == 2
    @test collect(keys(node.dhd.cut_constraints)) == [C]
    return
end

function test_deleted_cut_readded()
    node = _two_stage_model()[1]
    joint = node.dhd.joint
    F, S = AffExpr, MOI.GreaterThan{Float64}
    A = _add_test_cut(node) # θ ≥ 16 − 2x'
    _add_test_cut(node; height = 20.0) # θ ≥ 26 − 2x'
    # A is deleted.
    @test !haskey(node.dhd.cut_constraints, A)
    C = _add_test_cut(node; height = 0.0, slope = 0.0, point = 0.0) # θ ≥ 0
    # At x' = 0, A is higher than C so A is readded and C is deleted.
    @test A.constraint_ref !== nothing
    @test C.constraint_ref === nothing
    @test num_constraints(node.subproblem, F, S) == 2 # A and B
    @test num_constraints(joint, F, S) == 4
    @test haskey(node.dhd.cut_constraints, A)
    @test !haskey(node.dhd.cut_constraints, C)
    @test all(is_valid(joint, row) for row in node.dhd.cut_constraints[A])
    return
end

function test_cut_file_round_trip()
    model = _two_stage_model()
    node = model[1]
    # Cut 1: θ ≥ 16 - 2x' with point = 3
    # Cut 2: θ ≥ -4 + 2x' with point = 8
    # Cut 1 is 10 at x' = 3 and 0 at x' = 8.
    # Cut 2 is 2 at x' = 3 and 12 at x' = 8.
    # So they are both kept.
    _add_test_cut(node)
    _add_test_cut(node; height = 12.0, slope = 2.0, point = 8.0)
    filename = tempname()
    SDDP.write_cuts_to_file(model, filename)
    new_model = _two_stage_model()
    SDDP.read_cuts_from_file(new_model, filename)
    rm(filename)
    new_node = new_model[1]
    F, S = AffExpr, MOI.GreaterThan{Float64}
    @test num_constraints(new_node.subproblem, F, S) == 2
    @test num_constraints(new_node.dhd.joint, F, S) == 4
    V = node.bellman_function.global_theta
    new_V = new_node.bellman_function.global_theta
    for (cut, new_cut) in zip(V.cuts, new_V.cuts)
        @test new_cut.intercept ≈ cut.intercept
        @test new_cut.coefficients[:x] ≈ cut.coefficients[:x]
        rows = new_node.dhd.cut_constraints[new_cut]
        @test all(normalized_rhs(row) ≈ cut.intercept for row in rows)
    end
    return
end

function test_add_all_cuts()
    model = _two_stage_model()
    node = model[1]
    joint = node.dhd.joint
    F, S = AffExpr, MOI.GreaterThan{Float64}
    A = _add_test_cut(node)
    _add_test_cut(node; height = 20.0)
    @test num_constraints(joint, F, S) == 2 # A is deleted.
    SDDP.add_all_cuts(model) # Restores the deleted A.
    @test A.constraint_ref !== nothing
    @test num_constraints(node.subproblem, F, S) == 2
    @test num_constraints(joint, F, S) == 4
    @test all(is_valid(joint, row) for row in node.dhd.cut_constraints[A])
    return
end


"Simple hydro with two DHD stages, or its three-stage DH-aug twin."
function _two_stage_hydro(; dhd::Bool)
    return SDDP.LinearPolicyGraph(;
        stages = dhd ? 2 : 3,
        lower_bound = 0.0
    ) do sp, t
        @variable(sp, 0 <= x_storage <= 8, SDDP.State, initial_value = 6)
        if dhd
            @variable(sp, 0 <= u_thermal <= 9, SDDP.HereAndNow)
            thermal = u_thermal
        else
            @variable(sp, 0 <= u_thermal <= 9, SDDP.State, initial_value = 0)
            thermal = u_thermal.in
            if t == 1
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

function test_two_stage_hydro_value()
    model = _two_stage_hydro(; dhd = true)
    set_optimizer(model, HiGHS.Optimizer)
    node = model[1]
    # Manually adding the stage 2 cost-to-go: V(x') = max(0, 140 - 20x').
    SDDP._add_cut(
        node.bellman_function.global_theta,
        140.0,
        Dict(:x_storage => -20.0),
        Dict(:x_storage => 0.0),
        nothing,
        nothing,
    )
    joint = node.dhd.joint
    # Manually setting initial incoming state value for the joint subproblem.
    fix(node.dhd.first_stage[node.subproblem[:x_storage].in], 6.0)
    optimize!(joint) # Optimal value = 150 with u ∈ [1, 7]
    @test is_solved_and_feasible(joint)
    # Compare the joint solve with hand calculation.
    @test objective_value(joint) ≈ 150
    u = value(node.dhd.first_stage[node.subproblem[:u_thermal]])
    @test 1 - 1e-6 <= u <= 7 + 1e-6
    for c in node.dhd.recourse_copies
        x_out = value(c.variables[node.subproblem[:x_storage].out])
        @test value(c.theta) ≈ max(0, 140 - 20 * x_out) atol = 1e-6
    end
    # Compare the joint solve with SDDP's deterministic equivalent.
    det = SDDP.deterministic_equivalent(
        _two_stage_hydro(; dhd = false),
        HiGHS.Optimizer
    )
    set_silent(det)
    optimize!(det)
    @test objective_value(det) ≈ objective_value(joint)
    return
end

end # module

TestDHDBellmanFunctions.runtests()

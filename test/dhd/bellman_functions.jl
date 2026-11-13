module TestDHDBellmanFunctions

using SDDP
using Test

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

end # module

TestDHDBellmanFunctions.runtests()
module TestDHDDeterministicEquivalent

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

"Simple hydro from the examples with `stages` DHD stages, or its DH-aug twin."
function _simple_hydro(; stages::Int, dhd::Bool, Ω = [2.0, 3.0], voll = 500)
    return SDDP.LinearPolicyGraph(;
        stages = dhd ? stages : stages + 1,
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer,
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
        SDDP.parameterize(ω -> fix(w_inflow, ω), sp, Ω)
        @constraint(sp, thermal + u_hydro == 9 - u_unmet_demand)
        @constraint(
            sp,
            x_storage.out == x_storage.in - u_hydro - u_spill + w_inflow
        )
        @stageobjective(sp, voll * u_unmet_demand + 20 * thermal)
    end
end

"Solve the deterministic equivalent of `model` and return the JuMP model."
function _solve_deterministic_equivalent(model)
    det = SDDP.deterministic_equivalent(model, HiGHS.Optimizer)
    set_silent(det)
    optimize!(det)
    return det
end

function test_one_stage()
    model = _simple_hydro(; stages = 1, dhd = true)
    det = _solve_deterministic_equivalent(model)
    # The one-stage joint subproblem at x = 6 commits u = 1.
    @test objective_value(det) ≈ 20
    ret = SDDP.solve_joint_subproblem(
        model,
        model[1],
        Dict(:x_storage => 6.0);
        duality_handler = nothing,
    )
    @test objective_value(det) ≈ ret.objective
    return
end

function test_two_stages()
    model = _simple_hydro(; stages = 2, dhd = true)
    det = _solve_deterministic_equivalent(model)
    @test objective_value(det) ≈ 150
    return
end

function test_siblings_share_here_and_now()
    model = _simple_hydro(; stages = 2, dhd = true)
    det = _solve_deterministic_equivalent(model)
    # Node 1 is copied once per inflow, but it commits a single value.
    copies = filter(x -> name(x) == "u_thermal#1", all_variables(det))
    @test length(copies) == 2
    @test value(copies[1]) ≈ value(copies[2])
    F, S = AffExpr, MOI.EqualTo{Float64}
    # 2 equality constraints in each of 6 tree nodes: 2 * 6.
    # 4 linking constraints (node 2's x.in == node 1's x.out).
    # Non-anticipativity for root and each node-1 copy: 3.
    @test num_constraints(det, F, S) == 2 * 6 + 4 + 3.
    return
end

function test_matches_dh_aug()
    for (Ω, voll) in [([2.0, 3.0], 500), ([0.0, 2.0, 4.0], 50)]
        dhd = _simple_hydro(; stages = 4, dhd = true, Ω, voll)
        dh_aug = _simple_hydro(; stages = 4, dhd = false, Ω, voll)
        @test objective_value(_solve_deterministic_equivalent(dhd)) ≈
              objective_value(_solve_deterministic_equivalent(dh_aug))
    end
    # The tutorial's decision-hazard model has optimum = 410.
    det = _solve_deterministic_equivalent(_simple_hydro(; stages = 4, dhd = true))
    @test objective_value(det) ≈ 410
    return
end

function test_fresh_model_error()
    model = _simple_hydro(; stages = 2, dhd = true)
    SDDP.simulate(model, 1)
    @test_throws(
        ErrorException(
            "Unable to formulate deterministic equivalent: " *
            "Here-and-now variables were fixed by an earlier solve. " *
            "Can only form deterministic equivalent on a fresh model."
        ),
        SDDP.deterministic_equivalent(model, HiGHS.Optimizer)
    )
    return
end

end # module

TestDHDDeterministicEquivalent.runtests()

module TestHereAndNow

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

function test_HereAndNow_bounds()
    model = SDDP.LinearPolicyGraph(; stages = 2, lower_bound = 0.0) do sp, t
        @variable(sp, x >= 0, SDDP.State, initial_value = 0.0)
        @variable(sp, 0 <= u <= 9, SDDP.HereAndNow)
    end
    for t in 1:2
        u = model[t].subproblem[:u]
        @test model[t].here_and_now == [u]
        @test JuMP.lower_bound(u) == 0.0
        @test JuMP.upper_bound(u) == 9.0
    end
    return
end

function test_HerAndNow_container()
    model = SDDP.LinearPolicyGraph(; stages = 1, lower_bound = 0.0) do sp, t
        @variable(sp, x >= 0, SDDP.State, initial_value = 0.0)
        @variable(sp, u[1:3] >= 0, SDDP.HereAndNow)
    end
    @test model[1].here_and_now == model[1].subproblem[:u]
    return
end

function test_HereAndNow_integer()
    model = SDDP.LinearPolicyGraph(; stages = 1, lower_bound = 0.0) do sp, t
        @variable(sp, x >= 0, SDDP.State, initial_value = 0.0)
        @variable(sp, u, Int, SDDP.HereAndNow)
    end
    @test is_integer(model[1].subproblem[:u])
    return
end

function test_no_HereAndNow()
    model = SDDP.LinearPolicyGraph(; stages = 1, lower_bound = 0.0) do sp, t
        @variable(sp, x >= 0, SDDP.State, initial_value = 0.0)
    end
    @test isempty(model[1].here_and_now)
    return
end

function test_HereAndNow_unknown_keyword()
    builder(sp, t) = @variable(sp, u, SDDP.HereAndNow; unknown_keyword = 1)
    @test_throws ErrorException SDDP.LinearPolicyGraph(builder; stages = 1, lower_bound = 0.0)
    return
end

end # module

TestHereAndNow.runtests()
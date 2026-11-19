module TestDHDValidation

using SDDP
using Test
import HiGHS

module Hydro
include(joinpath(@__DIR__, "..", "..", "examples", "hydro_models.jl"))
end

module Batteries
include(joinpath(@__DIR__, "..", "..", "examples", "batteries_models.jl"))
end

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

"Exact optimum of `model` from its deterministic equivalent."
function _exact(model)
    det = SDDP.deterministic_equivalent(model, HiGHS.Optimizer)
    set_silent(det)
    optimize!(det)
    return objective_value(det)
end

"Expected cost of the policy in `model` over every equally likely noise path."
function _policy_cost(model, Ω, stages)
    total = 0.0
    paths = Iterators.product(ntuple(_ -> Ω, stages)...) # Ω × Ω × Ω × Ω
    for w in paths
        path = [(t, w[t]) for t in 1:stages]
        scheme = SDDP.Historical(path)
        sim = only(SDDP.simulate(model, 1; sampling_scheme = scheme))
        total += sum(s[:stage_objective] for s in sim)
    end
    return total / length(paths)
end

"Check that `dhd()` and `dh_aug()` agree and that training finds the optimum."
function _validate(dhd, dh_aug, Ω, stages; iteration_limit)
    exact = _exact(dh_aug())
    @test _exact(dhd()) ≈ exact rtol = 1e-6
    model = dhd()
    SDDP.train(model; iteration_limit, print_level = 0)
    @test SDDP.calculate_bound(model) ≈ exact rtol = 1e-6
    @test _policy_cost(model, Ω, stages) ≈ exact rtol = 1e-6
    return exact
end

function test_simple_hydro()
    exact = _validate(
        () -> Hydro.simple_hydro_dhd(),
        () -> Hydro.simple_hydro_dh_aug(),
        Hydro.SIMPLE_INFLOW,
        4;
        iteration_limit = 10,
    )
    @test exact ≈ 410 # Documented optimum from the tutorial.
    return
end

function test_seasonal_hydro()
    years = 2
    _validate(
        () -> Hydro.seasonal_hydro_dhd(; years),
        () -> Hydro.seasonal_hydro_dh_aug(; years),
        Hydro.SEASONAL_NOISE,
        Hydro.N_SEASONS * years;
        iteration_limit = 150,
    )
    return
end

function test_batteries()
    T, Ω = 6, [-4.0, 0.0, 4.0]
    _validate(
        () -> Batteries.batteries_dhd(; T, Ω),
        () -> Batteries.batteries_dh_aug(; T, Ω),
        Ω,
        T;
        iteration_limit = 300,
    )
    return
end

end # module

TestDHDValidation.runtests()

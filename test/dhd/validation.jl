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

"Expected cost of the policy in `model` over every path of its acyclic graph."
function _policy_cost(model::SDDP.PolicyGraph{T}) where {T}
    return _path_cost(model, model.root_children, Tuple{T,Any}[], 1.0)
end

"Expected cost of the paths that continue `path`, which has `probability`."
function _path_cost(model, children, path, probability)
    total = 0.0
    for child in children
        node = model[child.term]
        for noise in node.noise_terms
            p = probability * child.probability * noise.probability
            next_path = vcat(path, [(child.term, noise.term)])
            if isempty(node.children)
                scheme = SDDP.Historical(next_path)
                sim = only(SDDP.simulate(model, 1; sampling_scheme = scheme))
                total += p * sum(s[:stage_objective] for s in sim)
            else
                total += _path_cost(model, node.children, next_path, p)
            end
        end
    end
    return total
end

"Check that `dhd()` matches `reference()` and that training finds the optimum."
function _validate(dhd, reference; iteration_limit, kwargs...)
    exact = _exact(reference())
    @test _exact(dhd()) ≈ exact rtol = 1e-6
    model = dhd()
    SDDP.train(model; iteration_limit, print_level = 0, kwargs...)
    @test SDDP.calculate_bound(model) ≈ exact rtol = 1e-6
    @test _policy_cost(model) ≈ exact rtol = 1e-6
    return exact
end

function test_simple_hydro()
    exact = _validate(
        () -> Hydro.simple_hydro_dhd(),
        () -> Hydro.simple_hydro_dh_aug();
        iteration_limit = 10,
    )
    @test exact ≈ 410 # Documented optimum from the tutorial.
    return
end

function test_seasonal_hydro()
    years = 2
    _validate(
        () -> Hydro.seasonal_hydro_dhd(; years),
        () -> Hydro.seasonal_hydro_dh_aug(; years);
        iteration_limit = 150,
    )
    return
end

function test_batteries()
    T, Ω = 6, [-4.0, 0.0, 4.0]
    _validate(
        () -> Batteries.batteries_dhd(; T, Ω),
        () -> Batteries.batteries_dh_aug(; T, Ω);
        iteration_limit = 300,
    )
    return
end

function test_markovian_hydro()
    # Train with and without sharing cuts between with the same children;
    # by default, cut sharing is on.
    for refine_at_similar_nodes in (true, false)
        exact = _validate(
            () -> Hydro.markovian_hydro_dhd(),
            () -> Hydro.markovian_hydro_split();
            iteration_limit = 30,
            refine_at_similar_nodes,
        )
        @test exact ≈ 285
    end
    return
end

function test_markovian_hydro_cut_sharing()
    # Shared cuts give every node of a stage its exact cut in one iteration.
    model = Hydro.markovian_hydro_dhd()
    SDDP.train(model; iteration_limit = 1, print_level = 0)
    @test SDDP.calculate_bound(model) ≈ 285 rtol = 1e-6
    return
end

end # module

TestDHDValidation.runtests()

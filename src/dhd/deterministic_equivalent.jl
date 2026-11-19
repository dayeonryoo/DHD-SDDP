"Error if a recourse solve has fixed the here-and-now variables of `node`."
function _throw_if_here_and_now_fixed(node::Node)
    if any(JuMP.is_fixed, node.here_and_now)
        throw_detequiv_error(
            "Here-and-now variables were fixed by an earlier solve. " *
            "Can only form deterministic equivalent on a fresh model."
        )
    end
    return
end

"Make siblings from the same DHD node share their here-and-now values."
function _add_non_anticipativity_constraints(
    model::JuMP.Model,
    siblings::Vector{ScenarioTreeNode{T}},
) where {T}
    groups = Dict{T, Vector{ScenarioTreeNode{T}}}()
    for child in siblings
        push!(get!(groups, child.node.index, ScenarioTreeNode{T}[]), child)
    end
    for group in values(groups)
        first_child = first(group)
        for child in group[2:end]
            for (u, dest) in child.here_and_now
                @constraint(model, dest == first_child.here_and_now[u])
            end
        end
    end
    return
end

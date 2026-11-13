"One noise copy in the joint model: its noise, variable map and stage cost."
struct RecourseCopy
    noise::Noise
    variables::Dict{JuMP.VariableRef,JuMP.VariableRef}
    stage_objective::Any
end

"Joint subproblem of a DHD node."
struct DHDData
    joint::JuMP.Model
    first_stage::Dict{JuMP.VariableRef,JuMP.VariableRef}
    recourse_copies::Vector{RecourseCopy}
end

"Return true if `node` has here-and-now variables."
is_dhd(node::Node) = node.dhd !== nothing

"Error if `model` has DHD nodes since `fname` does not support them yet."
function _throw_if_dhd(model::PolicyGraph, fname::String)
    if any(is_dhd, values(model.nodes))
        error("`SDDP.$fname` does not support DHD nodes yet.")
    end
    return
end

"Return true if `f` contains only first-stage variables."
_is_first_stage(f::Real, first_stage) = true

_is_first_stage(f::JuMP.VariableRef, first_stage) = haskey(first_stage, f)

function _is_first_stage(f::JuMP.GenericAffExpr, first_stage)
    return all(haskey(first_stage, x) for x in keys(f.terms))
end

function _is_first_stage(f::JuMP.GenericQuadExpr, first_stage)
    return _is_first_stage(f.aff, first_stage) && all(
        haskey(first_stage, p.a) && haskey(first_stage, p.b) for
        p in keys(f.terms)
    )
end

function _is_first_stage(f::AbstractVector, first_stage)
    return all(_is_first_stage(fi, first_stage) for fi in f)
end

_is_first_stage(f::Any, first_stage) = false

"Error if parameterize changed a first-stage constraint."
function _check_unchanged(con, obj, first_stage_constraints)
    old = get(first_stage_constraints, con, nothing)
    if old === nothing || !isequal(old.func, obj.func) || old.set != obj.set
        error(
            "`parameterize` must not modify `$con`: it contains only " *
            "here-and-now variables and incoming states.",
        )
    end
    return
end

"Expand the HD `node` to a DHD model if it has here-and-now variables,
by building a joint subproblem with recourse copies for each noise term."
function _expand_to_dhd!(node::Node, optimizer, direct_mode::Bool)
    if isempty(node.here_and_now)
        return
    end
    if node.objective_state !== nothing
        error("Objective states are not supported for DHD nodes.")
    end

    template = node.subproblem
    joint = construct_subproblem(optimizer, direct_mode)
    joint.ext[:sddp_policy_graph] = template.ext[:sddp_policy_graph]
    joint.ext[:sddp_node] = node
    JuMP.set_objective_sense(joint, JuMP.objective_sense(template))

    # First-stage variables are copied to the joint model.
    first_stage = Dict{JuMP.VariableRef,JuMP.VariableRef}()
    for state in values(node.states)
        first_stage[state.in] = @variable(joint, base_name = name(state.in))
    end
    for u in node.here_and_now
        first_stage[u] = @variable(joint, base_name = name(u))
    end

    # Recourse copies are created for each noise term.
    first_stage_constraints = Dict{JuMP.ConstraintRef,Any}()
    recourse_copies = RecourseCopy[]
    for (k, noise) in enumerate(node.noise_terms)
        node.parameterize(noise.term)
        variables = copy(first_stage)
        for x in JuMP.all_variables(template)
            if !haskey(variables, x)
                variables[x] = @variable(joint, base_name = "$(name(x))#$k")
            end
        end
        for (F, S) in JuMP.list_of_constraint_types(template)
            for con in JuMP.all_constraints(template, F, S)
                obj = JuMP.constraint_object(con)
                if _is_first_stage(obj.func, first_stage)
                    if k > 1
                        _check_unchanged(con, obj, first_stage_constraints)
                        continue
                    else
                        first_stage_constraints[con] = obj
                    end
                end
                f = copy_and_replace_variables(obj.func, variables)
                @constraint(joint, f in obj.set)
            end
        end
        C = copy_and_replace_variables(node.stage_objective, variables)
        push!(recourse_copies, RecourseCopy(noise, variables, C))
    end

    JuMP.set_objective_function(
        joint,
        sum(c.noise.probability * c.stage_objective for c in recourse_copies),
    )
    node.dhd = DHDData(joint, first_stage, recourse_copies)
    return
end
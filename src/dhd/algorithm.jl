"Attach the node's optimizer to the joint subproblem if it has none."
function _initialize_joint_solver(node::Node)
    if !is_dhd(node)
        return
    end
    joint = node.dhd.joint
    if JuMP.mode(joint) == JuMP.DIRECT
        return
    elseif MOI.Utilities.state(JuMP.backend(joint)) != MOIU.NO_OPTIMIZER
        return
    end
    JuMP.set_optimizer(joint, node.optimizer)
    JuMP.set_silent(joint)
    return
end

"Drop the optimizer of the joint subproblem."
function _uninitialize_joint_solver(node::Node)
    if !is_dhd(node)
        return
    end
    joint = node.dhd.joint
    if JuMP.mode(joint) == JuMP.DIRECT
        return
    elseif MOI.Utilities.state(JuMP.backend(joint)) != MOIU.NO_OPTIMIZER
        MOI.Utilities.drop_optimizer(joint)
    end
    return
end

"Return true if `joint` has a primal solution, and a dual one if required."
function _has_joint_solution(joint::JuMP.Model; require_dual::Bool)
    ok = (JuMP.FEASIBLE_POINT, JuMP.NEARLY_FEASIBLE_POINT)
    if !(JuMP.primal_status(joint) in ok)
        return false
    end
    return !require_dual || (JuMP.dual_status(joint) in ok)
end

"Solve the joint subproblem, and retry once from scratch if it has no solution."
function _optimize_joint!(model::PolicyGraph, node::Node; require_dual::Bool)
    joint = node.dhd.joint
    JuMP.optimize!(joint)
    lock(model.lock) do
        model.ext[:total_solves] = get(model.ext, :total_solves, 0) + 1
        return
    end
    if JuMP.termination_status(joint) == MOI.INTERRUPTED
        throw(InterruptException())
    end
    if _has_joint_solution(joint; require_dual)
        return
    end
    # Retrying once from scratch.
    model.ext[:numerical_issue] = true
    if JuMP.mode(joint) != JuMP.DIRECT
        MOI.Utilities.reset_optimizer(joint)
        JuMP.optimize!(joint)
    end
    # If retryinig fails, throws an error.
    if !_has_joint_solution(joint; require_dual)
        error(
            "Unable to solve the joint subproblem of node $(node.index).\n" *
            "   Termination status : $(JuMP.termination_status(joint))\n" *
            "   Primal status      : $(JuMP.primal_status(joint))\n" *
            "   Dual status        : $(JuMP.dual_status(joint))",
        )
    end
    return
end

"Return no duals when no duality handler is given."
_joint_duals(::Node, ::Nothing) = Dict{Symbol,Float64}()

"Return the duals of the fixed incoming states in the joint subproblem."
function _joint_duals(node::Node, ::ContinuousConicDuality)
    joint = node.dhd.joint
    sign = JuMP.objective_sense(joint) == MOI.MIN_SENSE ? 1 : -1
    return Dict{Symbol,Float64}(
        name => sign * JuMP.dual(JuMP.FixRef(node.dhd.first_stage[state.in]))
        for (name, state) in node.states
    )
end

"LagrangianDuality, StrengthenedConicDuality, FixedDiscreteDuality, BadnitDuality."
function _joint_duals(::Node, handler::AbstractDualityHandler)
    return error("`$(typeof(handler))` is not supported for DHD nodes yet.")
end

"Solve the joint subproblem of `node` at the incoming `state`."
function solve_joint_subproblem(
    model::PolicyGraph,
    node::Node,
    state::Dict{Symbol,Float64};
    duality_handler::Union{Nothing,AbstractDualityHandler},
)
    _initialize_solver(node; throw_error = false)
    # Fix x.in.
    for (name, value) in state
        JuMP.fix(node.dhd.first_stage[node.states[name].in], value)
    end
    _optimize_joint!(model, node; require_dual = duality_handler !== nothing)
    here_and_now = Dict{JuMP.VariableRef,Float64}(
        u => JuMP.value(node.dhd.first_stage[u]) for u in node.here_and_now
    )
    return (
        here_and_now = here_and_now,
        objective = JuMP.objective_value(node.dhd.joint),
        duals = _joint_duals(node, duality_handler),
    )
end

"Fix the here-and-now variables of the template at the given values."
function _fix_here_and_now!(
    node::Node,
    here_and_now::Dict{JuMP.VariableRef,Float64},
)
    for (u, value) in here_and_now
        JuMP.fix(u, value; force = true)
    end
    return
end

"Solve the recourse subproblem of `node` with the here-and-now values fixed."
function solve_recourse_subproblem(
    model::PolicyGraph,
    node::Node,
    state::Dict{Symbol,Float64},
    here_and_now::Dict{JuMP.VariableRef,Float64},
    noise,
    scenario_path,
)
    _fix_here_and_now!(node, here_and_now)
    return solve_subproblem(
        model,
        node,
        state,
        noise,
        scenario_path;
        duality_handler = nothing,
    )
end

"Solve the joint subproblem, then the recourse subproblem for `noise`."
function solve_dhd_node(
    model::PolicyGraph,
    node::Node,
    state::Dict{Symbol,Float64},
    noise,
    scenario_path,
)
    joint = solve_joint_subproblem(model, node, state; duality_handler = nothing)
    return solve_recourse_subproblem(
        model,
        node,
        state,
        joint.here_and_now,
        noise,
        scenario_path,
    )
end

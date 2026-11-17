"Add a cost-to-go variable to each recourse copy and set the joint objective."
function _initialize_dhd_bellman_function!(node::Node)
    if !is_dhd(node)
        return
    end

    Θ = bellman_term(node.bellman_function)
    joint = node.dhd.joint
    for (l, c) in enumerate(node.dhd.recourse_copies)
        c.theta = @variable(joint, base_name = "Θ#$l")
        if JuMP.has_lower_bound(Θ)
            JuMP.set_lower_bound(c.theta, JuMP.lower_bound(Θ))
        end
        if JuMP.has_upper_bound(Θ)
            JuMP.set_upper_bound(c.theta, JuMP.upper_bound(Θ))
        end
    end

    JuMP.set_objective_function(
        joint,
        sum(
            c.noise.probability * (c.stage_objective + c.theta)
            for c in node.dhd.recourse_copies
        )
    )
    return
end

"Add `cut` to each recourse copy of the joint subproblem."
function _add_cut_to_recourse_copies!(V::ConvexApproximation, cut::Cut)
    node = get_node(JuMP.owner_model(V.theta))
    if !is_dhd(node)
        return
    end
    if V !== node.bellman_function.global_theta
        error("Multi-cut is not supported for DHD nodes.")
    end
    joint = node.dhd.joint
    is_min = JuMP.objective_sense(joint) == MOI.MIN_SENSE
    rows = JuMP.ConstraintRef[]
    for c in node.dhd.recourse_copies
        expr = @expression(
            joint,
            c.theta -
            sum(cut.coefficients[k] * c.variables[x] for (k, x) in V.states)
        )
        row = if is_min
            @constraint(joint, expr >= cut.intercept)
        else
            @constraint(joint, expr <= cut.intercept)
        end
        push!(rows, row)
    end
    node.dhd.cut_constraints[cut] = rows
    return
end

"Delete the rows of `cut`from each recourse copy of the joint subproblem."
function _delete_cut_from_recourse_copies!(V::ConvexApproximation, cut::Cut)
    node = get_node(JuMP.owner_model(V.theta))
    if !is_dhd(node)
        return
    end
    rows = pop!(node.dhd.cut_constraints, cut)
    for row in rows
        JuMP.delete(node.dhd.joint, row)
    end
    return
end

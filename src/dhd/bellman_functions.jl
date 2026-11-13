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

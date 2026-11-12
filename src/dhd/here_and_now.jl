"Tag for here-and-now variables: @variable(sp, u, SDDP.HereAndNow)."
struct HereAndNow end

struct HereAndNowInfo
    info::JuMP.VariableInfo
end

function JuMP.build_variable(
    _error::Function,
    info::JuMP.VariableInfo,
    ::Type{HereAndNow};
    kwargs...,
)
    if !isempty(kwargs)
        _error(
            "Unexpected keyword arguments: " * join(keys(kwargs), ". ")
        )
    end
    return HereAndNowInfo(info)
end

function JuMP.add_variable(
    subproblem::JuMP.Model,
    here_and_now_info::HereAndNowInfo,
    name::String,
)
    u = JuMP.add_variable(
        subproblem,
        JuMP.ScalarVariable(here_and_now_info.info),
        name,
    )
    node = get_node(subproblem)
    push!(node.here_and_now, u)
    return u
end
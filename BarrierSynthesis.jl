using SumOfSquares
using CSDP
using MosekTools
using DynamicPolynomials
using LinearAlgebra

function prepare_domain(x::Vector{<:Variable}, bounds::Vector{Vector{Float64}})
    poly_list = [ 
         @set(x[i] - l >= 0) ∩ @set(u - x[i] >= 0 ) ∩ @set((x[i] -l)*(u-x[i]) >= 0)
         for (i, (l, u)) in enumerate(bounds)
         ] 
    poly_list
 end
 
 function get_random(limits::Vector{Vector{Float64}}, g::Polynomial)
     function get_random_scalar(lb, ub )
         lb + rand()*(ub - lb) 
     end
     while (true)
         pt = [get_random_scalar(l[1], l[2]) for l in limits]
         if g(pt[1], pt[2]) >= 0
             continue
         else
             return pt
         end
     end
 end

 function get_random_multi(limits::Vector{Vector{Float64}}, g::Polynomial,  h::Polynomial)
    function get_random_scalar(lb, ub )
        lb + rand()*(ub - lb) 
    end
    while (true)
        pt = [get_random_scalar(l[1], l[2]) for l in limits]
        if g(pt[1], pt[2]) >= 0 && h(pt[1], pt[2]) >= 0
            continue
        else
            return pt
        end
    end
end
 
 function generate_barrier(x, u, bounds, g, vectorField, U, test_pts; max_degree=4,ϵ = 1, λ = 1, γ = 10.)
     solver = optimizer_with_attributes(CSDP.Optimizer)
     model = SOSModel(solver)
     dom_list = prepare_domain(x, bounds)
     dom = reduce( (s1, s2) -> s1 ∩ s2, dom_list)
     println("Domain: $dom")
     # negative inside the obstacle
     monos = monomials(x, 0:max_degree)
     N = length(monos) 
     @variable(model, -γ <= c[1:N] <= γ)
     B = polynomial(c[1:end], monos) 
     @constraint(model, B <= -ϵ, domain=dom ∩ @set(g >= 0) )
     B_dot = dot(differentiate(B,x), vectorField)
     B_dot_with_u = subs(B_dot, u => U)
     @constraint(model, B_dot_with_u >= λ * B, domain=dom)
     set_objective_sense(model, MOI.FEASIBILITY_SENSE)
     objective_fn = sum([B(pt...) for pt in test_pts])
     @objective(model, Max, objective_fn) # keep as many points outside the barrier as you can
     JuMP.optimize!(model)
     #stat = JuMP.primal_status(model)
     println(solution_summary(model))
     value(B)
 end

function findBarrierFixedControlInput_HybridDubins(x, ctrl::Vector{Float64}, g, dynamics, test_pts; max_degree=4,ϵ = 1., λ = 1., γ = 10.)
    function prepare_domain(var, lb, ub)
        dom = @set(var >= lb) ∩ @set(var <= ub) ∩ @set( (var-lb)*(ub-var) >= 0)
        return dom
    end
    solver = optimizer_with_attributes(CSDP.Optimizer)
    model = SOSModel(solver);
    # @polyvar(x[1:4]) # x[1] is x, x[2] is y, x[3] is θ, x[4] is ω
    # dynamics = [
    #     (ctrl -> [(2.0/π)*(x[3] + π/2.0)-0.2x[4],(-2.0/π)*(x[3]+π)-0.2x[4], ctrl[1], 0]),
    #     (ctrl -> [(2.0/π)*(x[3] + π/2.0)-0.2x[4],(2.0/π)*x[3]+0.2x[4], ctrl[1], 0]),
    #     (ctrl -> [(-2.0/π)*(x[3] - π/2.0)+0.2x[4],(2.0/π)*x[3]+0.2x[4], ctrl[1], 0]),
    #     (ctrl -> [(-2.0/π)*(x[3] - π/2.0)+0.2x[4],(-2.0/π)*(x[3]-π)-0.2x[4], ctrl[1], 0]),
    # ]
    # g = 0.1 - x[1]^2 - x[2]^2 # Obstacle: is a (x,y) ball of radius 0.1 around origin
    monos = monomials(x, 0:max_degree)
    N = length(monos) 
    @variable(model, -γ <= c[1:N] <= γ)
    B = polynomial(c[1:end], monos)
    dom =  prepare_domain(x[1], -10., 10.) ∩ prepare_domain(x[2], -10., 10.) ∩ prepare_domain(x[4], -1., 1.) ∩ prepare_domain(x[3], -π, π) #     plot!(xlims = limits, ylims = limits)

    #dom2 = @set(x[1]^2 + x[2]^2 >= K) ∩ prepare_domain(x[4], -1., 1.) #∩ prepare_domain(x[3], -π, π)
    @constraint(model, B <= -ϵ, domain = dom ∩ @set(g >= 0))
    #@constraint(model, B >= ϵ, domain = dom2)
    #dB = differentiate(B, x)

    for (i, dyn_i) in enumerate(dynamics)
        dyn_with_ctrl = dyn_i(ctrl)
        dBdt = dot(differentiate(B, x), dyn_with_ctrl)
        dom_i = prepare_domain(x[3], (i-3) * π/2, (i-2) * π/2)
        # dom_i = reduce( (s1, s2) -> s1 ∩ s2, dom_i)
        @constraint(model, dBdt >= λ * B, domain = dom ∩ dom_i)
    end

    set_objective_sense(model, MOI.FEASIBILITY_SENSE)
    objective_fn = sum([B(pt...) for pt in test_pts])
    @objective(model, Max, objective_fn) # keep as many points outside the barrier as you can
    JuMP.optimize!(model)
    stat = JuMP.primal_status(model)
    println(solution_summary(model))
    println("Status obtained: $(stat)")
    if stat != FEASIBLE_POINT
        return missing
    end
    return value(B)
end 

function findBarrierFixedControlInput_HybridCT(x, ctrl::Vector{Float64}, g, dynamics, vector_field, test_pts; max_degree=4,ϵ = 1., λ = 1., γ = 10.,ψ = 10.)
    function prepare_domain_hybrid(var, lb, ub)
        dom = @set(var >= lb) ∩ @set(var <= ub) ∩ @set( (var-lb)*(ub-var) >= 0)
        return dom
    end
    solver = optimizer_with_attributes(CSDP.Optimizer)
    model = SOSModel(solver);
    monos = monomials(x, 0:max_degree)
    N = length(monos) 
    @variable(model, -γ <= c[1:N] <= γ)
    B = polynomial(c[1:end], monos)
    # dom =  @set(4 - x[1]^2 - x[2]^2 >= 0) ∩ prepare_domain(x[6], -1., 1.)
    # dom =  prepare_domain(x[6], -1., 1.) ∩ prepare_domain(x[4], -π, π)
    dom_list = prepare_domain(x, bounds)
    dom = reduce( (s1, s2) -> s1 ∩ s2, dom_list)
    println("Domain: $dom")

    @constraint(model, B <= -ϵ, domain = dom ∩ @set(g >= 0))
    # dB = differentiate(B, x)

    for (i, dyn_i) in enumerate(dynamics)
        dyn_with_ctrl = dyn_i(ctrl)
        dBdt = dot(differentiate(B, x), dyn_with_ctrl)
        dom_i = prepare_domain_hybrid(x[4], (i-3) * π/2, (i-2) * π/2)
        @constraint(model, dBdt >= λ * B, domain = dom ∩ dom_i)
        # @constraint(model, dBdt >= - ψ * norm(x), domain = dom ∩ dom_i) #dwell time constraint
    end

    set_objective_sense(model, MOI.FEASIBILITY_SENSE)
    objective_fn = sum([B(pt...) for pt in test_pts])
    @objective(model, Max, objective_fn) # keep as many points outside the barrier as you can
    JuMP.optimize!(model)
    stat = JuMP.primal_status(model)
    solution_summary(model)
    println("Status obtained: $(stat)")
    if stat != FEASIBLE_POINT
        return missing
    else
        B_dot = []
        for (vector_field_j) in vector_field
            B_dot_j = dot(differentiate(B,x), vector_field_j)
            push!(B_dot, value(B_dot_j))
        end
    end
    return value(B),B_dot
end

function findBarrierFixedControlInput_HybridCT_multi(x, ctrl::Vector{Float64}, g, h, dynamics, test_pts; max_degree=4,ϵ = 1., λ = 1., γ = 10.)
    function prepare_domain_hybrid(var, lb, ub)
        dom = @set(var >= lb) ∩ @set(var <= ub) ∩ @set( (var-lb)*(ub-var) >= 0)
        return dom
    end
    solver = optimizer_with_attributes(CSDP.Optimizer)
    model = SOSModel(solver);
    monos = monomials(x, 0:max_degree)
    N = length(monos) 
    @variable(model, -γ <= c[1:N] <= γ)
    B = polynomial(c[1:end], monos)
    # dom =  @set(4 - x[1]^2 - x[2]^2 >= 0) ∩ prepare_domain(x[6], -1., 1.)
    # dom =  prepare_domain(x[6], -1., 1.) ∩ prepare_domain(x[4], -π, π)
    dom_list = prepare_domain(x, bounds)
    dom = reduce( (s1, s2) -> s1 ∩ s2, dom_list)
    println("Domain: $dom")

    @constraint(model, B <= -ϵ, domain = dom ∩ @set(g >= 0))
    @constraint(model, B <= -ϵ, domain = dom ∩ @set(h >= 0))
    dB = differentiate(B, x)

    for (i, dyn_i) in enumerate(dynamics)
        dyn_with_ctrl = dyn_i(ctrl)
        dBdt = dot(differentiate(B, x), dyn_with_ctrl)
        dom_i = prepare_domain_hybrid(x[4], (i-3) * π/2, (i-2) * π/2)
        @constraint(model, dBdt >= λ * B, domain = dom ∩ dom_i)
    end

    set_objective_sense(model, MOI.FEASIBILITY_SENSE)
    objective_fn = sum([B(pt...) for pt in test_pts])
    @objective(model, Max, objective_fn) # keep as many points outside the barrier as you can
    JuMP.optimize!(model)
    stat = JuMP.primal_status(model)
    solution_summary(model)
    println("Status obtained: $(stat)")
    if stat != FEASIBLE_POINT
        return missing
    end
    return value(B)
end

function findBarrierFixedControlInput_HybridPM(x, ctrl::Vector{Float64}, g, dynamics, vector_field, test_pts; max_degree=4,ϵ = 1., λ = 1., γ = 10.)
    function prepare_domain_hybrid(var, lb, ub)
        dom = @set(var >= lb) ∩ @set(var <= ub) ∩ @set( (var-lb)*(ub-var) >= 0)
        return dom
    end
    solver = optimizer_with_attributes(CSDP.Optimizer)
    model = SOSModel(solver);
    monos = monomials(x, 0:max_degree)
    N = length(monos) 
    @variable(model, -γ <= c[1:N] <= γ)
    B = polynomial(c[1:end], monos)
    # dom =  @set(4 - x[1]^2 - x[2]^2 >= 0) ∩ prepare_domain(x[6], -1., 1.)
    # dom =  prepare_domain(x[7], -1., 1.) ∩ prepare_domain(x[5], -π, π)
    dom_list = prepare_domain(x, bounds)
    println("Domain_list: $dom_list")
    dom = reduce( (s1, s2) -> s1 ∩ s2, dom_list)
    println("Domain: $dom")

    @constraint(model, B <= -ϵ, domain = dom ∩ @set(g >= 0))
    dB = differentiate(B, x)

    for (i, dyn_i) in enumerate(dynamics)
        dyn_with_ctrl = dyn_i(ctrl)
        dBdt = dot(differentiate(B, x), dyn_with_ctrl)
        dom_i = prepare_domain_hybrid(x[5], (i-3) * π/2, (i-2) * π/2)
        # dom_i = reduce( (s1, s2) -> s1 ∩ s2, dom_i)
        @constraint(model, dBdt >= λ * B, domain = dom ∩ dom_i)
    end

    set_objective_sense(model, MOI.FEASIBILITY_SENSE)
    objective_fn = sum([B(pt...) for pt in test_pts])
    @objective(model, Max, objective_fn) # keep as many points outside the barrier as you can
    JuMP.optimize!(model)
    stat = JuMP.primal_status(model)
    solution_summary(model)
    println("Status obtained: $(stat)")
    if stat != FEASIBLE_POINT
        return missing
    else
        B_dot = []
        for (vector_field_j) in vector_field
            B_dot_j = dot(differentiate(B,x), vector_field_j)
            push!(B_dot, value(B_dot_j))
        end
    end
    return value(B),B_dot
end

function generate_random_test_points(limits, B_ancestors, g, n;max_iters=10000)
    pts = []
    num_iters = 0
    println("\t Generating random test points")
    while(size(pts)[1] < n && num_iters < max_iters)
        num_iters += 1
        x = rand() * (limits[1][2] - limits[1][1]) + limits[1][1]
        y = rand() * (limits[2][2] - limits[2][1]) + limits[2][1]
        θ = rand() * (limits[3][2] - limits[3][1]) + limits[3][1]
        w = rand() * (limits[4][2] - limits[4][1]) + limits[4][1]
        if g(x, y) >= 0
            continue
        end
        if any([B_val(x, y, θ, w) >= 0 for B_val in B_ancestors])
            continue
        end
        push!(pts, (x, y, θ, w))
        
    end
    println("\t Done: with {$(size(pts)[1])} points")
    return pts
end

function findSuccessiveBarrierDB(x, ctrl::Vector{Float64},g, dynamics, test_pts,ancestors, pt_to_eliminate::Vector{Float64};
    max_degree=4,ϵ = 1, λ = 1, γ = 10.)
    function prepare_domain_hybrid(var, lb, ub)
        dom = @set(var >= lb) ∩ @set(var <= ub) ∩ @set((var - lb) * (ub - var) >= 0)
        return dom
    end

    solver = optimizer_with_attributes(CSDP.Optimizer, MOI.Silent() => true)
    model = SOSModel(solver)
    monos = monomials(x, 0:max_degree)
    N = length(monos) 
    @variable(model, -γ <= c[1:N] <= γ)
    B = polynomial(c[1:end], monos) 

    # dom =  prepare_domain(x[4], -1., 1.) ∩ prepare_domain(x[3], -π, π) #     plot!(xlims = limits, ylims = limits)

    dom_list = prepare_domain(x, bounds)
    dom = reduce( (s1, s2) -> s1 ∩ s2, dom_list)
    println("Domain: $dom")
    # negative inside the obstacle
    @constraint(model, B <= -ϵ, domain=dom ∩ @set(g >= 0) )


    for (B_val) in (ancestors)
        dom = dom ∩ @set(B_val <= ϵ)
    end

    
    for (i, dyn) in enumerate(dynamics)
        dyn_with_ctrl = dyn(ctrl)
        dBdt = dot(differentiate(B, x), dyn_with_ctrl)
        dom_i = prepare_domain_hybrid(x[3], (i-3) * π/2, (i-2) * π/2)

        new_domain = dom ∩ (reduce(∩, [@set(B <= 0.) for B in ancestors]))
        @constraint(model, dBdt >= λ * B, domain = new_domain ∩ dom_i)
        # println("Constraint added for domain: $(dom ∩ dom_i)")
    end
    
    # eliminate the point we would like to eliminate
    @constraint(model, B(pt_to_eliminate...) >= ϵ)

    set_objective_sense(model, MOI.FEASIBILITY_SENSE)
    objective_fn = sum([B(pt...) for pt in test_pts])
    @objective(model, Max, objective_fn) # keep as many points outside the barrier as you can
    JuMP.optimize!(model)
    stat = JuMP.primal_status(model)
    solution_summary(model)

    if stat != FEASIBLE_POINT
        return missing
    end
    println(solution_summary(model))

    return value(B)
end


function findSuccessiveBarrierCT(x, ctrl::Vector{Float64},g, dynamics, vector_field, test_pts,ancestors, pt_to_eliminate::Vector{Float64};
    max_degree=4,ϵ = 1, λ = 1, γ = 10., ψ  = 10.)
    function prepare_domain_hybrid(var, lb, ub)
        dom = @set(var >= lb) ∩ @set(var <= ub) ∩ @set((var - lb) * (ub - var) >= 0)
        return dom
    end

    solver = optimizer_with_attributes(CSDP.Optimizer, MOI.Silent() => true)
    model = SOSModel(solver)
    monos = monomials(x, 0:max_degree)
    N = length(monos) 
    @variable(model, -γ <= c[1:N] <= γ)
    B = polynomial(c[1:end], monos) 

    # dom =  prepare_domain(x[6], -1., 1.) ∩ prepare_domain(x[4], -π, π) #     plot!(xlims = limits, ylims = limits)
    dom_list = prepare_domain(x, bounds)
    dom = reduce( (s1, s2) -> s1 ∩ s2, dom_list)
    println("Domain: $dom")
    # negative inside the obstacle
    @constraint(model, B <= -ϵ, domain=dom ∩ @set(g >= 0) )


    for (B_val) in (ancestors)
        dom = dom ∩ @set(B_val <= ϵ)
    end

    
    for (i, dyn) in enumerate(dynamics)
        dyn_with_ctrl = dyn(ctrl)
        dBdt = dot(differentiate(B, x), dyn_with_ctrl)
        dom_i = prepare_domain_hybrid(x[4], (i-3) * π/2, (i-2) * π/2)

        new_domain = dom ∩ (reduce(∩, [@set(B <= 0.) for B in ancestors]))
        @constraint(model, dBdt >= λ * B, domain = new_domain ∩ dom_i)
        # @constraint(model, dBdt >= - ψ * norm(x), domain = dom ∩ dom_i) #dwell time constraint

        # println("Constraint added for domain: $(dom ∩ dom_i)")
    end
    # println("Domain: $new_domain")
    # eliminate the point we would like to eliminate
    @constraint(model, B(pt_to_eliminate...) >= ϵ)

    set_objective_sense(model, MOI.FEASIBILITY_SENSE)
    objective_fn = sum([B(pt...) for pt in test_pts])
    @objective(model, Max, objective_fn) # keep as many points outside the barrier as you can
    JuMP.optimize!(model)
    stat = JuMP.primal_status(model)
    solution_summary(model)

    if stat != FEASIBLE_POINT
        return missing
    else
        B_dot = []
        for (vector_field_j) in vector_field
            B_dot_j = dot(differentiate(B,x), vector_field_j)
            push!(B_dot, value(B_dot_j))
        end
    end
    println(solution_summary(model))

    return value(B), B_dot
end

function findSuccessiveBarrierCTmulti(x, ctrl::Vector{Float64},g,h, dynamics, test_pts,ancestors, pt_to_eliminate::Vector{Float64};
    max_degree=4,ϵ = 1, λ = 1, γ = 10.)
    function prepare_domain_hybrid(var, lb, ub)
        dom = @set(var >= lb) ∩ @set(var <= ub) ∩ @set((var - lb) * (ub - var) >= 0)
        return dom
    end

    solver = optimizer_with_attributes(CSDP.Optimizer, MOI.Silent() => true)
    model = SOSModel(solver)
    monos = monomials(x, 0:max_degree)
    N = length(monos) 
    @variable(model, -γ <= c[1:N] <= γ)
    B = polynomial(c[1:end], monos) 

    # dom =  prepare_domain(x[6], -1., 1.) ∩ prepare_domain(x[4], -π, π) #     plot!(xlims = limits, ylims = limits)
    dom_list = prepare_domain(x, bounds)
    dom = reduce( (s1, s2) -> s1 ∩ s2, dom_list)
    println("Domain: $dom")
    # negative inside the obstacle
    @constraint(model, B <= -ϵ, domain=dom ∩ @set(g >= 0) ∩ @set(h >= 0) )


    for (B_val) in (ancestors)
        dom = dom ∩ @set(B_val <= ϵ)
    end

    
    for (i, dyn) in enumerate(dynamics)
        dyn_with_ctrl = dyn(ctrl)
        dBdt = dot(differentiate(B, x), dyn_with_ctrl)
        dom_i = prepare_domain_hybrid(x[4], (i-3) * π/2, (i-2) * π/2)

        new_domain = dom ∩ (reduce(∩, [@set(B <= 0.) for B in ancestors]))
        @constraint(model, dBdt >= λ * B, domain = new_domain ∩ dom_i)
        # println("Constraint added for domain: $(dom ∩ dom_i)")
    end
    # println("Domain: $new_domain")
    # eliminate the point we would like to eliminate
    @constraint(model, B(pt_to_eliminate...) >= ϵ)

    set_objective_sense(model, MOI.FEASIBILITY_SENSE)
    objective_fn = sum([B(pt...) for pt in test_pts])
    @objective(model, Max, objective_fn) # keep as many points outside the barrier as you can
    JuMP.optimize!(model)
    stat = JuMP.primal_status(model)
    solution_summary(model)

    if stat != FEASIBLE_POINT
        return missing
    end
    println(solution_summary(model))

    return value(B)
end

function findSuccessiveBarrierPM(x, ctrl::Vector{Float64},g, dynamics, vector_field, test_pts,ancestors, pt_to_eliminate::Vector{Float64};
    max_degree=4,ϵ = 1, λ = 1, γ = 10.)
    function prepare_domain_hybrid(var, lb, ub)
        dom = @set(var >= lb) ∩ @set(var <= ub) ∩ @set((var - lb) * (ub - var) >= 0)
        return dom
    end

    solver = optimizer_with_attributes(CSDP.Optimizer, MOI.Silent() => true)
    model = SOSModel(solver)
    monos = monomials(x, 0:max_degree)
    N = length(monos) 
    @variable(model, -γ <= c[1:N] <= γ)
    B = polynomial(c[1:end], monos) 

    # dom =  prepare_domain(x[7], -1., 1.) ∩ prepare_domain(x[5], -π, π) #     plot!(xlims = limits, ylims = limits)
    dom_list = prepare_domain(x, bounds)
    dom = reduce( (s1, s2) -> s1 ∩ s2, dom_list)
    println("Domain: $dom")

    # negative inside the obstacle
    @constraint(model, B <= -ϵ, domain=dom ∩ @set(g >= 0) )


    for (B_val) in (ancestors)
        dom = dom ∩ @set(B_val <= ϵ)
    end

    
    for (i, dyn) in enumerate(dynamics)
        dyn_with_ctrl = dyn(ctrl)
        dBdt = dot(differentiate(B, x), dyn_with_ctrl)
        dom_i = prepare_domain_hybrid(x[5], (i-3) * π/2, (i-2) * π/2)
        # dom_i = reduce( (s1, s2) -> s1 ∩ s2, dom_i)

        new_domain = dom ∩ (reduce(∩, [@set(B <= 0.) for B in ancestors]))
        @constraint(model, dBdt >= λ * B, domain = new_domain ∩ dom_i)
        # println("Constraint added for domain: $(dom ∩ dom_i)")
    end
    # println("Domain: $new_domain")
    # eliminate the point we would like to eliminate
    @constraint(model, B(pt_to_eliminate...) >= ϵ)

    set_objective_sense(model, MOI.FEASIBILITY_SENSE)
    objective_fn = sum([B(pt...) for pt in test_pts])
    @objective(model, Max, objective_fn) # keep as many points outside the barrier as you can
    JuMP.optimize!(model)
    stat = JuMP.primal_status(model)
    solution_summary(model)

    if stat != FEASIBLE_POINT
        return missing
    else
        B_dot = []
        for (vector_field_j) in vector_field
            B_dot_j = dot(differentiate(B,x), vector_field_j)
            push!(B_dot, value(B_dot_j))
        end
    end
    println(solution_summary(model))

    return value(B), B_dot
end

function compute_next_level_barriersDB(x::Vector{<:Variable}, u::Vector{<:Variable}, bounds::Vector{Vector{Float64}}, 
                                    g::Polynomial, vectorField, 
                                    U::Vector{Vector{Float64}}, test_pts::Vector{Vector{Float64}}, 
                                    ancestors::Vector{<:Polynomial}) 
    eliminated = []
    second_level_barriers=[]
    for (j,pt) in enumerate(test_pts) 
        if j in eliminated
            # ignore the ones already eliminated
            continue
        end
        # go through each control 
        for u_val in U
            # generate a barrier 
            B = findSuccessiveBarrierDB(x, u_val, g, dynamics, test_pts, ancestors, pt)
            # check if we found something
            if !ismissing(B)
                println("Bingo: Found $B")
                useful = false # check if it is actually useful
                for (k,pt_new) in enumerate(test_pts)
                    if k in eliminated
                        continue
                    end
                   if (B(pt_new...) >= 0.)
                    push!(eliminated, k)   
                    println("\t eliminated $pt_new")
                    useful=true
                   end
                end
                if (useful)
                    println("Num remaining = $(size(test_pts)[1] - size(eliminated)[1])")
                    push!(second_level_barriers, (B, u_val))
                    break
                end
            end
        end
    end
    new_test_pts = [pt for (j, pt) in enumerate(test_pts) if !(j in eliminated)]
    return (second_level_barriers, new_test_pts)
end

function compute_next_level_barriersCT(x::Vector{<:Variable}, u::Vector{<:Variable}, bounds::Vector{Vector{Float64}}, 
                                    g::Polynomial, dynamics, vectorField,
                                    U::Vector{Vector{Float64}}, test_pts::Vector{Vector{Float64}}, 
                                    ancestors::Vector{<:Polynomial}) 
    eliminated = []
    second_level_barriers=[]
    for (j,pt) in enumerate(test_pts) 
        if j in eliminated
            # ignore the ones already eliminated
            continue
        end
        # go through each control 
        for u_val in U
            # generate a barrier 
            B = findSuccessiveBarrierCT(x, u_val, g, dynamics, vectorField, test_pts, ancestors, pt)
            # check if we found something
            if !ismissing(B)
                println("Bingo: Found $B")
                useful = false # check if it is actually useful
                for (k,pt_new) in enumerate(test_pts)
                    if k in eliminated
                        continue
                    end
                   if (B[1](pt_new...) >= 0.)
                    push!(eliminated, k)   
                    println("\t eliminated $pt_new")
                    useful=true
                   end
                end
                if (useful)
                    println("Num remaining = $(size(test_pts)[1] - size(eliminated)[1])")
                    println("Computing transit times")
                    t = refine_barrier_succ(x, u, bounds, u_bounds, g, vectorField, B[1],  B[2], ancestors)
                    push!(second_level_barriers, (B[1], t, u_val))
                    break
                end
            end
        end
    end
    new_test_pts = [pt for (j, pt) in enumerate(test_pts) if !(j in eliminated)]
    return (second_level_barriers, new_test_pts)
end

function compute_next_level_barriersCT_multi(x::Vector{<:Variable}, u::Vector{<:Variable}, bounds::Vector{Vector{Float64}}, 
                                    g::Polynomial, h::Polynomial, dynamics, 
                                    U::Vector{Vector{Float64}}, test_pts::Vector{Vector{Float64}}, 
                                    ancestors::Vector{<:Polynomial}) 
    eliminated = []
    second_level_barriers=[]
    for (j,pt) in enumerate(test_pts) 
        if j in eliminated
            # ignore the ones already eliminated
            continue
        end
        # go through each control 
        for u_val in U
            # generate a barrier 
            B = findSuccessiveBarrierCTmulti(x, u_val, g, h, dynamics, test_pts, ancestors, pt)
            # check if we found something
            if !ismissing(B)
                println("Bingo: Found $B")
                useful = false # check if it is actually useful
                for (k,pt_new) in enumerate(test_pts)
                    if k in eliminated
                        continue
                    end
                   if (B(pt_new...) >= 0.)
                    push!(eliminated, k)   
                    println("\t eliminated $pt_new")
                    useful=true
                   end
                end
                if (useful)
                    println("Num remaining = $(size(test_pts)[1] - size(eliminated)[1])")
                    push!(second_level_barriers, (B, u_val))
                    break
                end
            end
        end
    end
    new_test_pts = [pt for (j, pt) in enumerate(test_pts) if !(j in eliminated)]
    return (second_level_barriers, new_test_pts)
end

function compute_next_level_barriersPM(x::Vector{<:Variable}, u::Vector{<:Variable}, bounds::Vector{Vector{Float64}}, 
                                    g::Polynomial, dynamics, vectorField, 
                                    U::Vector{Vector{Float64}}, test_pts::Vector{Vector{Float64}}, 
                                    ancestors::Vector{<:Polynomial}) 
    eliminated = []
    second_level_barriers=[]
    for (j,pt) in enumerate(test_pts) 
        if j in eliminated
            # ignore the ones already eliminated
            continue
        end
        # go through each control 
        for u_val in U
            # generate a barrier 
            B = findSuccessiveBarrierPM(x, u_val, g, dynamics, vectorField, test_pts, ancestors, pt)
            # check if we found something
            if !ismissing(B)
                println("Bingo: Found $B")
                useful = false # check if it is actually useful
                for (k,pt_new) in enumerate(test_pts)
                    if k in eliminated
                        continue
                    end
                   if (B[1](pt_new...) >= 0.)
                    push!(eliminated, k)   
                    println("\t eliminated $pt_new")
                    useful=true
                   end
                end
                if (useful)
                    println("Num remaining = $(size(test_pts)[1] - size(eliminated)[1])")
                    println("Computing transit times")
                    t = refine_barrier_succ(x, u, bounds, u_bounds, g, vectorField, B[1],  B[2], ancestors)
                    push!(second_level_barriers, (B[1], u_val, t))
                    break
                end
            end
        end
    end
    new_test_pts = [pt for (j, pt) in enumerate(test_pts) if !(j in eliminated)]
    return (second_level_barriers, new_test_pts)
end

# function computeSuccessiveBarriers(B_init, U, T, dynamics; ϵ=0.1, λ=0.1) 
#     β = [] # empty list to store (B_k, u_k)
#     T_satisfied = Set([])
#     for b in B_init
#         for u_k in U
#             for (idx, k) in enumerate(T)
#                 if !(idx  in T_satisfied)
#                     # println("Computing barrier for $(b) and $(k) with control $(u_k)")
#                     B_k = findSuccessiveBarrier(b, k, u_k, dynamics; max_degree=4, λ=λ, ϵ=ϵ)
#                     if !ismissing(B_k)
#                         println("Success: $(B_k)")
#                         push!(β, (B_k, u_k))
#                         push!(T_satisfied, idx)
#                         for (idx_hat,khat) in enumerate(T) # push all the satisfied points back into T
#                             if idx_hat in T_satisfied
#                                 continue
#                             end
#                             if B_k(khat) >= ϵ
#                                 push!(T_satisfied, idx_hat)                                
#                             end
#                         end
#                         #filter!(e -> e ≠ k, T)
#                     else 
#                         println("Failed.")
#                     end
#                 end
#             end
#         end
#     end
#     println("Number of test points satisfied: $(length(T_satisfied)) ")
#     return β
# end
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
        if g(pt[1], pt[2]) >= 0 || h(pt[1], pt[2]) >= 0
            continue
        else
            return pt
        end
    end
end
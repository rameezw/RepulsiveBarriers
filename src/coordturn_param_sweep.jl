using Random
using Printf
using Statistics
include("BarrierSynthesis.jl")

Random.seed!(42)

@polyvar u[1:2]
@polyvar(x[1:6]) # x[1] is x, x[2] is y, x[3] is vel, x[4] is θ, x[5] is ω, x[6] is error

dynamics = [
    (ctrl -> [x[3] * (2.0 / π) * (x[4] + π / 2.0) - 0.2x[6], x[3] * (-2.0 / π) * (x[4] + π) - 0.2x[6], ctrl[1], x[5], ctrl[2], 0.0]),
    (ctrl -> [x[3] * (2.0 / π) * (x[4] + π / 2.0) - 0.2x[6], x[3] * (2.0 / π) * x[4] + 0.2x[6], ctrl[1], x[5], ctrl[2], 0.0]),
    (ctrl -> [x[3] * (-2.0 / π) * (x[4] - π / 2.0) + 0.2x[6], x[3] * (2.0 / π) * x[4] + 0.2x[6], ctrl[1], x[5], ctrl[2], 0.0]),
    (ctrl -> [x[3] * (-2.0 / π) * (x[4] - π / 2.0) + 0.2x[6], x[3] * (-2.0 / π) * (x[4] - π) - 0.2x[6], ctrl[1], x[5], ctrl[2], 0.0]),
]

g = 1^2 - x[1]^2 - x[2]^2
bounds = [[-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0], [-π, π], [-10.0, 10.0], [-1.0, 1.0]]

function findRepulsiveBarrier_HybridCT(x, ctrl::Vector{Float64}, g, dynamics, test_pts; max_degree=4, ϵ=1.0, τ=0.1, γ=10.0, K=1.0, δ=1.0, α=1.0, state_bounds=bounds)
    function prepare_domain_hybrid(var, lb, ub)
        dom = @set(var >= lb) ∩ @set(var <= ub) ∩ @set((var - lb) * (ub - var) >= 0)
        return dom
    end

    @assert τ > 0.0 "τ must be positive."
    @assert δ > 0.0 "δ must be positive."
    @assert K >= 0.0 "K must be nonnegative."
    @assert α > 0.0 "α must be positive."

    λ = (1.0 / τ) * log(1.0 + K / δ)

    solver = optimizer_with_attributes(MosekTools.Optimizer)
    model = SOSModel(solver)
    set_silent(model)
    monos = monomials(x, 0:max_degree)
    N = length(monos)
    @variable(model, -γ <= c[1:N] <= γ)
    B = polynomial(c[1:end], monos)

    dom_list = prepare_domain(x, state_bounds)
    D = reduce((s1, s2) -> s1 ∩ s2, dom_list)

    @constraint(model, B >= ϵ, domain = D ∩ @set(g >= 0))

    for (i, dyn_i) in enumerate(dynamics)
        dyn_with_ctrl = dyn_i(ctrl)
        dBdt = dot(differentiate(B, x), dyn_with_ctrl)
        dom_i = prepare_domain_hybrid(x[4], (i - 3) * π / 2, (i - 2) * π / 2)
        @constraint(model, dBdt <= λ * B, domain = D ∩ dom_i)
    end

    @polyvar q1 q2
    Qα = @set(α^2 - q1^2 - q2^2 >= 0)
    B_shift = subs(B, x[1] => x[1] - q1, x[2] => x[2] - q2)
    @constraint(model, K >= B_shift - B, domain = D ∩ Qα)

    set_objective_sense(model, MOI.MIN_SENSE)
    objective_fn = sum((B(pt...) for pt in test_pts); init = 0.0)
    @objective(model, Min, objective_fn)

    JuMP.optimize!(model)
    stat = JuMP.primal_status(model)
    if stat != FEASIBLE_POINT
        return missing
    end

    return value(B), K
end

build_U(a, b) = [[-a, -b], [-a, b], [a, -b], [a, b]]

function synthesize_bank(Uset, ϵ, K, δ; max_degree=4, n_tests=80, α=1.0)
    pts = [get_random(bounds, g) for _ in 1:n_tests]
    pts_work = copy(pts)
    barriers = Polynomial[]

    for ctrl in Uset
        res = findRepulsiveBarrier_HybridCT(x, ctrl, g, dynamics, pts_work; max_degree=max_degree, ϵ=ϵ, K=K, δ=δ, α=α)
        if res === missing
            return nothing
        end
        B, _ = res
        push!(barriers, B)
        filter!(pt -> B(pt...) > 0.0, pts_work)
    end

    train_cov = 1.0 - length(pts_work) / max(length(pts), 1)
    return (barriers=barriers, train_cov=train_cov)
end

function estimate_xy_metrics(all_barriers; limits=(-10.0, 10.0), step=0.8, x3_val=5.0, x4_vals=[-2.0, 2.0], x5_val=0.0, x6_val=0.0)
    xs = collect(limits[1]:step:limits[2])
    ys = collect(limits[1]:step:limits[2])
    n_total = length(xs) * length(ys)
    cov_vals = Float64[]
    minb_means = Float64[]
    for x4_val in x4_vals
        n_safe = 0
        minb_sum = 0.0
        for xv in xs, yv in ys
            min_b = minimum(B(xv, yv, x3_val, x4_val, x5_val, x6_val) for B in all_barriers)
            minb_sum += min_b
            if min_b <= 0.0
                n_safe += 1
            end
        end
        push!(cov_vals, n_safe / max(n_total, 1))
        push!(minb_means, minb_sum / max(n_total, 1))
    end
    return (coverage=mean(cov_vals), mean_minB=mean(minb_means))
end

coarse_cfg = []
for (a, b) in [(3.0, 3.0), (4.0, 4.0), (5.0, 5.0), (5.0, 3.0)]
    for ϵ in (0.02, 0.05, 0.10)
        for K in (0.5, 1.0, 1.5)
            for δ in (0.5, 0.8)
                push!(coarse_cfg, (a=a, b=b, ϵ=ϵ, K=K, δ=δ))
            end
        end
    end
end

println("Running coarse sweep on ", length(coarse_cfg), " configs...")
coarse_results = NamedTuple[]
for cfg in coarse_cfg
    Uset = build_U(cfg.a, cfg.b)
    out = synthesize_bank(Uset, cfg.ϵ, cfg.K, cfg.δ; max_degree=2, n_tests=40)
    if out === nothing
        push!(coarse_results, (cfg..., feasible=false, eval_cov=0.0, train_cov=0.0))
        continue
    end
    m = estimate_xy_metrics(out.barriers; step=0.8, x4_vals=[-2.0, 2.0])
    push!(coarse_results, (cfg..., feasible=true, eval_cov=m.coverage, mean_minB=m.mean_minB, train_cov=out.train_cov))
end

feas_coarse = filter(r -> r.feasible, coarse_results)
if isempty(feas_coarse)
    error("No feasible configurations found in coarse sweep.")
end

sort!(feas_coarse, by = r -> (r.eval_cov, -r.mean_minB, r.train_cov), rev=true)
println("\nTop coarse configs:")
for (i, r) in enumerate(feas_coarse[1:min(5, end)])
    @printf("%d) U=(%.1f,%.1f), eps=%.2f, K=%.2f, delta=%.2f | eval=%.3f meanMinB=%.3f train=%.3f\n", i, r.a, r.b, r.ϵ, r.K, r.δ, r.eval_cov, r.mean_minB, r.train_cov)
end

println("\nRefining top coarse configs with degree-3 synthesis...")
refine_pool = feas_coarse[1:min(15, end)]
refined3 = NamedTuple[]
for r in refine_pool
    Uset = build_U(r.a, r.b)
    out = synthesize_bank(Uset, r.ϵ, r.K, r.δ; max_degree=3, n_tests=60)
    if out === nothing
        push!(refined3, (r..., refine_feasible=false, refine_eval=0.0, refine_mean_minB=Inf, refine_train=0.0))
        continue
    end
    m = estimate_xy_metrics(out.barriers; step=0.5, x4_vals=[-2.0, 2.0])
    push!(refined3, (r..., refine_feasible=true, refine_eval=m.coverage, refine_mean_minB=m.mean_minB, refine_train=out.train_cov))
end

feas_refined3 = filter(r -> r.refine_feasible, refined3)
if isempty(feas_refined3)
    error("No feasible configurations found in degree-3 refinement stage.")
end

sort!(feas_refined3, by = r -> (r.refine_eval, -r.refine_mean_minB, r.refine_train), rev=true)

println("\nTop degree-3 refined configs:")
for (i, r) in enumerate(feas_refined3[1:min(5, end)])
    @printf("%d) U=(%.1f,%.1f), eps=%.2f, K=%.2f, delta=%.2f | eval=%.3f meanMinB=%.3f train=%.3f\n", i, r.a, r.b, r.ϵ, r.K, r.δ, r.refine_eval, r.refine_mean_minB, r.refine_train)
end

println("\nTrying degree-4 on top degree-3 candidates...")
refined4 = NamedTuple[]
for r in feas_refined3[1:min(5, end)]
    Uset = build_U(r.a, r.b)
    out = synthesize_bank(Uset, r.ϵ, r.K, r.δ; max_degree=4, n_tests=80)
    if out === nothing
        continue
    end
    m = estimate_xy_metrics(out.barriers; step=0.5, x4_vals=[-2.0, 2.0])
    push!(refined4, (r..., d4_eval=m.coverage, d4_mean_minB=m.mean_minB, d4_train=out.train_cov))
end

use_degree4 = !isempty(refined4)
if use_degree4
    sort!(refined4, by = r -> (r.d4_eval, -r.d4_mean_minB, r.d4_train), rev=true)
    best = first(refined4)
else
    best = first(feas_refined3)
end

println("\nBEST CONFIG:")
@printf("U = [[-%.1f,-%.1f],[-%.1f,%.1f],[%.1f,-%.1f],[%.1f,%.1f]]\n", best.a, best.b, best.a, best.b, best.a, best.b, best.a, best.b)
@printf("eps = %.2f, K = %.2f, delta = %.2f\n", best.ϵ, best.K, best.δ)
if use_degree4
    @printf("coverage eval (degree-4) = %.3f, meanMinB = %.3f, train = %.3f\n", best.d4_eval, best.d4_mean_minB, best.d4_train)
else
    @printf("coverage eval (degree-3) = %.3f, meanMinB = %.3f, train = %.3f\n", best.refine_eval, best.refine_mean_minB, best.refine_train)
end

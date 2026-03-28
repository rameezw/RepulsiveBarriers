using LinearAlgebra
using Plots
using Statistics

wrap_to_pi(ang) = mod(ang + π, 2π) - π

function obstacle_state(t)
    c = [2.5 * cos(0.18 * t), 1.8 * sin(0.12 * t)]
    ċ = [-2.5 * 0.18 * sin(0.18 * t), 1.8 * 0.12 * cos(0.12 * t)]
    return c, ċ
end

function reference_state(t)
    x0, y0 = -8.0, -8.0
    vrefx, vrefy = 0.40, 0.40
    r = [x0 + vrefx * t, y0 + vrefy * t]
    ṙ = [vrefx, vrefy]
    return r, ṙ
end

barrier_value(B, xrel, yrel, θ) = B(xrel, yrel, θ, 0.0)

function active_barrier(all_barriers, xrel, yrel, θ)
    vals = [barrier_value(B, xrel, yrel, θ) for B in all_barriers]
    idx = argmax(vals)
    return idx, vals[idx], vals
end

# function finite_diff_grad(B, xrel, yrel, θ; step_size=1e-4)
#     dBx = (barrier_value(B, xrel + step_size, yrel, θ) - barrier_value(B, xrel - step_size, yrel, θ)) / (2step_size)
#     dBy = (barrier_value(B, xrel, yrel + step_size, θ) - barrier_value(B, xrel, yrel - step_size, θ)) / (2step_size)
#     dBθ = (barrier_value(B, xrel, yrel, θ + step_size) - barrier_value(B, xrel, yrel, θ - step_size)) / (2step_size)
#     return dBx, dBy, dBθ
# end

function nominal_tracking_control(x, t)
    r_la, _ = reference_state(t + 1.0)
    θ_goal = atan(r_la[2] - x[2], r_la[1] - x[1])
    e_θ = wrap_to_pi(θ_goal - x[3])
    u_nom = 1.15 * e_θ
    return clamp(u_nom, -1, 1)
end

# function cbf_safety_filter(x, t, u_nom, center_hat, all_barriers, v;
#     trigger=0.25, α=2.5, umin=-1.5, umax=1.5, R_safe=1.2, rep_gain=3.2)

#     xrel = x[1] - center_hat[1]
#     yrel = x[2] - center_hat[2]
#     θ = x[3]

#     idx, B_val, _ = active_barrier(all_barriers, xrel, yrel, θ)
#     B_act = all_barriers[idx]

#     dist_hat = hypot(xrel, yrel)
#     safety_active = (B_val <= trigger) || (dist_hat <= R_safe + 1.5)
#     if !safety_active
#         return clamp(u_nom, umin, umax), false, B_val, idx
#     end

#     _, obs_vel = obstacle_state(t)
#     dBx, dBy, dBθ = finite_diff_grad(B_act, xrel, yrel, θ)

#     ẋrel_no_u = v * cos(θ) - obs_vel[1]
#     ẏrel_no_u = v * sin(θ) - obs_vel[2]

#     a = dBθ
#     b = dBx * ẋrel_no_u + dBy * ẏrel_no_u + α * B_val

#     # Turn away from estimated obstacle center as a robust local repulsive action.
#     θ_away = atan(yrel, xrel)
#     u_rep = clamp(rep_gain * wrap_to_pi(θ_away - θ), umin, umax)
#     u_f = clamp(0.35 * u_nom + 0.65 * u_rep, umin, umax)
#     if abs(a) > 1e-8
#         bound = -b / a
#         if a > 0
#             u_f = max(u_f, bound)
#         else
#             u_f = min(u_f, bound)
#         end
#         u_f = clamp(u_f, umin, umax)
#     else
#         if b < 0
#             u_f = sign(-b) * umax
#         end
#     end

#     return u_f, true, B_val, idx
# end

function sample_hold_barrier_policy(x, center_hat, all_barriers, barrier_controls, δ, K)
    xrel = x[1] - center_hat[1]
    yrel = x[2] - center_hat[2]
    θ = x[3]
    vals = [barrier_value(B, xrel, yrel, θ) for B in all_barriers]

    # Union-safe set semantics: safety is certified when at least one barrier is <= 0.
    idx_min = argmin(vals)
    B_crit = vals[idx_min]

    # Trigger override in the repulsion band near the safety boundary.
    if B_crit > -δ-K
        return true, barrier_controls[idx_min], B_crit, idx_min, vals
    end

    return false, 0.0, B_crit, idx_min, vals
end

function run_repulsive_hybrid_dubins_demo(all_barriers; v=5, τ_steps=1, dt=0.05, T=40.0, k_override=1.0, δ=1.0, barrier_controls=nothing, umin=nothing, umax=nothing, x0=[-10.0, -10.0, π/2])
    N = Int(round(T / dt))
    x = copy(x0)
    K = k_override

    if isnothing(barrier_controls)
        nB = length(all_barriers)
        if nB == 2
            barrier_controls = [-1.5, 1.5]
        else
            barrier_controls = fill(0.0, nB)
        end
    end
    @assert length(barrier_controls) == length(all_barriers) "barrier_controls must match number of barriers"

    # By default, use the same control bounds used to synthesize/assign barrier controls.
    if isnothing(umin)
        umin = minimum(barrier_controls)
    end
    if isnothing(umax)
        umax = maximum(barrier_controls)
    end

    X = zeros(N + 1, 3)
    X[1, :] .= x

    u_hist = zeros(N)
    u_nom_hist = zeros(N)
    B_hist = zeros(N)
    active_idx_hist = zeros(Int, N)
    barrier_override_hist = falses(N)
    dist_true_hist = zeros(N)
    track_err_hist = zeros(N)

    obs_true_hist = zeros(N + 1, 2)
    obs_hat_hist = zeros(N + 1, 2)
    ref_hist = zeros(N + 1, 2)

    obs_true_hist[1, :] .= obstacle_state(0.0)[1]
    obs_hat = copy(obs_true_hist[1, :])
    obs_hat_hist[1, :] .= obs_hat
    ref_hist[1, :] .= reference_state(0.0)[1]

    hold_override_on = false
    hold_u_bar = 0.0
    hold_B_val = 0.0
    hold_idx = 1

    for k in 1:N
        t = (k - 1) * dt

        if (k - 1) % τ_steps == 0
            obs_hat = copy(obstacle_state(t)[1])
        end

        v_cmd = v

        u_nom = nominal_tracking_control(x, t)
        if (k - 1) % τ_steps == 0
            hold_override_on, hold_u_bar, hold_B_val, hold_idx, _ =
                sample_hold_barrier_policy(x, obs_hat, all_barriers, barrier_controls, δ, K)
        end

        override_on = hold_override_on
        B_val = hold_B_val
        idx = hold_idx
        u = override_on ? hold_u_bar : u_nom
        u = clamp(u, umin, umax)

        ẋ = [v_cmd * cos(x[3]), v_cmd * sin(x[3]), u]
        x = x .+ dt .* ẋ
        x[3] = wrap_to_pi(x[3])

        X[k + 1, :] .= x
        u_hist[k] = u
        u_nom_hist[k] = u_nom
        B_hist[k] = B_val
        active_idx_hist[k] = idx
        barrier_override_hist[k] = override_on
        obs_now = obstacle_state(t)[1]
        dist_true_hist[k] = hypot(x[1] - obs_now[1], x[2] - obs_now[2])
        r_now = reference_state(t)[1]
        track_err_hist[k] = hypot(x[1] - r_now[1], x[2] - r_now[2])

        obs_true_hist[k + 1, :] .= obstacle_state(t + dt)[1]
        obs_hat_hist[k + 1, :] .= obs_hat
        ref_hist[k + 1, :] .= reference_state(t + dt)[1]
    end

    # Build time grids from N to keep lengths consistent even when T/dt is non-integer.
    ts = collect(0.0:dt:(N * dt))
    ts_hist = ts[1:end-1]

    p_traj = plot(X[:, 1], X[:, 2], lw=2, label="system", aspect_ratio=1)
    plot!(p_traj, ref_hist[:, 1], ref_hist[:, 2], lw=2, ls=:dash, label="reference")
    plot!(p_traj, obs_true_hist[:, 1], obs_true_hist[:, 2], lw=2, ls=:dot, label="obstacle center")
    plot!(p_traj, obs_hat_hist[:, 1], obs_hat_hist[:, 2], lw=2, ls=:dashdot, label="recentered center")
    plot!(p_traj, title="Trajectory with moving obstacle and τ-step recentering", xlabel="x", ylabel="y")

    p_B = plot(ts_hist, B_hist, lw=2, label="B (active recentered barrier)", xlabel="time", ylabel="B")
    plot!(p_B, ts_hist, 0 .* B_hist, ls=:dash, lw=1.5, label="violation boundary")
    plot!(p_B, ts_hist, (-δ) .* ones(length(B_hist)), ls=:dot, lw=1.5, label="-δ")
    plot!(p_B, ts_hist, (-(δ + K)) .* ones(length(B_hist)), ls=:dot, lw=1.5, label="-δ-K")
    plot!(p_B, title="Barrier margin")

    p_u = plot(ts_hist, u_nom_hist, lw=2, ls=:dash, label="u_nom", xlabel="time", ylabel="turn-rate")
    plot!(p_u, ts_hist, u_hist, lw=2, label="u_applied")
    plot!(p_u, title="Nominal vs safety-filtered control")

    mkpath("figures")
    anim = @animate for k in 1:4:length(ts)
        p = plot(xlims=(-10, 10), ylims=(-10, 10), aspect_ratio=1,
            xlabel="x", ylabel="y", title="Repulsive barrier simulation (moving obstacle)")

        plot!(p, ref_hist[:, 1], ref_hist[:, 2], lw=2, ls=:dash, color=:gray, label="reference")
        plot!(p, X[1:k, 1], X[1:k, 2], lw=2.5, color=:blue, label="system")

        if k > 1 && barrier_override_hist[min(k - 1, end)]
            scatter!(p, [X[k, 1]], [X[k, 2]], color=:red, markersize=5, label="barrier override")
        else
            scatter!(p, [X[k, 1]], [X[k, 2]], color=:blue, markersize=5, label="state")
        end

        cx, cy = obs_true_hist[k, 1], obs_true_hist[k, 2]
        hx, hy = obs_hat_hist[k, 1], obs_hat_hist[k, 2]

        θc = LinRange(0, 2π, 160)
        obs_x = cx .+ 1.0 .* cos.(θc)
        obs_y = cy .+ 1.0 .* sin.(θc)
        plot!(p, obs_x, obs_y, lw=2, color=:black, label="obstacle")

        scatter!(p, [cx], [cy], color=:black, markersize=4, label="obs center")
        scatter!(p, [hx], [hy], marker=:x, color=:magenta, markersize=5, label="recentered")

        annotate!(p, -11.5, 9.1, text("t = $(round(ts[k], digits=2)) s", 9))
        if k > 1
            annotate!(p, -9.5, 8.3, text("B = $(round(B_hist[min(k - 1, end)], digits=3))", 9))
            mode_str = barrier_override_hist[min(k - 1, end)] ? "override" : "nominal"
            annotate!(p, -9.5, 7.5, text("mode = $mode_str", 9))
        end

        p
    end

    gif_path = "figures/repulsive_hybrid_dubins_moving_obstacle.gif"
    gif(anim, gif_path, fps=20)

    println("Simulation finished")
    println("minimum recentered barrier value = ", minimum(B_hist))
    println("minimum true obstacle distance = ", minimum(dist_true_hist))
    println("mean tracking error = ", mean(track_err_hist))
    println("max tracking error = ", maximum(track_err_hist))
    println("number of barrier overrides = ", count(barrier_override_hist))
    println("Animation saved to ", gif_path)

    p_anim = plot(X[:, 1], X[:, 2], lw=2.5, label="trajectory preview", aspect_ratio=1)
    plot!(p_anim, ref_hist[:, 1], ref_hist[:, 2], lw=2, ls=:dash, label="reference")
    plot!(p_anim, title="Preview (full animation is in GIF)", xlabel="x", ylabel="y")

    return (
        ts=ts,
        X=X,
        B_hist=B_hist,
        u_hist=u_hist,
        u_nom_hist=u_nom_hist,
        obs_true_hist=obs_true_hist,
        obs_hat_hist=obs_hat_hist,
        ref_hist=ref_hist,
        barrier_override_hist=barrier_override_hist,
        p_traj=p_traj,
        p_B=p_B,
        p_u=p_u,
        anim=p_anim,
        gif_path=gif_path,
        min_dist=minimum(dist_true_hist),
        mean_track_err=mean(track_err_hist),
        max_track_err=maximum(track_err_hist),
    )
end

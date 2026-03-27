using LinearAlgebra
using Plots
using Statistics

wrap_to_pi(ang) = mod(ang + pi, 2pi) - pi

function obstacle_state(t)
    c = [2.5 * cos(0.18 * t), 1.8 * sin(0.12 * t)]
    c_dot = [-2.5 * 0.18 * sin(0.18 * t), 1.8 * 0.12 * cos(0.12 * t)]
    return c, c_dot
end

function reference_state(t; p_start=[-15.0, -15.0], p_end=[15.0, 15.0], T_ref=20.0)
    alpha = clamp(t / T_ref, 0.0, 1.0)
    r = (1.0 - alpha) .* p_start .+ alpha .* p_end
    r_dot = (p_end .- p_start) ./ T_ref
    return r, r_dot
end

function barrier_value(B, state, center_hat)
    xrel = state[1] - center_hat[1]
    yrel = state[2] - center_hat[2]
    return B(xrel, yrel, state[3], state[4], state[5], state[6])
end

function critical_barrier(all_barriers, state, center_hat)
    vals = [barrier_value(B, state, center_hat) for B in all_barriers]
    idx = argmin(vals)
    return idx, vals[idx], vals
end

function sample_hold_barrier_policy(state, center_hat, all_barriers, barrier_controls, delta, K)
    idx, B_crit, vals = critical_barrier(all_barriers, state, center_hat)
    if B_crit > -(delta + K)
        return true, barrier_controls[idx], B_crit, idx, vals
    end
    return false, (0.0, 0.0), B_crit, idx, vals
end

function coordturn_mode_rhs(state, ctrl)
    dx = zeros(6)

    if (-pi <= state[4] < -pi / 2)
        dx[1] = state[3] * (2.0 / pi) * (state[4] + pi / 2.0) - 0.2 * state[6]
        dx[2] = state[3] * (-2.0 / pi) * (state[4] + pi) - 0.2 * state[6]
    elseif (-pi / 2 <= state[4] < 0.0)
        dx[1] = state[3] * (2.0 / pi) * (state[4] + pi / 2.0) - 0.2 * state[6]
        dx[2] = state[3] * (2.0 / pi) * state[4] + 0.2 * state[6]
    elseif (0.0 <= state[4] < pi / 2)
        dx[1] = state[3] * (-2.0 / pi) * (state[4] - pi / 2.0) + 0.2 * state[6]
        dx[2] = state[3] * (2.0 / pi) * state[4] + 0.2 * state[6]
    else
        dx[1] = state[3] * (-2.0 / pi) * (state[4] - pi / 2.0) + 0.2 * state[6]
        dx[2] = state[3] * (-2.0 / pi) * (state[4] - pi) - 0.2 * state[6]
    end

    dx[3] = ctrl[1]
    dx[4] = state[5]
    dx[5] = ctrl[2]
    dx[6] = 0.0
    return dx
end

function nominal_tracking_control(state, t; u1min=-5.0, u1max=5.0, u2min=-5.0, u2max=5.0)
    r, r_dot = reference_state(t + 1.0)
    v_des = clamp(norm(r_dot), 0.0, 8.0)

    theta_goal = atan(r[2] - state[2], r[1] - state[1])
    theta_err = wrap_to_pi(theta_goal - state[4])

    omega_des = 1.2 * theta_err
    u1_nom = clamp(0.9 * (v_des - state[3]), u1min, u1max)
    u2_nom = clamp(1.5 * (omega_des - state[5]), u2min, u2max)
    return (u1_nom, u2_nom)
end

function run_repulsive_hybrid_coordturn_demo(all_barriers;
    barrier_controls=nothing,
    delta=1.0,
    k_override=1.0,
    dt=0.05,
    T=45.0,
    tau_steps=10,
    x0=[-15.0, -15.0, 0.0, pi / 4, 0.0, 0.0],
    ref_start=[-15.0, -15.0],
    ref_end=[15.0, 15.0],
    ref_T=20.0,
    u1min=nothing,
    u1max=nothing,
    u2min=nothing,
    u2max=nothing)

    N = Int(round(T / dt))
    state = copy(x0)
    K = k_override

    if isnothing(barrier_controls)
        barrier_controls = [(-5.0, -5.0), (-5.0, 5.0), (5.0, -5.0), (5.0, 5.0)]
    end

    @assert length(barrier_controls) == length(all_barriers) "barrier_controls must match number of barriers"

    if isnothing(u1min)
        u1min = minimum(first(c) for c in barrier_controls)
    end
    if isnothing(u1max)
        u1max = maximum(first(c) for c in barrier_controls)
    end
    if isnothing(u2min)
        u2min = minimum(last(c) for c in barrier_controls)
    end
    if isnothing(u2max)
        u2max = maximum(last(c) for c in barrier_controls)
    end

    X = zeros(N + 1, 6)
    X[1, :] .= state

    B_hist = zeros(N)
    B_idx_hist = zeros(Int, N)
    override_hist = falses(N)
    dist_true_hist = zeros(N)
    track_err_hist = zeros(N)

    u1_nom_hist = zeros(N)
    u2_nom_hist = zeros(N)
    u1_hist = zeros(N)
    u2_hist = zeros(N)

    obs_true_hist = zeros(N + 1, 2)
    obs_hat_hist = zeros(N + 1, 2)
    ref_hist = zeros(N + 1, 2)

    obs_true_hist[1, :] .= obstacle_state(0.0)[1]
    obs_hat = copy(obs_true_hist[1, :])
    obs_hat_hist[1, :] .= obs_hat
    ref_hist[1, :] .= reference_state(0.0; p_start=ref_start, p_end=ref_end, T_ref=ref_T)[1]

    hold_override = false
    hold_u = (0.0, 0.0)

    for k in 1:N
        t = (k - 1) * dt

        if (k - 1) % tau_steps == 0
            obs_hat = copy(obstacle_state(t)[1])
            hold_override, hold_u, _, _, _ =
                sample_hold_barrier_policy(state, obs_hat, all_barriers, barrier_controls, delta, K)
        end

        u_nom = nominal_tracking_control(state, t; u1min=u1min, u1max=u1max, u2min=u2min, u2max=u2max)
        u = hold_override ? hold_u : u_nom
        u = (
            clamp(u[1], u1min, u1max),
            clamp(u[2], u2min, u2max),
        )

        dstate = coordturn_mode_rhs(state, u)
        state = state .+ dt .* dstate
        state[4] = wrap_to_pi(state[4])

        _, B_crit, _ = critical_barrier(all_barriers, state, obs_hat)
        idx, _, _ = critical_barrier(all_barriers, state, obs_hat)

        X[k + 1, :] .= state
        B_hist[k] = B_crit
        B_idx_hist[k] = idx
        override_hist[k] = hold_override

        u1_nom_hist[k] = u_nom[1]
        u2_nom_hist[k] = u_nom[2]
        u1_hist[k] = u[1]
        u2_hist[k] = u[2]

        obs_true_hist[k + 1, :] .= obstacle_state(t + dt)[1]
        obs_hat_hist[k + 1, :] .= obs_hat
        ref_hist[k + 1, :] .= reference_state(t + dt; p_start=ref_start, p_end=ref_end, T_ref=ref_T)[1]

        obs_now = obstacle_state(t)[1]
        dist_true_hist[k] = hypot(state[1] - obs_now[1], state[2] - obs_now[2])
        r_now = reference_state(t; p_start=ref_start, p_end=ref_end, T_ref=ref_T)[1]
        track_err_hist[k] = hypot(state[1] - r_now[1], state[2] - r_now[2])
    end

    ts = collect(0.0:dt:(N * dt))
    ts_hist = ts[1:end-1]


    p_traj = plot(X[:, 1], X[:, 2], lw=2, label="system", aspect_ratio=1, xlims=(-16, 16), ylims=(-16, 16))
    plot!(p_traj, ref_hist[:, 1], ref_hist[:, 2], lw=2, ls=:dash, label="reference")
    plot!(p_traj, obs_true_hist[:, 1], obs_true_hist[:, 2], lw=2, ls=:dot, label="obstacle center")
    plot!(p_traj, obs_hat_hist[:, 1], obs_hat_hist[:, 2], lw=2, ls=:dashdot, label="recentered center")
    plot!(p_traj, title="CoordTurn with moving obstacle and tau-step recentering", xlabel="x", ylabel="y")

    p_B = plot(ts_hist, B_hist, lw=2, label="B (active recentered barrier)", xlabel="time", ylabel="B")
    plot!(p_B, ts_hist, 0 .* B_hist, ls=:dash, lw=1.5, label="violation boundary")
    plot!(p_B, ts_hist, (-delta) .* ones(length(B_hist)), ls=:dot, lw=1.5, label="-delta")
    plot!(p_B, ts_hist, (-(delta + K)) .* ones(length(B_hist)), ls=:dot, lw=1.5, label="-delta-K")
    plot!(p_B, title="Barrier margin")

    p_u = plot(ts_hist, u1_nom_hist, lw=2, ls=:dash, label="u1_nom", xlabel="time", ylabel="control")
    plot!(p_u, ts_hist, u1_hist, lw=2, label="u1_applied")
    plot!(p_u, ts_hist, u2_nom_hist, lw=2, ls=:dashdot, label="u2_nom")
    plot!(p_u, ts_hist, u2_hist, lw=2, label="u2_applied")
    plot!(p_u, title="Nominal vs safety-filtered control")

    mkpath("figures")
    anim_obj = @animate for k in 1:4:length(ts)
        p = plot(xlims=(-16, 16), ylims=(-16, 16), aspect_ratio=1,
            xlabel="x", ylabel="y", title="CoordTurn repulsive barrier simulation")

        plot!(p, ref_hist[:, 1], ref_hist[:, 2], lw=2, ls=:dash, color=:gray, label="reference")
        plot!(p, X[1:k, 1], X[1:k, 2], lw=2.5, color=:blue, label="system")

        if k > 1 && override_hist[min(k - 1, end)]
            scatter!(p, [X[k, 1]], [X[k, 2]], color=:red, markersize=5, label="barrier override")
        else
            scatter!(p, [X[k, 1]], [X[k, 2]], color=:blue, markersize=5, label="state")
        end

        cx, cy = obs_true_hist[k, 1], obs_true_hist[k, 2]
        hx, hy = obs_hat_hist[k, 1], obs_hat_hist[k, 2]
        theta = LinRange(0, 2pi, 160)

        obs_x = cx .+ 1.0 .* cos.(theta)
        obs_y = cy .+ 1.0 .* sin.(theta)
        plot!(p, obs_x, obs_y, lw=2, color=:black, label="obstacle")

        scatter!(p, [cx], [cy], color=:black, markersize=4, label="obs center")
        scatter!(p, [hx], [hy], marker=:x, color=:magenta, markersize=5, label="recentered")
        annotate!(p, -9.5, 9.1, text("t = $(round(ts[k], digits=2)) s", 9))
        if k > 1
            annotate!(p, -9.0, 8.3, text("B = $(round(B_hist[min(k - 1, end)], digits=3))", 9))
            mode_str = override_hist[min(k - 1, end)] ? "override" : "nominal"
            annotate!(p, -9.0, 7.5, text("mode = $mode_str", 9))
        end
        p
    end

    gif_path = "figures/repulsive_hybrid_coordturn_moving_obstacle.gif"
    gif(anim_obj, gif_path, fps=20)

    println("Simulation finished")
    println("minimum recentered barrier value = ", minimum(B_hist))
    println("minimum true obstacle distance = ", minimum(dist_true_hist))
    println("mean tracking error = ", mean(track_err_hist))
    println("max tracking error = ", maximum(track_err_hist))
    println("number of barrier overrides = ", count(override_hist))
    println("Animation saved to ", gif_path)

    p_anim = plot(X[:, 1], X[:, 2], lw=2.5, label="trajectory preview", aspect_ratio=1)
    plot!(p_anim, ref_hist[:, 1], ref_hist[:, 2], lw=2, ls=:dash, label="reference")
    plot!(p_anim, title="Preview (full animation is in GIF)", xlabel="x", ylabel="y")

    return (
        ts=ts,
        X=X,
        B_hist=B_hist,
        B_idx_hist=B_idx_hist,
        override_hist=override_hist,
        u1_nom_hist=u1_nom_hist,
        u2_nom_hist=u2_nom_hist,
        u1_hist=u1_hist,
        u2_hist=u2_hist,
        obs_true_hist=obs_true_hist,
        obs_hat_hist=obs_hat_hist,
        ref_hist=ref_hist,
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

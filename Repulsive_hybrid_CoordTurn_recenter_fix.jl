using LinearAlgebra
using Plots
using Statistics

wrap_to_pi(ang) = mod(ang + pi, 2pi) - pi

function obstacle_state(t;
    p_start=[-15.0, -15.0],
    p_end=[15.0, 15.0],
    T_path=20.0,
    amplitude=9.0,
    lateral_bias=8.0,
    cycles=2.0,
    phase=pi / 2)

    delta = p_end .- p_start
    seg_len = max(norm(delta), 1e-6)
    tangent = delta ./ seg_len
    normal = [-tangent[2], tangent[1]]

    alpha = clamp(t / T_path, 0.0, 1.0)
    dalpha_dt = (0.0 < t < T_path) ? (1.0 / T_path) : 0.0

    base = (1.0 - alpha) .* p_start .+ alpha .* p_end
    base_dot = dalpha_dt .* delta

    angle = 2pi * cycles * alpha + phase
    lateral = lateral_bias + amplitude * sin(angle)
    lateral_dot = amplitude * cos(angle) * (2pi * cycles * dalpha_dt)

    c = base .+ lateral .* normal
    c_dot = base_dot .+ lateral_dot .* normal
    return c, c_dot
end

function reference_state(t; p_start=[-15.0, -15.0], p_end=[15.0, 15.0], T_ref=20.0)
    alpha = clamp(t / T_ref, 0.0, 1.0)
    s = alpha^2 * (3.0 - 2.0 * alpha)
    ds_dt = (0.0 < t < T_ref) ? (6.0 * alpha * (1.0 - alpha) / T_ref) : 0.0
    r = (1.0 - s) .* p_start .+ s .* p_end
    r_dot = ds_dt .* (p_end .- p_start)
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
    obs_amplitude=9.0,
    obs_lateral_bias=8.0,
    obs_cycles=2.0,
    obs_phase=pi / 2,
    plot_half_span=16.0,
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

    obs_true_hist[1, :] .= obstacle_state(0.0;
        p_start=ref_start,
        p_end=ref_end,
        T_path=ref_T,
        amplitude=obs_amplitude,
        lateral_bias=obs_lateral_bias,
        cycles=obs_cycles,
        phase=obs_phase)[1]
    obs_hat = copy(obs_true_hist[1, :])
    obs_hat_hist[1, :] .= obs_hat
    ref_hist[1, :] .= reference_state(0.0; p_start=ref_start, p_end=ref_end, T_ref=ref_T)[1]

    hold_override = false
    hold_u = (0.0, 0.0)

    for k in 1:N
        t = (k - 1) * dt

        if (k - 1) % tau_steps == 0
            obs_hat = copy(obstacle_state(t;
                p_start=ref_start,
                p_end=ref_end,
                T_path=ref_T,
                amplitude=obs_amplitude,
                lateral_bias=obs_lateral_bias,
                cycles=obs_cycles,
                phase=obs_phase)[1])
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

        obs_true_hist[k + 1, :] .= obstacle_state(t + dt;
            p_start=ref_start,
            p_end=ref_end,
            T_path=ref_T,
            amplitude=obs_amplitude,
            lateral_bias=obs_lateral_bias,
            cycles=obs_cycles,
            phase=obs_phase)[1]
        obs_hat_hist[k + 1, :] .= obs_hat
        ref_hist[k + 1, :] .= reference_state(t + dt; p_start=ref_start, p_end=ref_end, T_ref=ref_T)[1]

        obs_now = obstacle_state(t;
            p_start=ref_start,
            p_end=ref_end,
            T_path=ref_T,
            amplitude=obs_amplitude,
            lateral_bias=obs_lateral_bias,
            cycles=obs_cycles,
            phase=obs_phase)[1]
        dist_true_hist[k] = hypot(state[1] - obs_now[1], state[2] - obs_now[2])
        r_now = reference_state(t; p_start=ref_start, p_end=ref_end, T_ref=ref_T)[1]
        track_err_hist[k] = hypot(state[1] - r_now[1], state[2] - r_now[2])
    end

    ts = collect(0.0:dt:(N * dt))
    ts_hist = ts[1:end-1]

    function _override_spans(flags, ts_hist_local, dt_local)
        spans = Tuple{Float64, Float64}[]
        in_span = false
        t0 = 0.0
        for i in eachindex(flags)
            if flags[i] && !in_span
                in_span = true
                t0 = ts_hist_local[i]
            elseif !flags[i] && in_span
                push!(spans, (t0, ts_hist_local[i]))
                in_span = false
            end
        end
        if in_span
            push!(spans, (t0, ts_hist_local[end] + dt_local))
        end
        return spans
    end

    override_spans = _override_spans(override_hist, ts_hist, dt)
    sample_times = ts[1:tau_steps:end]
    obs_snap_idx = unique(round.(Int, range(1, length(ts), length=min(5, length(ts)))))
    theta_circle = LinRange(0, 2pi, 200)
    path_center = 0.5 .* (ref_start .+ ref_end)

    default(
        legendfontsize=8,
        guidefontsize=10,
        tickfontsize=8,
        titlefontsize=11,
        linewidth=2,
        framestyle=:box,
        gridalpha=0.18,
        foreground_color_legend=nothing,
        background_color_legend=RGBA(1, 1, 1, 0.85),
        dpi=220,
    )

    p_traj = plot(
        aspect_ratio=1,
        xlabel="x", ylabel="y",
        xlims=(path_center[1] - plot_half_span, path_center[1] + plot_half_span),
        ylims=(path_center[2] - plot_half_span, path_center[2] + plot_half_span),
        title="Workspace trajectory",
        legend=:bottomright,
    )
    plot!(p_traj, ref_hist[:, 1], ref_hist[:, 2], lw=2, ls=:dash, color=:gray45, label="reference")
    plot!(p_traj, X[:, 1], X[:, 2], lw=2.6, color=:dodgerblue3, label="filtered trajectory")
    plot!(p_traj, obs_true_hist[:, 1], obs_true_hist[:, 2], lw=1.8, color=:forestgreen, alpha=0.8, label="obstacle center path")
    plot!(p_traj, obs_hat_hist[:, 1], obs_hat_hist[:, 2], lw=1.3, ls=:dashdot, color=:darkmagenta, alpha=0.8, label="recentered path")
    scatter!(p_traj, [X[1, 1]], [X[1, 2]], marker=:circle, ms=4, color=:dodgerblue3, label="start")
    scatter!(p_traj, [X[end, 1]], [X[end, 2]], marker=:star5, ms=7, color=:dodgerblue4, label="end")

    for (j, idx) in enumerate(obs_snap_idx)
        cx, cy = obs_true_hist[idx, 1], obs_true_hist[idx, 2]
        circx = cx .+ 1.0 .* cos.(theta_circle)
        circy = cy .+ 1.0 .* sin.(theta_circle)
        plot!(p_traj, circx, circy, color=:orange2, ls=:dash, lw=1.4,
            alpha=(j == length(obs_snap_idx) ? 0.9 : 0.5),
            label=(j == 1 ? "obstacle snapshots" : ""))
        scatter!(p_traj, [cx], [cy], color=:forestgreen, ms=3,
            alpha=(j == length(obs_snap_idx) ? 0.9 : 0.55),
            label=(j == 1 ? "obstacle center" : ""))
    end

    p_B = plot(ts_hist, B_hist, lw=2.2, color=:dodgerblue3,
        label="active barrier", xlabel="time", ylabel="B", title="Barrier margin", legend=:bottomright)
    hline!(p_B, [0.0], color=:orangered2, ls=:dash, lw=1.8, label="0")
    hline!(p_B, [-delta], color=:forestgreen, ls=:dot, lw=1.8, label="-δ")
    hline!(p_B, [-(delta + K)], color=:darkorchid2, ls=:dashdot, lw=1.8, label="-δ-K")
    for (a, b) in override_spans
        vspan!(p_B, [a, b], color=:gray85, alpha=0.35, label=false)
    end
    for tmark in sample_times
        vline!(p_B, [tmark], color=:gray70, lw=0.7, ls=:dot, alpha=0.35, label=false)
    end

    u1_nominal_only = [override_hist[i] ? NaN : u1_hist[i] for i in eachindex(u1_hist)]
    u1_override_only = [override_hist[i] ? u1_hist[i] : NaN for i in eachindex(u1_hist)]
    u2_nominal_only = [override_hist[i] ? NaN : u2_hist[i] for i in eachindex(u2_hist)]
    u2_override_only = [override_hist[i] ? u2_hist[i] : NaN for i in eachindex(u2_hist)]

    p_u = plot(
        ts_hist,
        u1_nominal_only,
        lw=2.0,
        color=:deepskyblue3,
        label="u1 applied (nominal mode)",
        xlabel="time",
        ylabel="control",
        title="Applied controls by mode",
        legend=:bottomright,
    )
    plot!(p_u, ts_hist, u1_override_only, lw=2.0, color=:orangered3, label="u1 applied (override mode)")
    plot!(p_u, ts_hist, u2_nominal_only, lw=1.8, color=:teal, ls=:dash, label="u2 applied (nominal mode)")
    plot!(p_u, ts_hist, u2_override_only, lw=1.8, color=:orange, ls=:dash, label="u2 applied (override mode)")
    for (a, b) in override_spans
        vspan!(p_u, [a, b], color=:gray85, alpha=0.35, label=false)
    end

    p_combined = plot(
        p_traj, p_B, p_u,
        layout=@layout([a{0.95w} ; b{0.6w} ; c{0.55w}]),
        size=(900, 1100),
    )

    mkpath("figures")
    savefig(p_traj, "figures/coordturn_traj.pdf")
    savefig(p_B, "figures/coordturn_barrier.pdf")
    savefig(p_u, "figures/coordturn_control.pdf")
    savefig(p_combined, "figures/coordturn_results_combined.pdf")

    anim_obj = @animate for k in 1:4:length(ts)
        p = plot(xlims=(path_center[1] - plot_half_span, path_center[1] + plot_half_span), ylims=(path_center[2] - plot_half_span, path_center[2] + plot_half_span), aspect_ratio=1,
            xlabel="x", ylabel="y", title="Repulsive barrier simulation (moving obstacle)")

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
        ann_x = path_center[1] + 0.58 * plot_half_span
        ann_y = path_center[2] - 0.60 * plot_half_span
        annotate!(p, ann_x, ann_y, text("t = $(round(ts[k], digits=2)) s", 9))
        if k > 1
            annotate!(p, ann_x, ann_y + 1.2, text("B = $(round(B_hist[min(k - 1, end)], digits=3))", 9))
            mode_str = override_hist[min(k - 1, end)] ? "override" : "nominal"
            annotate!(p, ann_x, ann_y + 2.4, text("mode = $mode_str", 9))
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
        p_combined=p_combined,
        anim=p_anim,
        gif_path=gif_path,
        traj_pdf_path="figures/coordturn_traj.pdf",
        barrier_pdf_path="figures/coordturn_barrier.pdf",
        control_pdf_path="figures/coordturn_control.pdf",
        combined_pdf_path="figures/coordturn_results_combined.pdf",
        min_dist=minimum(dist_true_hist),
        mean_track_err=mean(track_err_hist),
        max_track_err=maximum(track_err_hist),
    )
end

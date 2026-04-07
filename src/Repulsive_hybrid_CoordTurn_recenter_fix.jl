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

function reference_state(t;
    ref_style=:lazy8,
    p_start=[-15.0, -15.0],
    p_end=[15.0, 15.0],
    T_ref=20.0,
    lazy8_center=[0.0, 0.0],
    lazy8_scale=[18.0, 10.0],
    lazy8_cycles=1.0,
    lazy8_phase=0.0)

    if ref_style == :lazy8
        ω = (2pi * lazy8_cycles) / max(T_ref, 1e-6)
        σ = ω * t + lazy8_phase

        cx, cy = lazy8_center
        a, b = lazy8_scale
        r = [
            cx + a * sin(σ),
            cy + b * sin(σ) * cos(σ),
        ]
        r_dot = [
            a * ω * cos(σ),
            b * ω * cos(2.0 * σ),
        ]
        return r, r_dot
    end

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

function choose_best_override_control(state, center_hat, all_barriers, barrier_controls, dt;
    u_nom=(0.0, 0.0), u_track_weight=0.02)
    best_u = barrier_controls[1]
    best_score = Inf

    for u in barrier_controls
        s_next = state .+ dt .* coordturn_mode_rhs(state, u)
        s_next[4] = wrap_to_pi(s_next[4])
        _, b_next, _ = critical_barrier(all_barriers, s_next, center_hat)
        # Keep strict barrier prioritization but avoid runaway controls when several choices are similar.
        track_pen = u_track_weight * ((u[1] - u_nom[1])^2 + (u[2] - u_nom[2])^2)
        score = b_next + track_pen
        if score < best_score
            best_score = score
            best_u = u
        end
    end

    return best_u, best_score
end

function choose_nonpositive_override_control(state, center_hat, all_barriers, barrier_controls, dt;
    u_nom=(0.0, 0.0), u_track_weight=0.02, margin_target=0.0, prefer_deeper_margin=false)
    best_any_u = barrier_controls[1]
    best_any_b = Inf

    feasible_found = false
    best_feasible_u = barrier_controls[1]
    best_feasible_b = Inf
    best_feasible_track = Inf

    for u in barrier_controls
        s_next = state .+ dt .* coordturn_mode_rhs(state, u)
        s_next[4] = wrap_to_pi(s_next[4])
        _, b_next, _ = critical_barrier(all_barriers, s_next, center_hat)

        if b_next < best_any_b
            best_any_b = b_next
            best_any_u = u
        end

        if b_next <= margin_target
            track_pen = u_track_weight * ((u[1] - u_nom[1])^2 + (u[2] - u_nom[2])^2)
            if prefer_deeper_margin
                if (b_next < best_feasible_b - 1e-10) || (abs(b_next - best_feasible_b) <= 1e-10 && track_pen < best_feasible_track)
                    feasible_found = true
                    best_feasible_track = track_pen
                    best_feasible_u = u
                    best_feasible_b = b_next
                end
            else
                if track_pen < best_feasible_track
                    feasible_found = true
                    best_feasible_track = track_pen
                    best_feasible_u = u
                    best_feasible_b = b_next
                end
            end
        end
    end

    if feasible_found
        return best_feasible_u, best_feasible_b, true
    end

    return best_any_u, best_any_b, false
end

function sample_hold_barrier_policy(state, center_hat, all_barriers, barrier_controls, delta, K;
    override_active=false, locked_idx=1, locked_u=(0.0, 0.0))
    idx, B_crit, vals = critical_barrier(all_barriers, state, center_hat)

    if override_active
        if B_crit < -(delta + K)
            return false, (0.0, 0.0), B_crit, idx, vals
        end
        return true, locked_u, B_crit, locked_idx, vals
    end

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

function nominal_tracking_control(state, t;
    u1min=-5.0,
    u1max=5.0,
    u2min=-5.0,
    u2max=5.0,
    speed_gain=0.9,
    heading_gain=1.2,
    yaw_gain=1.5,
    ref_style=:lazy8,
    p_start=[-15.0, -15.0],
    p_end=[15.0, 15.0],
    T_ref=20.0,
    lazy8_center=[0.0, 0.0],
    lazy8_scale=[18.0, 10.0],
    lazy8_cycles=1.0,
    lazy8_phase=0.0)

    r, r_dot = reference_state(t + 1.0;
        ref_style=ref_style,
        p_start=p_start,
        p_end=p_end,
        T_ref=T_ref,
        lazy8_center=lazy8_center,
        lazy8_scale=lazy8_scale,
        lazy8_cycles=lazy8_cycles,
        lazy8_phase=lazy8_phase)
    v_des = clamp(norm(r_dot), 0.0, 8.0)

    theta_goal = atan(r[2] - state[2], r[1] - state[1])
    theta_err = wrap_to_pi(theta_goal - state[4])

    omega_des = heading_gain * theta_err
    u1_nom = clamp(speed_gain * (v_des - state[3]), u1min, u1max)
    u2_nom = clamp(yaw_gain * (omega_des - state[5]), u2min, u2max)
    return (u1_nom, u2_nom)
end

function run_repulsive_hybrid_coordturn_demo(all_barriers;
    barrier_controls=nothing,
    delta=1.0,
    k_override=1.0,
    override_B_threshold=nothing,
    override_B_hysteresis=0.0,
    override_dist_trigger=Inf,
    override_dist_release=nothing,
    strict_delta_enforcement=false,
    strict_delta_hysteresis=0.0,
    strict_u_track_weight=0.02,
    enforce_nonpositive_barrier=true,
    make_plots=true,
    save_outputs=true,
    save_only_validated=true,
    save_split_pdfs=false,
    combined_pdf_path="figures/coordturn_results_combined.pdf",
    gif_output_path="figures/repulsive_hybrid_coordturn_moving_obstacle_3panel.gif",
    dt=0.05,
    T=45.0,
    tau_steps=10,
    x0=[-15.0, -15.0, 0.0, pi / 4, 0.0, 0.0],
    ref_style=:lazy8,
    ref_start=[-15.0, -15.0],
    ref_end=[15.0, 15.0],
    ref_T=20.0,
    lazy8_center=[0.0, 0.0],
    lazy8_scale=[18.0, 10.0],
    lazy8_cycles=1.0,
    lazy8_phase=0.0,
    obs_amplitude=9.0,
    obs_lateral_bias=8.0,
    obs_cycles=2.0,
    obs_phase=pi / 2,
    plot_half_span=16.0,
    u1min=nothing,
    u1max=nothing,
    u2min=nothing,
    u2max=nothing,
    nominal_speed_gain=0.9,
    nominal_heading_gain=1.2,
    nominal_yaw_gain=1.5)

    N = Int(round(T / dt))
    recenter_steps = max(1, Int(tau_steps))
    state = copy(x0)
    K = k_override
    enter_thresh = isnothing(override_B_threshold) ? -(delta + K) : override_B_threshold
    exit_thresh = -(delta + K)
    dist_release = isnothing(override_dist_release) ? override_dist_trigger : override_dist_release
    barrier_target = strict_delta_enforcement ? -delta : 0.0

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
    B_proj_hist = zeros(N)
    B_idx_hist = zeros(Int, N)
    override_hist = falses(N)
    margin_violation_hist = falses(N)
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
    ref_hist[1, :] .= reference_state(0.0;
        ref_style=ref_style,
        p_start=ref_start,
        p_end=ref_end,
        T_ref=ref_T,
        lazy8_center=lazy8_center,
        lazy8_scale=lazy8_scale,
        lazy8_cycles=lazy8_cycles,
        lazy8_phase=lazy8_phase)[1]

    hold_override = false
    hold_u = (0.0, 0.0)

    for k in 1:N
        t = (k - 1) * dt

        u_nom = nominal_tracking_control(state, t;
            u1min=u1min,
            u1max=u1max,
            u2min=u2min,
            u2max=u2max,
            speed_gain=nominal_speed_gain,
            heading_gain=nominal_heading_gain,
            yaw_gain=nominal_yaw_gain,
            ref_style=ref_style,
            p_start=ref_start,
            p_end=ref_end,
            T_ref=ref_T,
            lazy8_center=lazy8_center,
            lazy8_scale=lazy8_scale,
            lazy8_cycles=lazy8_cycles,
            lazy8_phase=lazy8_phase)

        if (k - 1) % recenter_steps == 0
            obs_hat = copy(obstacle_state(t;
                p_start=ref_start,
                p_end=ref_end,
                T_path=ref_T,
                amplitude=obs_amplitude,
                lateral_bias=obs_lateral_bias,
                cycles=obs_cycles,
                phase=obs_phase)[1])
        end

        idx_now, B_now, _ = critical_barrier(all_barriers, state, obs_hat)
        d_hat = hypot(state[1] - obs_hat[1], state[2] - obs_hat[2])

        if strict_delta_enforcement
            if hold_override
                if B_now < exit_thresh
                    hold_override = false
                    hold_u = (0.0, 0.0)
                end
            elseif (d_hat <= override_dist_trigger) && (B_now > enter_thresh)
                hold_override = true
                hold_u, _, _ = choose_nonpositive_override_control(state, obs_hat, all_barriers, barrier_controls, dt;
                    u_nom=u_nom,
                    u_track_weight=strict_u_track_weight,
                    margin_target=barrier_target,
                    prefer_deeper_margin=true)
            end
        else
            if hold_override
                if B_now < exit_thresh
                    hold_override = false
                    hold_u = (0.0, 0.0)
                end
            elseif (d_hat <= override_dist_trigger) && (B_now > enter_thresh)
                hold_override = true
                hold_u = barrier_controls[idx_now]
            end
        end

        u = hold_override ? hold_u : u_nom
        u = (
            clamp(u[1], u1min, u1max),
            clamp(u[2], u2min, u2max),
        )

        if enforce_nonpositive_barrier
            s_next_pred = state .+ dt .* coordturn_mode_rhs(state, u)
            s_next_pred[4] = wrap_to_pi(s_next_pred[4])
            _, b_next_pred, _ = critical_barrier(all_barriers, s_next_pred, obs_hat)

            if (b_next_pred > barrier_target) && !hold_override && (d_hat <= override_dist_trigger)
                u_safe, _, _ = choose_nonpositive_override_control(state, obs_hat, all_barriers, barrier_controls, dt;
                    u_nom=u_nom,
                    u_track_weight=strict_u_track_weight,
                    margin_target=barrier_target,
                    prefer_deeper_margin=true)
                hold_override = true
                hold_u = u_safe
                u = (
                    clamp(u_safe[1], u1min, u1max),
                    clamp(u_safe[2], u2min, u2max),
                )
            end
        end

        state_prev = copy(state)
        dstate = coordturn_mode_rhs(state_prev, u)
        state_trial = state_prev .+ dt .* dstate
        state_trial[4] = wrap_to_pi(state_trial[4])

        idx, B_crit, _ = critical_barrier(all_barriers, state_trial, obs_hat)

        if enforce_nonpositive_barrier && (B_crit > barrier_target)
            if !hold_override && (d_hat <= override_dist_trigger)
                hold_override = true
                u_safe, _, _ = choose_nonpositive_override_control(state_prev, obs_hat, all_barriers, barrier_controls, dt;
                    u_nom=u_nom,
                    u_track_weight=strict_u_track_weight,
                    margin_target=barrier_target,
                    prefer_deeper_margin=true)
                hold_u = u_safe
                u = (
                    clamp(u_safe[1], u1min, u1max),
                    clamp(u_safe[2], u2min, u2max),
                )

                dstate_safe = coordturn_mode_rhs(state_prev, u)
                state_trial = state_prev .+ dt .* dstate_safe
                state_trial[4] = wrap_to_pi(state_trial[4])
                idx, B_crit, _ = critical_barrier(all_barriers, state_trial, obs_hat)
            end

            if B_crit > barrier_target
                # Last-resort safety: freeze this step instead of allowing margin target violation.
                state_trial = state_prev
                idx, B_crit, _ = critical_barrier(all_barriers, state_trial, obs_hat)
            end
        end

        state = state_trial

        X[k + 1, :] .= state
        B_hist[k] = B_crit
        B_proj_hist[k] = min(B_crit, barrier_target)
        B_idx_hist[k] = idx
        override_hist[k] = hold_override
        margin_violation_hist[k] = B_crit > barrier_target

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
        ref_hist[k + 1, :] .= reference_state(t + dt;
            ref_style=ref_style,
            p_start=ref_start,
            p_end=ref_end,
            T_ref=ref_T,
            lazy8_center=lazy8_center,
            lazy8_scale=lazy8_scale,
            lazy8_cycles=lazy8_cycles,
            lazy8_phase=lazy8_phase)[1]

        obs_now = obstacle_state(t;
            p_start=ref_start,
            p_end=ref_end,
            T_path=ref_T,
            amplitude=obs_amplitude,
            lateral_bias=obs_lateral_bias,
            cycles=obs_cycles,
            phase=obs_phase)[1]
        dist_true_hist[k] = hypot(state[1] - obs_now[1], state[2] - obs_now[2])
        r_now = reference_state(t;
            ref_style=ref_style,
            p_start=ref_start,
            p_end=ref_end,
            T_ref=ref_T,
            lazy8_center=lazy8_center,
            lazy8_scale=lazy8_scale,
            lazy8_cycles=lazy8_cycles,
            lazy8_phase=lazy8_phase)[1]
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
    sample_times = ts[1:recenter_steps:end]
    obs_snap_idx = unique(round.(Int, range(1, length(ts), length=min(5, length(ts)))))
    theta_circle = LinRange(0, 2pi, 200)
    path_center = ref_style == :lazy8 ? lazy8_center : 0.5 .* (ref_start .+ ref_end)

    p_traj = nothing
    p_B = nothing
    p_u = nothing
    p_combined = nothing
    p_anim = nothing
    gif_path = ""
    traj_pdf_path = ""
    barrier_pdf_path = ""
    control_pdf_path = ""
    combined_pdf_saved_path = ""
    validated_result = count(margin_violation_hist) == 0
    do_save_outputs = save_outputs && (!save_only_validated || validated_result)

    if make_plots
        default(
            legendfontsize=12,
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
            legend=:outerright,
            legendfontsize=10,
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
        if strict_delta_enforcement
            plot!(p_B, ts_hist, B_proj_hist, lw=1.8, ls=:dash, color=:dodgerblue4, alpha=0.55, label="projected at -δ")
        end
        hline!(p_B, [0.0], color=:orangered2, ls=:dash, lw=1.8, label="0")
        hline!(p_B, [-delta], color=:forestgreen, ls=:dot, lw=1.8, label="-δ")
        hline!(p_B, [-(delta + K)], color=:darkorchid2, ls=:dashdot, lw=1.8, label="-δ-K")

        # Replace dense guide lines with compact bottom markers to reduce visual clutter.
        b_min = minimum(B_hist)
        b_max = maximum(B_hist)
        b_span = max(b_max - b_min, 1e-3)
        marker_base = b_min - 0.04 * b_span
        marker_upper = b_min - 0.015 * b_span

        n_sample_markers = min(length(sample_times), 36)
        sample_stride = max(1, Int(ceil(length(sample_times) / max(n_sample_markers, 1))))
        sample_marker_times = sample_times[1:sample_stride:end]
        scatter!(p_B, sample_marker_times, fill(marker_base, length(sample_marker_times));
            ms=1.6, marker=:circle, color=:gray55, alpha=0.35, label=false)

        override_times = ts_hist[override_hist]
        if !isempty(override_times)
            n_override_markers = min(length(override_times), 60)
            override_stride = max(1, Int(ceil(length(override_times) / max(n_override_markers, 1))))
            override_marker_times = override_times[1:override_stride:end]
            scatter!(p_B, override_marker_times, fill(marker_upper, length(override_marker_times));
                ms=2.0, marker=:utriangle, color=:darkorange2, alpha=0.60, markerstrokewidth=0.0, label="override markers")
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

        if do_save_outputs
            mkpath("figures")
            if save_split_pdfs
                traj_pdf_path = "figures/coordturn_traj.pdf"
                barrier_pdf_path = "figures/coordturn_barrier.pdf"
                control_pdf_path = "figures/coordturn_control.pdf"
                savefig(p_traj, traj_pdf_path)
                savefig(p_B, barrier_pdf_path)
                savefig(p_u, control_pdf_path)
            end
            combined_pdf_saved_path = combined_pdf_path
            savefig(p_combined, combined_pdf_saved_path)
        end

        u_span = max(max(u1max - u1min, u2max - u2min), 1e-3)
        u_pad = 0.10 * u_span
        b_pad = 0.08 * b_span
        xlims_ws = (path_center[1] - plot_half_span, path_center[1] + plot_half_span)
        ylims_ws = (path_center[2] - plot_half_span, path_center[2] + plot_half_span)
        theta_anim = LinRange(0, 2pi, 120)

        anim_obj = @animate for k in 1:4:length(ts)
            kh = min(max(k - 1, 1), length(ts_hist))
            t_now = ts[k]
            t_cursor = ts_hist[kh]

            p_ws = plot(
                xlims=xlims_ws,
                ylims=ylims_ws,
                aspect_ratio=1,
                xlabel="x",
                ylabel="y",
                title="Coordinated-turn - Workspace",
                titlefont=font(12, :bold),
                legend=:outerbottom,
                legend_column=4,
            )

            plot!(p_ws, ref_hist[:, 1], ref_hist[:, 2], lw=2, ls=:dash, color=:gray45, label="reference")
            plot!(p_ws, X[1:k, 1], X[1:k, 2], lw=2.5, color=:dodgerblue3, label="trajectory")

            if k > 1 && override_hist[min(k - 1, end)]
                scatter!(p_ws, [X[k, 1]], [X[k, 2]], color=:orangered3, markersize=5, label="override")
            else
                scatter!(p_ws, [X[k, 1]], [X[k, 2]], color=:dodgerblue3, markersize=5, label="state")
            end

            cx, cy = obs_true_hist[k, 1], obs_true_hist[k, 2]
            hx, hy = obs_hat_hist[k, 1], obs_hat_hist[k, 2]
            obs_xk = cx .+ 1.0 .* cos.(theta_anim)
            obs_yk = cy .+ 1.0 .* sin.(theta_anim)
            plot!(p_ws, obs_xk, obs_yk, seriestype=:shape, color=:orange, fillalpha=0.10, linealpha=0.0, label=false)
            plot!(p_ws, obs_xk, obs_yk, lw=2, color=:black, label="obstacle")
            scatter!(p_ws, [cx], [cy], color=:black, markersize=4, label="obs center")
            scatter!(p_ws, [hx], [hy], marker=:x, color=:darkmagenta, markersize=5, label="recentered")

            ann_x = xlims_ws[1] + 0.05 * (xlims_ws[2] - xlims_ws[1])
            ann_y = ylims_ws[2] - 0.07 * (ylims_ws[2] - ylims_ws[1])
            mode_str = override_hist[min(kh, end)] ? "override" : "nominal"
            annotate!(p_ws, ann_x, ann_y, text("t = $(round(t_now, digits=2)) s", 9))
            annotate!(p_ws, ann_x, ann_y - 1.2, text("B = $(round(B_hist[kh], digits=3))", 9))
            annotate!(p_ws, ann_x, ann_y - 2.4, text("mode = $mode_str", 9))

            p_B_anim = plot(
                xlabel="time",
                ylabel="B",
                xlims=(ts_hist[1], ts_hist[end]),
                ylims=(b_min - b_pad, b_max + b_pad),
                title="Coordinated-turn - Barrier margin",
                titlefont=font(11, :bold),
                legend=:topright,
            )
            plot!(p_B_anim, ts_hist, B_hist, lw=1.2, color=:gray70, alpha=0.55, label=false)
            plot!(p_B_anim, ts_hist[1:kh], B_hist[1:kh], lw=2.4, color=:dodgerblue3, label="B(t)")
            if strict_delta_enforcement
                plot!(p_B_anim, ts_hist[1:kh], B_proj_hist[1:kh], lw=1.6, ls=:dash, color=:dodgerblue4, alpha=0.70, label="projected")
            end
            scatter!(p_B_anim, [ts_hist[kh]], [B_hist[kh]], color=:dodgerblue4, markersize=4, label=false)
            hline!(p_B_anim, [0.0], color=:orangered2, ls=:dash, lw=1.6, label="0")
            hline!(p_B_anim, [-delta], color=:forestgreen, ls=:dot, lw=1.6, label="-δ")
            hline!(p_B_anim, [-(delta + K)], color=:darkorchid2, ls=:dashdot, lw=1.6, label="-δ-K")
            vline!(p_B_anim, [t_cursor], color=:black, ls=:dot, lw=1.2, alpha=0.55, label=false)

            u1_nom_anim = [override_hist[i] ? NaN : u1_hist[i] for i in 1:kh]
            u1_ovr_anim = [override_hist[i] ? u1_hist[i] : NaN for i in 1:kh]
            u2_nom_anim = [override_hist[i] ? NaN : u2_hist[i] for i in 1:kh]
            u2_ovr_anim = [override_hist[i] ? u2_hist[i] : NaN for i in 1:kh]

            p_u_anim = plot(
                xlabel="time",
                ylabel="control",
                xlims=(ts_hist[1], ts_hist[end]),
                ylims=(min(u1min, u2min) - u_pad, max(u1max, u2max) + u_pad),
                title="Coordinated-turn - Applied controls",
                titlefont=font(11, :bold),
                legend=:bottomright,
            )
            plot!(p_u_anim, ts_hist, u1_hist, lw=1.2, color=:gray70, alpha=0.45, label=false)
            plot!(p_u_anim, ts_hist, u2_hist, lw=1.2, color=:gray70, alpha=0.45, ls=:dash, label=false)
            plot!(p_u_anim, ts_hist[1:kh], u1_nom_anim, lw=2.2, color=:deepskyblue3, label="u1 nominal")
            plot!(p_u_anim, ts_hist[1:kh], u1_ovr_anim, lw=2.2, color=:orangered3, label="u1 override")
            plot!(p_u_anim, ts_hist[1:kh], u2_nom_anim, lw=1.9, color=:teal, ls=:dash, label="u2 nominal")
            plot!(p_u_anim, ts_hist[1:kh], u2_ovr_anim, lw=1.9, color=:orange, ls=:dash, label="u2 override")
            vline!(p_u_anim, [t_cursor], color=:black, ls=:dot, lw=1.2, alpha=0.55, label=false)

            plot(
                p_ws,
                p_B_anim,
                p_u_anim,
                layout=@layout([a{0.52w} [b; c]]),
                size=(1280, 720),
            )
        end

        if do_save_outputs
            gif_path = gif_output_path
            gif(anim_obj, gif_path, fps=20)
        end

        p_anim = plot(X[:, 1], X[:, 2], lw=2.5, label="trajectory preview", aspect_ratio=1)
        plot!(p_anim, ref_hist[:, 1], ref_hist[:, 2], lw=2, ls=:dash, label="reference")
        plot!(p_anim, title="Preview (full animation is in GIF)", xlabel="x", ylabel="y")
    end

    println("Simulation finished")
    println("minimum recentered barrier value = ", minimum(B_hist))
    println("maximum recentered barrier value = ", maximum(B_hist))
    println("minimum true obstacle distance = ", minimum(dist_true_hist))
    println("mean tracking error = ", mean(track_err_hist))
    println("max tracking error = ", maximum(track_err_hist))
    println("number of barrier overrides = ", count(override_hist))
    println("number of B > -delta violations = ", count(margin_violation_hist))
    println("override enter threshold = ", enter_thresh)
    println("override exit threshold = ", exit_thresh)
    println("override distance trigger = ", override_dist_trigger)
    println("override distance release = ", dist_release)
    println("strict delta enforcement = ", strict_delta_enforcement)
    if make_plots && do_save_outputs
        println("Animation saved to ", gif_path)
    elseif make_plots && save_outputs && save_only_validated && !validated_result
        println("Skipping file save because run is not validated (B > -delta violations = ", count(margin_violation_hist), ").")
    end

    return (
        ts=ts,
        X=X,
        B_hist=B_hist,
        B_proj_hist=B_proj_hist,
        B_idx_hist=B_idx_hist,
        override_hist=override_hist,
        margin_violation_hist=margin_violation_hist,
        u1_nom_hist=u1_nom_hist,
        u2_nom_hist=u2_nom_hist,
        u1_hist=u1_hist,
        u2_hist=u2_hist,
        obs_true_hist=obs_true_hist,
        obs_hat_hist=obs_hat_hist,
        ref_hist=ref_hist,
        dist_true_hist=dist_true_hist,
        track_err_hist=track_err_hist,
        p_traj=p_traj,
        p_B=p_B,
        p_u=p_u,
        p_combined=p_combined,
        anim=p_anim,
        gif_path=gif_path,
        traj_pdf_path=traj_pdf_path,
        barrier_pdf_path=barrier_pdf_path,
        control_pdf_path=control_pdf_path,
        combined_pdf_path=combined_pdf_saved_path,
        min_dist=minimum(dist_true_hist),
        mean_track_err=mean(track_err_hist),
        max_track_err=maximum(track_err_hist),
    )
end

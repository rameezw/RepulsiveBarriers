using LinearAlgebra
using Plots
using Statistics

wrap_to_pi(ang) = mod(ang + π, 2π) - π

function obstacle_state(t; center=[0.0, 0.0], radius=8.0, cross_angle=0.0, cross_speed=0.4, t_cross=8.0, half_travel=Inf, away_offset=6.0)
    # Continuous trajectory: cross near the chosen reference point then keep moving upward.
    # Keep legacy arguments for compatibility with existing notebook calls.
    angles = cross_angle isa Number ? [Float64(cross_angle)] : Float64.(collect(cross_angle))
    θ_cross = isempty(angles) ? (pi / 2) : atan(mean(sin.(angles)), mean(cos.(angles)))

    t_crosses = if t_cross isa Number
        [Float64(t_cross)]
    else
        Float64.(collect(t_cross))
    end
    t_mid = isempty(t_crosses) ? 8.0 : median(t_crosses)

    cross_pt = center .+ radius .* [cos(θ_cross), sin(θ_cross)]

    # Slow monotone upward drift with a tiny horizontal meander for visual separation.
    v_up = max(0.08, abs(cross_speed))
    x_amp = 0.12 * radius
    x_freq = 0.22

    x = cross_pt[1] - x_amp * sin(x_freq * (t - t_mid))
    y = cross_pt[2] + v_up * (t - t_mid)
    xdot = x_amp * x_freq * cos(x_freq * (t - t_mid))
    ydot = v_up

    c = [x, y]
    ċ = [xdot, ydot]
    return c, ċ
end

function reference_state(t; center=[0.0, 0.0], radius=8.0, omega=0.18, phase0=0.0)
    θref = phase0 + omega * t
    r = [center[1] + radius * cos(θref), center[2] + radius * sin(θref)]
    ṙ = [-radius * omega * sin(θref), radius * omega * cos(θref)]
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

function nominal_tracking_control(x, t; ref_center=[0.0, 0.0], ref_radius=8.0, ref_omega=0.18, ref_phase=0.0)
    r_la, _ = reference_state(t + 1.0; center=ref_center, radius=ref_radius, omega=ref_omega, phase0=ref_phase)
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

    # Strict sample-and-hold trigger in the repulsion band near the safety boundary.
    if B_crit > -δ-K
        return true, barrier_controls[idx_min], B_crit, idx_min, vals
    end

    return false, 0.0, B_crit, idx_min, vals
end

function run_repulsive_hybrid_dubins_demo(all_barriers; v=5, τ_steps=1, dt=0.05, T=40.0, k_override=1.0, δ=1.0, barrier_controls=nothing, umin=nothing, umax=nothing, x0=nothing,
    ref_center=[0.0, 0.0], ref_radius=8.0, ref_omega=0.18, ref_phase=0.0,
    obs_cross_angle=0.0, obs_cross_speed=1.2, obs_t_cross=8.0, obs_half_travel=Inf, obs_away_offset=6.0,
    plot_half_span=12.0)
    if isnothing(x0)
        x0 = [
            ref_center[1] + ref_radius * cos(ref_phase),
            ref_center[2] + ref_radius * sin(ref_phase),
            wrap_to_pi(ref_phase + π / 2),
        ]
    end
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

    obs_true_hist[1, :] .= obstacle_state(0.0; center=ref_center, radius=ref_radius, cross_angle=obs_cross_angle, cross_speed=obs_cross_speed, t_cross=obs_t_cross, half_travel=obs_half_travel, away_offset=obs_away_offset)[1]
    obs_hat = copy(obs_true_hist[1, :])
    obs_hat_hist[1, :] .= obs_hat
    ref_hist[1, :] .= reference_state(0.0; center=ref_center, radius=ref_radius, omega=ref_omega, phase0=ref_phase)[1]

    hold_override_on = false
    hold_u_bar = 0.0
    hold_B_val = 0.0
    hold_idx = 1

    for k in 1:N
        t = (k - 1) * dt

        if (k - 1) % τ_steps == 0
            obs_hat = copy(obstacle_state(t; center=ref_center, radius=ref_radius, cross_angle=obs_cross_angle, cross_speed=obs_cross_speed, t_cross=obs_t_cross, half_travel=obs_half_travel, away_offset=obs_away_offset)[1])
        end

        v_cmd = v

        u_nom = nominal_tracking_control(x, t; ref_center=ref_center, ref_radius=ref_radius, ref_omega=ref_omega, ref_phase=ref_phase)
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
        obs_now = obstacle_state(t; center=ref_center, radius=ref_radius, cross_angle=obs_cross_angle, cross_speed=obs_cross_speed, t_cross=obs_t_cross, half_travel=obs_half_travel, away_offset=obs_away_offset)[1]
        dist_true_hist[k] = hypot(x[1] - obs_now[1], x[2] - obs_now[2])
        r_now = reference_state(t; center=ref_center, radius=ref_radius, omega=ref_omega, phase0=ref_phase)[1]
        track_err_hist[k] = hypot(x[1] - r_now[1], x[2] - r_now[2])

        obs_true_hist[k + 1, :] .= obstacle_state(t + dt; center=ref_center, radius=ref_radius, cross_angle=obs_cross_angle, cross_speed=obs_cross_speed, t_cross=obs_t_cross, half_travel=obs_half_travel, away_offset=obs_away_offset)[1]
        obs_hat_hist[k + 1, :] .= obs_hat
        ref_hist[k + 1, :] .= reference_state(t + dt; center=ref_center, radius=ref_radius, omega=ref_omega, phase0=ref_phase)[1]
    end

    # Build time grids from N to keep lengths consistent even when T/dt is non-integer.
    ts = collect(0.0:dt:(N * dt))
    ts_hist = ts[1:end-1]

    # ---------- Publication-style plotting helpers ----------
    function _override_spans(flags, ts_hist, dt)
        spans = Tuple{Float64,Float64}[]
        in_span = false
        t0 = 0.0
        for i in eachindex(flags)
            if flags[i] && !in_span
                in_span = true
                t0 = ts_hist[i]
            elseif !flags[i] && in_span
                push!(spans, (t0, ts_hist[i]))
                in_span = false
            end
        end
        if in_span
            push!(spans, (t0, ts_hist[end] + dt))
        end
        return spans
    end

    override_spans = _override_spans(barrier_override_hist, ts_hist, dt)
    sample_times = ts[1:τ_steps:end]
    obs_snap_idx = unique(round.(Int, range(1, length(ts), length=min(5, length(ts)))))
    θc = LinRange(0, 2π, 200)

    default(
        legendfontsize=8,
        guidefontsize=10,
        tickfontsize=8,
        titlefontsize=11,
        linewidth=2,
        framestyle=:box,
        gridalpha=0.18,
        foreground_color_legend=nothing,
        background_color_legend=RGBA(1,1,1,0.85),
        dpi=220,
    )

    p_traj = plot(
        aspect_ratio=1,
        xlabel="x", ylabel="y",
        xlims=(ref_center[1] - plot_half_span, ref_center[1] + plot_half_span),
        ylims=(ref_center[2] - plot_half_span, ref_center[2] + plot_half_span),
        title="Workspace trajectory",
        legend=:bottomright,
    )
    plot!(p_traj, ref_hist[:, 1], ref_hist[:, 2], lw=2, ls=:dash, color=:gray45, label="reference")
    plot!(p_traj, X[:, 1], X[:, 2], lw=2.6, color=:dodgerblue3, label="filtered trajectory")
    plot!(p_traj, obs_true_hist[:, 1], obs_true_hist[:, 2], lw=1.8, color=:forestgreen, alpha=0.8, label="obstacle center path")
    plot!(p_traj, obs_hat_hist[:, 1], obs_hat_hist[:, 2], lw=1.3, ls=:dashdot, color=:darkmagenta, alpha=0.8, label="recentered path")
    scatter!(p_traj, [X[1,1]], [X[1,2]], marker=:circle, ms=4, color=:dodgerblue3, label="start")
    scatter!(p_traj, [X[end,1]], [X[end,2]], marker=:star5, ms=7, color=:dodgerblue4, label="end")

    for (j, idx) in enumerate(obs_snap_idx)
        cx, cy = obs_true_hist[idx, 1], obs_true_hist[idx, 2]
        circx = cx .+ 1.0 .* cos.(θc)
        circy = cy .+ 1.0 .* sin.(θc)
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
    hline!(p_B, [-δ], color=:forestgreen, ls=:dot, lw=1.8, label="-δ")
    hline!(p_B, [-(δ + K)], color=:darkorchid2, ls=:dashdot, lw=1.8, label="-δ-K")
    for (a, b) in override_spans
        vspan!(p_B, [a, b], color=:gray85, alpha=0.35, label=false)
    end
    for tmark in sample_times
        vline!(p_B, [tmark], color=:gray70, lw=0.7, ls=:dot, alpha=0.35, label=false)
    end

    # Show a single applied control trace, color-coded by execution mode.
    u_nominal_only = [barrier_override_hist[i] ? NaN : u_hist[i] for i in eachindex(u_hist)]
    u_override_only = [barrier_override_hist[i] ? u_hist[i] : NaN for i in eachindex(u_hist)]

    p_u = plot(
        ts_hist,
        u_nominal_only,
        lw=2.0,
        color=:deepskyblue3,
        label="u_applied (nominal mode)",
        xlabel="time",
        ylabel="turn-rate",
        title="Applied control by mode",
        legend=:bottomright,
    )
    plot!(p_u, ts_hist, u_override_only, lw=2.0, color=:orangered3, label="u_applied (override mode)")
    for (a, b) in override_spans
        vspan!(p_u, [a, b], color=:gray85, alpha=0.35, label=false)
    end

    p_combined = plot(
        p_traj, p_B, p_u,
        layout=@layout([a{0.95w} ; b{0.6w} ; c{0.55w}]),
        size=(900, 1100),
    )

    mkpath("figures")
    savefig(p_traj, "figures/dubins_traj.pdf")
    savefig(p_B, "figures/dubins_barrier.pdf")
    savefig(p_u, "figures/dubins_control.pdf")
    savefig(p_combined, "figures/dubins_results_combined.pdf")
    anim = @animate for k in 1:4:length(ts)
        p = plot(
            xlims=(ref_center[1] - plot_half_span, ref_center[1] + plot_half_span),
            ylims=(ref_center[2] - plot_half_span, ref_center[2] + plot_half_span),
            aspect_ratio=1,
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

        annotate!(p, 11.5, -9.6, text("t = $(round(ts[k], digits=2)) s", 9))
        if k > 1
            annotate!(p, 9.5, -8.5, text("B = $(round(B_hist[min(k - 1, end)], digits=3))", 9))
            mode_str = barrier_override_hist[min(k - 1, end)] ? "override" : "nominal"
            annotate!(p, 9.5, -7.3, text("mode = $mode_str", 9))
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
        p_combined=p_combined,
        anim=p_anim,
        gif_path=gif_path,
        min_dist=minimum(dist_true_hist),
        mean_track_err=mean(track_err_hist),
        max_track_err=maximum(track_err_hist),
    )
end

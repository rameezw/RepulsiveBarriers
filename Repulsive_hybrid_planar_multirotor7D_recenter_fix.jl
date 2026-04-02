module RepulsiveHybridPlanarMultirotorRecenterFix

using Plots
using Statistics

export run_repulsive_hybrid_planar_multirotor_demo

function make_obstacle_trajectory(v_obs::Tuple{Float64, Float64}, x0_obs::Tuple{Float64, Float64})
    vx, vy = v_obs
    x0, y0 = x0_obs
    ax = max(abs(vx), 0.15)
    ay = max(abs(vy), 0.12)
    wx = max(abs(vx), 0.08)
    wy = max(abs(vy), 0.06)
    return t -> [x0 + ax * sin(wx * t), y0 + ay * cos(wy * t)]
end

function reference_state(t::Float64; p_start::Tuple{Float64, Float64}=(-10.0, -10.0), p_end::Tuple{Float64, Float64}=(10.0, 10.0), T_ref::Float64=6.5)
    return reference_state(t;
        ref_style=:line,
        p_start=p_start,
        p_end=p_end,
        T_ref=T_ref,
        lazy8_center=(0.0, 0.0),
        lazy8_scale=(8.0, 5.0),
        lazy8_cycles=1.0,
        lazy8_phase=0.0,
    )
end

function reference_state(t::Float64;
    ref_style::Symbol=:line,
    p_start::Tuple{Float64, Float64}=(-10.0, -10.0),
    p_end::Tuple{Float64, Float64}=(10.0, 10.0),
    T_ref::Float64=6.5,
    lazy8_center::Tuple{Float64, Float64}=(0.0, 0.0),
    lazy8_scale::Tuple{Float64, Float64}=(8.0, 5.0),
    lazy8_cycles::Float64=1.0,
    lazy8_phase::Float64=0.0)

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

    alpha = clamp(t / max(T_ref, 1e-6), 0.0, 1.0)
    s = alpha^2 * (3.0 - 2.0 * alpha)
    ds_dt = (0.0 < t < T_ref) ? (6.0 * alpha * (1.0 - alpha) / max(T_ref, 1e-6)) : 0.0

    r = [
        (1.0 - s) * p_start[1] + s * p_end[1],
        (1.0 - s) * p_start[2] + s * p_end[2],
    ]
    r_dot = [
        ds_dt * (p_end[1] - p_start[1]),
        ds_dt * (p_end[2] - p_start[2]),
    ]
    return r, r_dot
end

function recenter_state(x::AbstractVector{<:Real}, c::AbstractVector{<:Real})
    return [x[1] - c[1], x[2] - c[2], x[3], x[4], x[5], x[6], x[7]]
end

function eval_barrier(B, xhat::AbstractVector{<:Real})
    return B(xhat...)
end

function critical_barrier(Bs::Vector, xhat::AbstractVector{<:Real})
    vals = [eval_barrier(B, xhat) for B in Bs]
    idx = argmin(vals)
    return vals[idx], idx, vals
end

function choose_best_override_control(xk::Vector{Float64}, c_hat::Vector{Float64}, U::Vector{Vector{Float64}}, Bs::Vector, dt::Float64;
    u_nom::Tuple{Float64, Float64}=(0.0, 0.0),
    u_track_weight::Float64=0.03)

    best_u = (U[1][1], U[1][2])
    best_score = Inf
    for ui in U
        u = (ui[1], ui[2])
        xnext = xk .+ dt .* dynamics_rhs(xk, u)
        xnext[5] = atan(sin(xnext[5]), cos(xnext[5]))
        xhat_next = recenter_state(xnext, c_hat)
        b_next, _, _ = critical_barrier(Bs, xhat_next)
        track_pen = u_track_weight * ((u[1] - u_nom[1])^2 + (u[2] - u_nom[2])^2)
        score = b_next + track_pen
        if score < best_score
            best_score = score
            best_u = u
        end
    end
    return best_u, best_score
end

function in_region(theta::Float64, idx::Int)
    lb = (idx - 3) * pi / 2
    ub = (idx - 2) * pi / 2
    return (theta >= lb) && (theta <= ub)
end

function dynamics_rhs(x::Vector{Float64}, u::Tuple{Float64, Float64})
    u1, u2 = u
    theta = x[5]
    err = x[7]
    usum = u1 + u2

    dx = zeros(Float64, 7)
    dx[1] = x[3]
    dx[2] = x[4]

    # Match the exact piecewise planar multirotor dynamics used during synthesis.
    if theta <= -pi/2
        dx[3] = usum * (-2.0 / pi) * (theta + pi) - 0.2 * err
        dx[4] = usum * ((2.0 / pi) * (theta + pi / 2.0) - 0.2 * err) - 2.0
    elseif theta <= 0.0
        dx[3] = usum * (2.0 / pi) * theta + 0.2 * err
        dx[4] = usum * ((2.0 / pi) * (theta + pi / 2.0) - 0.2 * err) - 2.0
    elseif theta <= pi/2
        dx[3] = usum * (2.0 / pi) * theta + 0.2 * err
        dx[4] = usum * ((-2.0 / pi) * (theta - pi / 2.0) + 0.2 * err) - 2.0
    else
        dx[3] = usum * (-2.0 / pi) * (theta - pi) - 0.2 * err
        dx[4] = usum * ((-2.0 / pi) * (theta - pi / 2.0) + 0.2 * err) - 2.0
    end

    dx[5] = x[6]
    dx[6] = u1 - u2
    dx[7] = 0.0
    return dx
end

function nominal_controller(x::Vector{Float64}, t::Float64;
    umax::Float64=2.0,
    ref_style::Symbol=:line,
    ref_start::Tuple{Float64, Float64}=(-10.0, -10.0),
    ref_end::Tuple{Float64, Float64}=(10.0, 10.0),
    ref_T::Float64=6.5,
    lazy8_center::Tuple{Float64, Float64}=(0.0, 0.0),
    lazy8_scale::Tuple{Float64, Float64}=(8.0, 5.0),
    lazy8_cycles::Float64=1.0,
    lazy8_phase::Float64=0.0,
    dt_nom::Float64=0.12)

    r, r_dot = reference_state(t + 4.0 * dt_nom;
        ref_style=ref_style,
        p_start=ref_start,
        p_end=ref_end,
        T_ref=ref_T,
        lazy8_center=lazy8_center,
        lazy8_scale=lazy8_scale,
        lazy8_cycles=lazy8_cycles,
        lazy8_phase=lazy8_phase)
    ex = r[1] - x[1]
    ey = r[2] - x[2]
    evx = r_dot[1] - x[3]
    evy = r_dot[2] - x[4]

    # Piecewise dynamics lose vertical effectiveness as |theta| grows. Compensate this loss.
    theta = x[5]
    lift_gain = if theta <= -pi/2
        (-2.0 / pi) * (theta + pi / 2.0)
    elseif theta <= 0.0
        (2.0 / pi) * (theta + pi / 2.0)
    elseif theta <= pi/2
        (-2.0 / pi) * (theta - pi / 2.0)
    else
        (2.0 / pi) * (theta - pi / 2.0)
    end
    lift_gain = clamp(lift_gain, 0.35, 1.0)

    sum_u_raw = 2.0 + 0.70 * ey + 0.45 * evy
    sum_u = clamp(sum_u_raw / lift_gain, 0.7, 3.0)

    theta_ref = clamp(0.10 * ex + 0.06 * evx, -0.5, 0.5)
    theta_err = atan(sin(theta_ref - x[5]), cos(theta_ref - x[5]))
    diff_u = clamp(1.2 * theta_err - 0.9 * x[6], -0.55, 0.55)

    u_floor = 0.35
    u1 = clamp(0.5 * (sum_u + diff_u), u_floor, umax)
    u2 = clamp(0.5 * (sum_u - diff_u), u_floor, umax)

    # Keep realized total thrust close to the intended sum_u after clamping.
    sum_real = u1 + u2
    sum_target = max(0.85 * sum_u, 2.0 * u_floor)
    if sum_real < sum_target
        boost = 0.5 * (sum_target - sum_real)
        u1 = clamp(u1 + boost, u_floor, umax)
        u2 = clamp(u2 + boost, u_floor, umax)
    end
    return (u1, u2)
end

function select_safe_control(xhat::Vector{Float64}, U::Vector{Vector{Float64}}, Bs::Vector, delta::Float64, K::Float64)
    Bcrit, idx, vals = critical_barrier(Bs, xhat)
    if Bcrit > -delta - K
        return true, (U[idx][1], U[idx][2]), Bcrit, idx, vals
    end

    return false, (0.0, 0.0), Bcrit, idx, vals
end

function select_emergency_control(xk::Vector{Float64}, c_true::Vector{Float64}, U::Vector{Vector{Float64}}, dt::Float64)
    best_u = (U[1][1], U[1][2])
    best_score = -Inf
    for ui in U
        u = (ui[1], ui[2])
        xnext = xk .+ dt .* dynamics_rhs(xk, u)
        score = (xnext[1] - c_true[1])^2 + (xnext[2] - c_true[2])^2
        if score > best_score
            best_score = score
            best_u = u
        end
    end
    return best_u
end

function run_repulsive_hybrid_planar_multirotor_demo(;
    Bs::Vector,
    U::Vector{Vector{Float64}},
    K::Float64=1.0,
    delta::Float64=1.0,
    alpha::Float64=0.1,
    tau::Float64=0.1,
    tau_steps=nothing,
    override_B_threshold=nothing,
    override_B_hysteresis::Float64=0.0,
    override_dist_trigger::Float64=Inf,
    override_dist_release=nothing,
    strict_delta_enforcement::Bool=false,
    strict_delta_hysteresis::Float64=0.0,
    strict_u_track_weight::Float64=0.03,
    make_plots::Bool=true,
    dt::Float64=0.01,
    T::Float64=20.0,
    x0::Vector{Float64}=[-10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ref_style::Symbol=:line,
    ref_start::Tuple{Float64, Float64}=(-10.0, -10.0),
    ref_end::Tuple{Float64, Float64}=(10.0, 10.0),
    ref_T::Float64=T,
    lazy8_center::Tuple{Float64, Float64}=(0.0, 0.0),
    lazy8_scale::Tuple{Float64, Float64}=(8.0, 5.0),
    lazy8_cycles::Float64=1.0,
    lazy8_phase::Float64=0.0,
    state_bounds::Vector{Tuple{Float64, Float64}}=[(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-Float64(pi), Float64(pi)), (-10.0, 10.0), (-1.0, 1.0)],
    v_obs::Tuple{Float64, Float64}=(0.12, 0.06),
    x0_obs::Tuple{Float64, Float64}=(-1.5, 0.0),
    gif_file::String="figures/repulsive_hybrid_planar_multirotor.gif"
)
    obs = make_obstacle_trajectory(v_obs, x0_obs)

    N = Int(floor(T / dt)) + 1
    ts = collect(range(0.0, step=dt, length=N))

    X = zeros(Float64, 7, N)
    X[:, 1] .= x0

    u_nom_hist = zeros(Float64, 2, N)
    u_app_hist = zeros(Float64, 2, N)
    B_hist = zeros(Float64, length(Bs), N)
    Bcrit_hist = zeros(Float64, N)
    barrier_override_hist = falses(N)
    obs_true_hist = zeros(Float64, 2, N)
    obs_hat_hist = zeros(Float64, 2, N)
    ref_hist = zeros(Float64, 2, N)
    track_err_hist = zeros(Float64, N)

    hold_u = nominal_controller(collect(x0), 0.0;
        ref_style=ref_style,
        ref_start=ref_start,
        ref_end=ref_end,
        ref_T=ref_T,
        lazy8_center=lazy8_center,
        lazy8_scale=lazy8_scale,
        lazy8_cycles=lazy8_cycles,
        lazy8_phase=lazy8_phase)
    hold_override = false
    hold_count = 0
    hold_horizon = isnothing(tau_steps) ? max(1, Int(round(tau / dt))) : max(1, Int(tau_steps))
    enter_thresh = isnothing(override_B_threshold) ? -(delta + K) : override_B_threshold
    exit_thresh = enter_thresh - abs(override_B_hysteresis)
    dist_release = isnothing(override_dist_release) ? override_dist_trigger : override_dist_release
    obs_hat = obs(0.0)
    ref_hist[:, 1] .= reference_state(0.0;
        ref_style=ref_style,
        p_start=ref_start,
        p_end=ref_end,
        T_ref=ref_T,
        lazy8_center=lazy8_center,
        lazy8_scale=lazy8_scale,
        lazy8_cycles=lazy8_cycles,
        lazy8_phase=lazy8_phase)[1]

    for k in 1:(N - 1)
        t = ts[k]
        xk = collect(X[:, k])
        c_true = obs(t)

        if hold_count <= 0
            obs_hat = copy(c_true)
        end

        obs_true_hist[:, k] .= c_true
        obs_hat_hist[:, k] .= obs_hat
        xhat = recenter_state(xk, obs_hat)

        for i in eachindex(Bs)
            B_hist[i, k] = eval_barrier(Bs[i], xhat)
        end

        u_nom = nominal_controller(xk, t;
            ref_style=ref_style,
            ref_start=ref_start,
            ref_end=ref_end,
            ref_T=ref_T,
            lazy8_center=lazy8_center,
            lazy8_scale=lazy8_scale,
            lazy8_cycles=lazy8_cycles,
            lazy8_phase=lazy8_phase,
            dt_nom=max(4.0 * dt, 0.08))
        u_nom_hist[:, k] .= [u_nom[1], u_nom[2]]

        if hold_count <= 0
            if !hold_override
                hold_u = u_nom
            end

            Bcrit, idx_crit, _ = critical_barrier(Bs, xhat)
            d_hat = hypot(xk[1] - obs_hat[1], xk[2] - obs_hat[2])

            if strict_delta_enforcement
                strict_exit = -delta - abs(strict_delta_hysteresis)
                if Bcrit > -delta
                    hold_override = true
                    hold_u, _ = choose_best_override_control(xk, obs_hat, U, Bs, dt;
                        u_nom=u_nom,
                        u_track_weight=strict_u_track_weight)
                elseif hold_override && (Bcrit <= strict_exit)
                    hold_override = false
                    hold_u = (0.0, 0.0)
                end
            else
                if hold_override
                    if (Bcrit <= exit_thresh) || (d_hat >= dist_release)
                        hold_override = false
                        hold_u = u_nom
                    else
                        hold_u, _ = choose_best_override_control(xk, obs_hat, U, Bs, dt;
                            u_nom=u_nom,
                            u_track_weight=0.03)
                    end
                elseif (d_hat <= override_dist_trigger) && (Bcrit > enter_thresh)
                    hold_override = true
                    hold_u = (U[idx_crit][1], U[idx_crit][2])
                else
                    hold_u = u_nom
                end
            end
            Bcrit_hist[k] = Bcrit
            hold_count = hold_horizon
        else
            Bcrit, _, _ = critical_barrier(Bs, xhat)
            Bcrit_hist[k] = Bcrit
        end

        hold_count -= 1

        barrier_override_hist[k] = hold_override
        u_app_hist[:, k] .= [hold_u[1], hold_u[2]]

        dx = dynamics_rhs(xk, hold_u)
        X[:, k + 1] .= xk .+ dt .* dx

        # Keep integrated state within the synthesis domain assumptions.
        X[3, k + 1] = clamp(X[3, k + 1], state_bounds[3][1], state_bounds[3][2])
        X[4, k + 1] = clamp(X[4, k + 1], state_bounds[4][1], state_bounds[4][2])
        X[5, k + 1] = atan(sin(X[5, k + 1]), cos(X[5, k + 1]))
        X[6, k + 1] = clamp(X[6, k + 1], state_bounds[6][1], state_bounds[6][2])
        X[7, k + 1] = clamp(X[7, k + 1], state_bounds[7][1], state_bounds[7][2])

        ref_hist[:, k + 1] .= reference_state(t + dt;
            ref_style=ref_style,
            p_start=ref_start,
            p_end=ref_end,
            T_ref=ref_T,
            lazy8_center=lazy8_center,
            lazy8_scale=lazy8_scale,
            lazy8_cycles=lazy8_cycles,
            lazy8_phase=lazy8_phase)[1]
        track_err_hist[k] = hypot(X[1, k] - ref_hist[1, k], X[2, k] - ref_hist[2, k])
    end

    xlast = collect(X[:, end])
    clast = obs(ts[end])
    obs_true_hist[:, end] .= clast
    obs_hat_hist[:, end] .= obs_hat
    xhat_last = recenter_state(xlast, obs_hat)
    for i in eachindex(Bs)
        B_hist[i, end] = eval_barrier(Bs[i], xhat_last)
    end
    Bcrit_hist[end] = minimum(B_hist[:, end])
    u_nom_hist[:, end] .= u_nom_hist[:, end - 1]
    u_app_hist[:, end] .= u_app_hist[:, end - 1]
    track_err_hist[end] = hypot(X[1, end] - ref_hist[1, end], X[2, end] - ref_hist[2, end])

    obs_x = obs_true_hist[1, :]
    obs_y = obs_true_hist[2, :]
    obs_hat_x = obs_hat_hist[1, :]
    obs_hat_y = obs_hat_hist[2, :]
    dist_to_obs = sqrt.((X[1, :] .- obs_x).^2 .+ (X[2, :] .- obs_y).^2)
    min_dist_to_obs = minimum(dist_to_obs)
    collision = any(dist_to_obs .<= alpha)
    n_override = count(barrier_override_hist)
    n_emergency_override = 0

    function _override_spans(flags, ts_local, dt_local)
        spans = Tuple{Float64, Float64}[]
        in_span = false
        t0 = 0.0
        for i in eachindex(flags)
            if flags[i] && !in_span
                in_span = true
                t0 = ts_local[i]
            elseif !flags[i] && in_span
                push!(spans, (t0, ts_local[i]))
                in_span = false
            end
        end
        if in_span
            push!(spans, (t0, ts_local[end] + dt_local))
        end
        return spans
    end

    override_spans = _override_spans(barrier_override_hist, ts, dt)
    tau_steps = max(1, Int(round(tau / dt)))
    sample_times = ts[1:tau_steps:end]
    obs_snap_idx = unique(round.(Int, range(1, length(ts), length=min(5, length(ts)))))
    theta_circle = LinRange(0, 2pi, 200)

    p_traj = nothing
    p_B = nothing
    p_u = nothing
    p_combined = nothing
    p_anim = nothing

    if make_plots
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
            title="Workspace trajectory",
            legend=:bottomright,
        )
        plot!(p_traj, ref_hist[1, :], ref_hist[2, :], lw=2, ls=:dash, color=:gray45, label="reference")
        plot!(p_traj, X[1, :], X[2, :], lw=2.6, color=:dodgerblue3, label="filtered trajectory")
        plot!(p_traj, obs_x, obs_y, lw=1.8, color=:forestgreen, alpha=0.8, label="obstacle center path")
        plot!(p_traj, obs_hat_x, obs_hat_y, lw=1.3, ls=:dashdot, color=:darkmagenta, alpha=0.8, label="recentered path")
        scatter!(p_traj, [X[1, 1]], [X[2, 1]], marker=:circle, ms=4, color=:dodgerblue3, label="start")
        scatter!(p_traj, [X[1, end]], [X[2, end]], marker=:star5, ms=7, color=:dodgerblue4, label="end")

        for (j, idx) in enumerate(obs_snap_idx)
            cx, cy = obs_x[idx], obs_y[idx]
            circx = cx .+ alpha .* cos.(theta_circle)
            circy = cy .+ alpha .* sin.(theta_circle)
            plot!(p_traj, circx, circy, color=:orange2, ls=:dash, lw=1.4,
                alpha=(j == length(obs_snap_idx) ? 0.9 : 0.5),
                label=(j == 1 ? "obstacle snapshots" : ""))
            scatter!(p_traj, [cx], [cy], color=:forestgreen, ms=3,
                alpha=(j == length(obs_snap_idx) ? 0.9 : 0.55),
                label=(j == 1 ? "obstacle center" : ""))
        end

        p_B = plot(ts, Bcrit_hist, lw=2.2, color=:dodgerblue3,
            label="active barrier", xlabel="time", ylabel="B", title="Barrier margin", legend=:bottomright)
        hline!(p_B, [0.0], color=:orangered2, ls=:dash, lw=1.8, label="0")
        hline!(p_B, [-delta], color=:forestgreen, ls=:dot, lw=1.8, label="-δ")
        hline!(p_B, [-(delta + K)], color=:darkorchid2, ls=:dashdot, lw=1.8, label="-δ-K")

        # Replace dense guide lines with compact bottom markers to reduce visual clutter.
        b_min = minimum(Bcrit_hist)
        b_max = maximum(Bcrit_hist)
        b_span = max(b_max - b_min, 1e-3)
        marker_base = b_min - 0.04 * b_span
        marker_upper = b_min - 0.015 * b_span

        n_sample_markers = min(length(sample_times), 36)
        sample_stride = max(1, Int(ceil(length(sample_times) / max(n_sample_markers, 1))))
        sample_marker_times = sample_times[1:sample_stride:end]
        scatter!(p_B, sample_marker_times, fill(marker_base, length(sample_marker_times));
            ms=1.6, marker=:circle, color=:gray55, alpha=0.35, label=false)

        override_times = ts[barrier_override_hist]
        if !isempty(override_times)
            n_override_markers = min(length(override_times), 60)
            override_stride = max(1, Int(ceil(length(override_times) / max(n_override_markers, 1))))
            override_marker_times = override_times[1:override_stride:end]
            scatter!(p_B, override_marker_times, fill(marker_upper, length(override_marker_times));
                ms=2.0, marker=:utriangle, color=:darkorange2, alpha=0.60, markerstrokewidth=0.0, label="override markers")
        end

        u1_nominal_only = [barrier_override_hist[i] ? NaN : u_app_hist[1, i] for i in eachindex(ts)]
        u1_override_only = [barrier_override_hist[i] ? u_app_hist[1, i] : NaN for i in eachindex(ts)]
        u2_nominal_only = [barrier_override_hist[i] ? NaN : u_app_hist[2, i] for i in eachindex(ts)]
        u2_override_only = [barrier_override_hist[i] ? u_app_hist[2, i] : NaN for i in eachindex(ts)]

        p_u = plot(
            ts,
            u1_nominal_only,
            lw=2.0,
            color=:deepskyblue3,
            label="u1 applied (nominal mode)",
            xlabel="time",
            ylabel="control",
            title="Applied controls by mode",
            legend=:bottomright,
        )
        plot!(p_u, ts, u1_override_only, lw=2.0, color=:orangered3, label="u1 applied (override mode)")
        plot!(p_u, ts, u2_nominal_only, lw=1.8, color=:teal, ls=:dash, label="u2 applied (nominal mode)")
        plot!(p_u, ts, u2_override_only, lw=1.8, color=:orange, ls=:dash, label="u2 applied (override mode)")
        for (a, b) in override_spans
            vspan!(p_u, [a, b], color=:gray85, alpha=0.35, label=false)
        end

        p_combined = plot(
            p_traj, p_B, p_u,
            layout=@layout([a{0.95w} ; b{0.6w} ; c{0.55w}]),
            size=(900, 1100),
        )

        mkpath("figures")
        savefig(p_traj, "figures/multirotor_traj.pdf")
        savefig(p_B, "figures/multirotor_barrier.pdf")
        savefig(p_u, "figures/multirotor_control.pdf")
        savefig(p_combined, "figures/multirotor_results_combined.pdf")

        x_all = vcat(X[1, :], obs_x)
        y_all = vcat(X[2, :], obs_y)
        x_margin = 0.15 * max(maximum(x_all) - minimum(x_all), 1.0)
        y_margin = 0.15 * max(maximum(y_all) - minimum(y_all), 1.0)
        xlims_anim = (minimum(x_all) - x_margin, maximum(x_all) + x_margin)
        ylims_anim = (minimum(y_all) - y_margin, maximum(y_all) + y_margin)

        obs_radius = alpha
        θ = LinRange(0, 2pi, 160)

        anim = @animate for k in 1:4:N
            p = plot(xlim=xlims_anim, ylim=ylims_anim, aspect_ratio=:equal, xlabel="x", ylabel="y", title="Repulsive barrier simulation (moving obstacle)")
            plot!(p, ref_hist[1, :], ref_hist[2, :], lw=2, ls=:dash, color=:gray, label="reference")
            plot!(p, X[1, 1:k], X[2, 1:k], label="system", lw=2.5, color=:blue)
            if k > 1 && barrier_override_hist[min(k - 1, end)]
                scatter!(p, [X[1, k]], [X[2, k]], label="barrier override", color=:red, markersize=5)
            else
                scatter!(p, [X[1, k]], [X[2, k]], label="state", color=:blue, markersize=5)
            end

            cx, cy = obs_x[k], obs_y[k]
            hx, hy = obs_hat_x[k], obs_hat_y[k]
            obs_px = cx .+ obs_radius .* cos.(θ)
            obs_py = cy .+ obs_radius .* sin.(θ)
            plot!(p, obs_px, obs_py, label="obstacle", lw=2, color=:black)
            scatter!(p, [cx], [cy], label="obs center", color=:black, markersize=4)
            scatter!(p, [hx], [hy], label="recentered", marker=:x, color=:magenta, markersize=5)
            annotate!(p, xlims_anim[1] + 0.05 * (xlims_anim[2] - xlims_anim[1]), ylims_anim[2] - 0.06 * (ylims_anim[2] - ylims_anim[1]), text("t = $(round(ts[k], digits=2)) s", 9))
            annotate!(p, xlims_anim[1] + 0.05 * (xlims_anim[2] - xlims_anim[1]), ylims_anim[2] - 0.12 * (ylims_anim[2] - ylims_anim[1]), text("B = $(round(Bcrit_hist[min(max(k - 1, 1), end)], digits=3))", 9))
            mode_str = (k > 1 && barrier_override_hist[min(k - 1, end)]) ? "override" : "nominal"
            annotate!(p, xlims_anim[1] + 0.05 * (xlims_anim[2] - xlims_anim[1]), ylims_anim[2] - 0.18 * (ylims_anim[2] - ylims_anim[1]), text("mode = $mode_str", 9))
            p
        end
        mkpath(dirname(gif_file))
        gif(anim, gif_file, fps=20)
    end

    println("Simulation finished")
    println("minimum recentered barrier value = ", minimum(Bcrit_hist))
    println("minimum true obstacle distance = ", min_dist_to_obs)
    println("mean tracking error = ", mean(track_err_hist))
    println("max tracking error = ", maximum(track_err_hist))
    println("number of barrier overrides = ", n_override)
    println("override enter threshold = ", enter_thresh)
    println("override exit threshold = ", exit_thresh)
    println("override distance trigger = ", override_dist_trigger)
    println("override distance release = ", dist_release)
    println("strict delta enforcement = ", strict_delta_enforcement)
    println("Animation saved to ", gif_file)

    if make_plots
        p_anim = plot(X[1, :], X[2, :], lw=2.5, label="trajectory preview", aspect_ratio=1)
        plot!(p_anim, ref_hist[1, :], ref_hist[2, :], lw=2, ls=:dash, label="reference")
        plot!(p_anim, title="Preview (full animation is in GIF)", xlabel="x", ylabel="y")
    end

    return (
        ts = ts,
        X = X,
        B_hist = B_hist,
        B_crit = Bcrit_hist,
        u_nom_hist = u_nom_hist,
        u_app_hist = u_app_hist,
        takeover_hist = barrier_override_hist,
        emergency_hist = falses(length(barrier_override_hist)),
        n_override = n_override,
        n_emergency_override = n_emergency_override,
        obs_true_hist = obs_true_hist,
        obs_hat_hist = obs_hat_hist,
        ref_hist = ref_hist,
        min_dist_to_obs = min_dist_to_obs,
        mean_track_err = mean(track_err_hist),
        max_track_err = maximum(track_err_hist),
        collision = collision,
        p_traj = p_traj,
        p_B = p_B,
        p_u = p_u,
        anim = p_anim,
        gif_path = gif_file,
    )
end

end # module

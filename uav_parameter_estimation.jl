using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using Optimization, OptimizationOptimJL, SciMLSensitivity
using Random
using Plots

# ------------------------------------------------------------
# Upgraded proof-of-concept:
# Coupled two-state longitudinal UAV-style model
#
# States:
#   v(t)      = forward speed (m/s)
#   theta(t)  = pitch angle (rad)
#
# Simplified dynamics:
#   dv/dt     = T_eff - Cd_eff*v^2 - g*theta
#   dtheta/dt = -k_theta*theta + k_v*(v - v_trim)
#
# Parameters to estimate:
#   T_eff   = effective thrust-to-mass term
#   Cd_eff  = effective quadratic drag term
#   k_theta = pitch restoring term
#   k_v     = simplified speed-to-pitch coupling term
# ------------------------------------------------------------

@parameters T_eff Cd_eff k_theta k_v
@variables v(t) theta(t)

g = 9.81
v_trim = 16.0

eqs = [
    D(v) ~ T_eff - Cd_eff * v^2 - g * theta
    D(theta) ~ -k_theta * theta + k_v * (v - v_trim)
]

@named uav_longitudinal_model = ODESystem(eqs, t)
sys = structural_simplify(uav_longitudinal_model)

# "True" parameters used to generate synthetic flight-style data
T_true = 8.0
Cd_true = 0.03
k_theta_true = 0.9
k_v_true = 0.02

u0 = [
    v => 14.0,
    theta => 0.06
]

p_true = [
    T_eff => T_true,
    Cd_eff => Cd_true,
    k_theta => k_theta_true,
    k_v => k_v_true
]

tspan = (0.0, 25.0)
save_times = 0.0:0.5:25.0

# Generate synthetic truth solution
prob_true = ODEProblem(sys, u0, tspan, p_true)
sol_true = solve(prob_true, Tsit5(); saveat = save_times)

v_clean = Array(sol_true[v])
theta_clean = Array(sol_true[theta])

# Create noisy "measured" flight-style data
Random.seed!(42)
sigma_v = 0.12
sigma_theta = 0.003

v_data = v_clean .+ sigma_v .* randn(length(v_clean))
theta_data = theta_clean .+ sigma_theta .* randn(length(theta_clean))

# Initial guess for estimation
p_guess = [6.5, 0.04, 0.6, 0.01]

function loss(p, _)
    T_val, Cd_val, k_theta_val, k_v_val = p

    if T_val <= 0 || Cd_val <= 0 || k_theta_val <= 0 || k_v_val < 0
        return 1.0e12
    end

    p_map = [
        T_eff => T_val,
        Cd_eff => Cd_val,
        k_theta => k_theta_val,
        k_v => k_v_val
    ]

    try
        prob = ODEProblem(sys, u0, tspan, p_map)
        sol = solve(prob, Tsit5(); saveat = save_times, abstol = 1e-8, reltol = 1e-8)

        v_pred = Array(sol[v])
        theta_pred = Array(sol[theta])

        # Weighted loss so velocity and pitch angle both matter appropriately
        loss_v = sum(((v_pred .- v_data) ./ sigma_v).^2) / length(v_data)
        loss_theta = sum(((theta_pred .- theta_data) ./ sigma_theta).^2) / length(theta_data)

        return loss_v + loss_theta
    catch
        return 1.0e12
    end
end

optf = OptimizationFunction(loss, Optimization.AutoForwardDiff())
optprob = OptimizationProblem(optf, p_guess)

# Two-stage optimisation for a more robust fit
result_nm = solve(optprob, NelderMead(); maxiters = 2000)
optprob_refined = OptimizationProblem(optf, result_nm.u)
result = solve(optprob_refined, BFGS(); maxiters = 2000)

T_est = result.u[1]
Cd_est = result.u[2]
k_theta_est = result.u[3]
k_v_est = result.u[4]

p_est = [
    T_eff => T_est,
    Cd_eff => Cd_est,
    k_theta => k_theta_est,
    k_v => k_v_est
]

prob_est = ODEProblem(sys, u0, tspan, p_est)
sol_est = solve(prob_est, Tsit5(); saveat = save_times)

v_est = Array(sol_est[v])
theta_est = Array(sol_est[theta])

println("----- Parameter estimation results -----")
println("True T_eff      = ", T_true)
println("Estimated T_eff = ", round(T_est, digits = 6))
println("True Cd_eff      = ", Cd_true)
println("Estimated Cd_eff = ", round(Cd_est, digits = 6))
println("True k_theta      = ", k_theta_true)
println("Estimated k_theta = ", round(k_theta_est, digits = 6))
println("True k_v      = ", k_v_true)
println("Estimated k_v = ", round(k_v_est, digits = 6))
println("Final loss     = ", round(loss(result.u, nothing), digits = 8))

# Convert pitch angle to degrees for clearer plotting
theta_clean_deg = theta_clean .* (180 / pi)
theta_data_deg = theta_data .* (180 / pi)
theta_est_deg = theta_est .* (180 / pi)

p1 = plot(
    save_times,
    v_clean,
    label = "True velocity",
    linewidth = 2,
    xlabel = "Time (s)",
    ylabel = "Velocity (m/s)",
    title = "Longitudinal UAV velocity fit"
)
scatter!(p1, save_times, v_data, label = "Noisy data", markersize = 3)
plot!(p1, save_times, v_est, label = "Fitted model", linewidth = 2)
savefig(p1, "longitudinal_velocity_fit.png")

p2 = plot(
    save_times,
    theta_clean_deg,
    label = "True pitch angle",
    linewidth = 2,
    xlabel = "Time (s)",
    ylabel = "Pitch angle (deg)",
    title = "Longitudinal UAV pitch-angle fit"
)
scatter!(p2, save_times, theta_data_deg, label = "Noisy data", markersize = 3)
plot!(p2, save_times, theta_est_deg, label = "Fitted model", linewidth = 2)
savefig(p2, "longitudinal_pitch_angle_fit.png")

println("Saved figure: longitudinal_velocity_fit.png")
println("Saved figure: longitudinal_pitch_angle_fit.png")
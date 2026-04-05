using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using Optimization, OptimizationOptimJL, SciMLSensitivity
using Random
using Statistics
using Plots

# ------------------------------------------------------------
# Upgrade: multi-trajectory parameter estimation + validation
#
# States:
#   v(t)      = forward speed (m/s)
#   theta(t)  = pitch angle (rad)
#
# Simplified longitudinal model:
#   dv/dt     = T_eff - Cd_eff*v^2 - g*theta
#   dtheta/dt = -k_theta*theta + k_v*(v - v_trim)
#
# Parameters to estimate:
#   T_eff, Cd_eff, k_theta, k_v
#
# Training:
#   Fit one common parameter set using two noisy trajectories
#
# Validation:
#   Test the fitted parameters on a third unseen trajectory
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

# True parameters used to generate synthetic flight-style data
T_true = 8.0
Cd_true = 0.03
k_theta_true = 0.9
k_v_true = 0.02

p_true = [
    T_eff => T_true,
    Cd_eff => Cd_true,
    k_theta => k_theta_true,
    k_v => k_v_true
]

tspan = (0.0, 25.0)
save_times = 0.0:0.5:25.0

# Measurement noise levels
sigma_v = 0.12
sigma_theta = 0.003

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

function simulate_states(u0_map, p_map)
    vals = merge(Dict(u0_map), Dict(p_map))
    prob = ODEProblem(sys, vals, tspan)
    sol = solve(prob, Tsit5(); saveat = save_times, abstol = 1e-8, reltol = 1e-8)
    return Array(sol[v]), Array(sol[theta])
end

function build_dataset(u0_map; rng = Random.default_rng())
    v_true, theta_true = simulate_states(u0_map, p_true)
    v_data = v_true .+ sigma_v .* randn(rng, length(v_true))
    theta_data = theta_true .+ sigma_theta .* randn(rng, length(theta_true))

    return (
        u0 = u0_map,
        v_true = v_true,
        theta_true = theta_true,
        v_data = v_data,
        theta_data = theta_data
    )
end

rmse(x, y) = sqrt(mean((x .- y).^2))

# ------------------------------------------------------------
# Build multiple synthetic trajectories
# ------------------------------------------------------------

Random.seed!(42)

train_u0s = [
    [v => 14.0, theta => 0.06],
    [v => 18.0, theta => -0.03]
]

val_u0 = [v => 15.5, theta => 0.025]

train_sets = [build_dataset(u0) for u0 in train_u0s]
val_set = build_dataset(val_u0)

# ------------------------------------------------------------
# Loss: joint fit across both training trajectories
# ------------------------------------------------------------

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

    total_loss = 0.0

    try
        for ds in train_sets
            v_pred, theta_pred = simulate_states(ds.u0, p_map)

            loss_v = sum(((v_pred .- ds.v_data) ./ sigma_v).^2) / length(v_pred)
            loss_theta = sum(((theta_pred .- ds.theta_data) ./ sigma_theta).^2) / length(theta_pred)

            total_loss += loss_v + loss_theta
        end

        return total_loss / length(train_sets)
    catch
        return 1.0e12
    end
end

# ------------------------------------------------------------
# Optimisation
# ------------------------------------------------------------

p_guess = [6.5, 0.04, 0.6, 0.01]

optf = OptimizationFunction(loss, Optimization.AutoForwardDiff())
optprob = OptimizationProblem(optf, p_guess)

result_nm = solve(optprob, NelderMead(); maxiters = 2000)
optprob_refined = OptimizationProblem(optf, result_nm.u)
result = solve(optprob_refined, BFGS(); maxiters = 2000)

T_est, Cd_est, k_theta_est, k_v_est = result.u

p_est = [
    T_eff => T_est,
    Cd_eff => Cd_est,
    k_theta => k_theta_est,
    k_v => k_v_est
]

# ------------------------------------------------------------
# Evaluate fitted model on training and validation data
# ------------------------------------------------------------

train_preds = map(train_sets) do ds
    v_pred, theta_pred = simulate_states(ds.u0, p_est)
    (
        v_pred = v_pred,
        theta_pred = theta_pred
    )
end

val_v_pred, val_theta_pred = simulate_states(val_set.u0, p_est)

# Metrics
train_v_rmse = [rmse(pred.v_pred, ds.v_data) for (pred, ds) in zip(train_preds, train_sets)]
train_theta_rmse = [rmse(pred.theta_pred, ds.theta_data) for (pred, ds) in zip(train_preds, train_sets)]

val_v_rmse = rmse(val_v_pred, val_set.v_data)
val_theta_rmse = rmse(val_theta_pred, val_set.theta_data)

println("----- Estimated parameters -----")
println("True T_eff        = ", T_true)
println("Estimated T_eff   = ", round(T_est, digits = 6))
println("True Cd_eff       = ", Cd_true)
println("Estimated Cd_eff  = ", round(Cd_est, digits = 6))
println("True k_theta      = ", k_theta_true)
println("Estimated k_theta = ", round(k_theta_est, digits = 6))
println("True k_v          = ", k_v_true)
println("Estimated k_v     = ", round(k_v_est, digits = 6))
println("Final loss        = ", round(loss(result.u, nothing), digits = 8))

println("\n----- Training RMSE -----")
for i in eachindex(train_sets)
    println("Run ", i, " velocity RMSE (m/s): ", round(train_v_rmse[i], digits = 6))
    println("Run ", i, " pitch RMSE (rad):    ", round(train_theta_rmse[i], digits = 6))
end

println("\n----- Validation RMSE -----")
println("Validation velocity RMSE (m/s): ", round(val_v_rmse, digits = 6))
println("Validation pitch RMSE (rad):    ", round(val_theta_rmse, digits = 6))

# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

theta_to_deg(x) = x .* (180 / pi)

# Training velocity plot
p_train_v = plot(
    title = "Training trajectories: velocity fit",
    xlabel = "Time (s)",
    ylabel = "Velocity (m/s)",
    legend = :best
)

plot!(p_train_v, save_times, train_sets[1].v_true, label = "Run 1 true", linewidth = 2)
scatter!(p_train_v, save_times, train_sets[1].v_data, label = "Run 1 data", markersize = 3)
plot!(p_train_v, save_times, train_preds[1].v_pred, label = "Run 1 fitted", linewidth = 2)

plot!(p_train_v, save_times, train_sets[2].v_true, label = "Run 2 true", linewidth = 2, linestyle = :dash)
scatter!(p_train_v, save_times, train_sets[2].v_data, label = "Run 2 data", markersize = 3)
plot!(p_train_v, save_times, train_preds[2].v_pred, label = "Run 2 fitted", linewidth = 2, linestyle = :dash)

# Training pitch plot
p_train_theta = plot(
    title = "Training trajectories: pitch-angle fit",
    xlabel = "Time (s)",
    ylabel = "Pitch angle (deg)",
    legend = :best
)

plot!(p_train_theta, save_times, theta_to_deg(train_sets[1].theta_true), label = "Run 1 true", linewidth = 2)
scatter!(p_train_theta, save_times, theta_to_deg(train_sets[1].theta_data), label = "Run 1 data", markersize = 3)
plot!(p_train_theta, save_times, theta_to_deg(train_preds[1].theta_pred), label = "Run 1 fitted", linewidth = 2)

plot!(p_train_theta, save_times, theta_to_deg(train_sets[2].theta_true), label = "Run 2 true", linewidth = 2, linestyle = :dash)
scatter!(p_train_theta, save_times, theta_to_deg(train_sets[2].theta_data), label = "Run 2 data", markersize = 3)
plot!(p_train_theta, save_times, theta_to_deg(train_preds[2].theta_pred), label = "Run 2 fitted", linewidth = 2, linestyle = :dash)

training_fig = plot(p_train_v, p_train_theta; layout = (1, 2), size = (1300, 450))
savefig(training_fig, "multi_trajectory_training_fit.png")

# Validation velocity plot
p_val_v = plot(
    save_times,
    val_set.v_true,
    label = "Validation true",
    linewidth = 2,
    title = "Validation trajectory: velocity fit",
    xlabel = "Time (s)",
    ylabel = "Velocity (m/s)",
    legend = :best
)
scatter!(p_val_v, save_times, val_set.v_data, label = "Validation data", markersize = 3)
plot!(p_val_v, save_times, val_v_pred, label = "Validation fitted", linewidth = 2)

# Validation pitch plot
p_val_theta = plot(
    save_times,
    theta_to_deg(val_set.theta_true),
    label = "Validation true",
    linewidth = 2,
    title = "Validation trajectory: pitch-angle fit",
    xlabel = "Time (s)",
    ylabel = "Pitch angle (deg)",
    legend = :best
)
scatter!(p_val_theta, save_times, theta_to_deg(val_set.theta_data), label = "Validation data", markersize = 3)
plot!(p_val_theta, save_times, theta_to_deg(val_theta_pred), label = "Validation fitted", linewidth = 2)

validation_fig = plot(p_val_v, p_val_theta; layout = (1, 2), size = (1300, 450))
savefig(validation_fig, "multi_trajectory_validation_fit.png")

println("\nSaved figure: multi_trajectory_training_fit.png")
println("Saved figure: multi_trajectory_validation_fit.png")
# UAV Parameter Estimation Proof of Concept

This is a small Julia-based proof of concept built to explore the type of modelling and parameter-estimation workflow used in data-driven UAV dynamics projects.

The model was created using ModelingToolkit and the SciML ecosystem. A simplified two-state longitudinal UAV-style model was defined, synthetic flight-style data was generated, and unknown parameters were then estimated by fitting a common parameter set across multiple training trajectories and testing that fit on a separate validation trajectory.

## Files

- `uav_parameter_estimation.jl` — main Julia script
- `multi_trajectory_training_fit.png` — training trajectory fit result
- `multi_trajectory_validation_fit.png` — validation trajectory fit result
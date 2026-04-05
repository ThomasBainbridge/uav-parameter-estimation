# UAV Parameter Estimation Proof of Concept

This is a small Julia-based proof of concept built to explore the type of modelling and parameter-estimation workflow used in data-driven UAV dynamics projects.

The model was created using ModelingToolkit and the SciML ecosystem. A simplified two-state longitudinal UAV-style model was defined, synthetic flight-style data was generated, and unknown parameters were then estimated by fitting the model back to that data.

## Files

- `uav_parameter_estimation.jl` — main Julia script
- `longitudinal_velocity_fit.png` — velocity fit result
- `longitudinal_pitch_angle_fit.png` — pitch-angle fit result
clear; close all; clc;

%% Copy reference generation từ code gốc
tau = 0.3; lr1 = 2.9; lt1 = 1.8; m0 = 0.8;
dt = 0.03; umax = 0.52; max_rate = 1.2; Np = 50; regularization = 1e-6;umin=-umax;
timeSpan = 20; timeStep = 0.005;
t_ref = (0:timeStep:timeSpan)';
x_a_r = t_ref * 2 * 20 / timeSpan;
Rparam = 20; alpha_l = 12; c = 10;
y_a_r = c * (exp(-log(2) * (x_a_r./Rparam).^alpha_l) - 1);
dx = gradient(x_a_r, t_ref); ddx = gradient(dx, t_ref);
dy = gradient(y_a_r, t_ref); ddy = gradient(dy, t_ref);
curvature = (dx .* ddy - dy .* ddx) ./ ((dx.^2 + dy.^2).^(3/2) + eps);
v_ref = sqrt(dx.^2 + dy.^2);
psi_ref = atan2(dy, dx);
delta_ref = atan(lr1 .* curvature);
t_ctrl = (0:dt:timeSpan)';
x_ref = interp1(t_ref, x_a_r, t_ctrl, 'linear')';
y_ref = interp1(t_ref, y_a_r, t_ctrl, 'linear')';
psi_ref_i = interp1(t_ref, psi_ref, t_ctrl, 'linear')';
delta_r_profile = interp1(t_ref, delta_ref, t_ctrl, 'linear')';
k_profile = interp1(t_ref, curvature, t_ctrl, 'linear')';
v_profile = interp1(t_ref, v_ref, t_ctrl, 'linear')';

%% Check requirements
if ~exist('bayesopt','file')
    error('Bayesian Optimization requires Statistics and Machine Learning Toolbox.');
end
if ~exist('quadprog','file')
    error('quadprog not found. Install Optimization Toolbox.');
end
if ~exist('mpc_loop','file')
    error('mpc_loop.m not found. Ensure it is in the MATLAB path.');
end

%% Define variables for BO
vars = [
    optimizableVariable('log_q_y', [log10(0.1), log10(10)], 'Type', 'real');
    optimizableVariable('log_q_psi', [log10(0.1), log10(10)], 'Type', 'real');
    optimizableVariable('log_q_psi_t', [log10(0.1), log10(10)], 'Type', 'real');
    optimizableVariable('log_ru', [log10(0.01), log10(1)], 'Type', 'real')
];

%% Objective function
objFcn = @(params) objective(params, dt, umax, max_rate, Np, regularization, ...
    x_ref, y_ref, psi_ref_i, delta_r_profile, k_profile, v_profile, m0, lr1, lt1, tau);

%% Run BO
results = bayesopt(objFcn, vars, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', 30, ...
    'NumSeedPoints', 10, ...
    'PlotFcn', @plotObjectiveModel, ...
    'Verbose', 1);

%% Display results
best_params = results.XAtMinObjective;
disp('Best parameters:');
disp(['q_y = ', num2str(10^best_params.log_q_y)]);
disp(['q_psi = ', num2str(10^best_params.log_q_psi)]);
disp(['q_psi_t = ', num2str(10^best_params.log_q_psi_t)]);
disp(['Ru = ', num2str(10^best_params.log_ru)]);
disp(['Min RMSE = ', num2str(results.MinObjective)]);

%% Verify with best parameters
Qx_opt = diag([10^best_params.log_q_y, 10^best_params.log_q_psi, 10^best_params.log_q_psi_t]);
Ru_opt = 10^best_params.log_ru;
[~, history_err, u_hist, x_act, y_act, x_t_act, y_t_act, psi_act, psi_t_act, delta_act] = ...
    mpc_loop(Qx_opt, Ru_opt, dt, umax, max_rate, Np, regularization, ...
    x_ref, y_ref, psi_ref_i, delta_r_profile, k_profile, v_profile, m0, lr1, lt1, tau);

%% Animation
figure; hold on; axis equal; grid on; title('Tractor and trailer tracking animation'); xlabel('x [m]'); ylabel('y [m]');
for k = 1:length(x_act)
    plot(x_act(1:k), y_act(1:k), 'b-', x_t_act(1:k), y_t_act(1:k), 'r-', 'LineWidth', 1.5);
    plot(x_ref, y_ref, 'k--', 'LineWidth', 1.5);
    legend('tractor','trailer','ref');
    drawnow; pause(0.01);
end

%% Plot results
figure('Units','normalized','Position',[0.05 0.05 0.9 0.8]);
subplot(2,2,1);
plot(x_ref, y_ref, 'k--','LineWidth',1.5); hold on;
plot(x_act,y_act,'b-','LineWidth',1.2); hold on;
plot(x_t_act, y_t_act, 'r-','LineWidth',1.5);
title('Reference and Actual Trajectory'); xlabel('x [m]'); ylabel('y [m]'); axis equal; grid on; legend('ref','tractor','trailer');

subplot(2,2,2);
plot(t_ctrl, history_err(1,:), 'LineWidth',1.5);
title('Tractor lateral error [m]'); xlabel('t [s]'); ylabel('y [m]'); grid on; legend('e_{y}');

subplot(2,2,3);
plot(t_ctrl, history_err(2,:), 'LineWidth',1.5); hold on;
plot(t_ctrl, history_err(3,:),'--','LineWidth',1.5);
title('Tractor and trailer yaw error [rad]'); xlabel('t [s]'); ylabel('rad'); grid on; legend('e_\psi','e_{\psi_t}');

subplot(2,2,4);
plot(t_ctrl(1:end-1), u_hist, 'LineWidth',1.5); hold on;
plot(t_ctrl(1:end-1), umax*ones(size(t_ctrl(1:end-1))), 'r--', 'LineWidth', 1.5);
plot(t_ctrl(1:end-1), umin*ones(size(t_ctrl(1:end-1))), 'r--', 'LineWidth', 1.5);
title('Steering control command [rad]'); xlabel('t [s]'); ylabel('rad'); legend('u_{cmd}','u_{max}','u_{min}'); grid on;

%% objective function
function total_rmse = objective(params, dt, umax, max_rate, Np, regularization, ...
    x_ref, y_ref, psi_ref_i, delta_r_profile, k_profile, v_profile, m0, lr1, lt1, tau)
    q_y = 10^params.log_q_y;
    q_psi = 10^params.log_q_psi;
    q_psi_t = 10^params.log_q_psi_t;
    ru = 10^params.log_ru;
    Qx_tune = diag([q_y, q_psi, q_psi_t]);
    
    [total_rmse, ~, ~, ~, ~, ~, ~, ~, ~, ~] = mpc_loop(Qx_tune, ru, dt, umax, max_rate, Np, regularization, ...
        x_ref, y_ref, psi_ref_i, delta_r_profile, k_profile, v_profile, m0, lr1, lt1, tau);
end
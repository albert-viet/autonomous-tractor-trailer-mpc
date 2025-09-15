% lqr_tracker.m
% Trajectory tracking for tractor-trailer using your nonlinear model + analytic linearSys and LQR

clear; close all; clc;

%% --- model parameters (use your values) ---
tau = 0.3; lr1 = 2.9; lt1 = 1.8; m0 = 0.8;

%% simulation / controller sampling
dt = 0.03;               % controller sample time (s)
umax = 0.52; umin = -umax;

%% LQR weights
Q = diag([10, 1, 5, 1]);  % weight on [e_y, e_psi, e_psi_t, delta]
R_lqr = 1;

%% Reference trajectory parameters (PDR sigmoidal)
timeSpan = 9;        % total time of reference (s)
timeStep = 0.005;    % high-res ref sampling
t_ref = (0:timeStep:timeSpan)';

% path in x (parameterized by time)
x_a_r = t_ref * 2 * 20 / timeSpan;   % goes 0 -> 40 m if timeSpan=9

% path y (PDR)
Rparam = 20; alpha_l = 12; c = 10;
y_a_r = c * (exp(-log(2) * (x_a_r./Rparam).^alpha_l) - 1);

% numerical derivatives (robust)
dx = gradient(x_a_r, t_ref);
ddx = gradient(dx, t_ref);
dy = gradient(y_a_r, t_ref);
ddy = gradient(dy, t_ref);

% curvature, speed, heading
eps0 = eps;
curvature = (dx .* ddy - dy .* ddx) ./ ((dx.^2 + dy.^2).^(3/2) + eps0);
v_ref = sqrt(dx.^2 + dy.^2);            % ref speed along path
psi_ref = atan2(dy, dx);
delta_ref = atan(lr1 .* curvature);

% interpolate reference to controller time grid
t_ctrl = (0:dt:timeSpan)';
Tsim = length(t_ctrl)-1;

x_ref = interp1(t_ref, x_a_r, t_ctrl, 'linear')';
y_ref = interp1(t_ref, y_a_r, t_ctrl, 'linear')';
psi_ref_i = interp1(t_ref, psi_ref, t_ctrl, 'linear')';
delta_r_profile = interp1(t_ref, delta_ref, t_ctrl, 'linear')';
k_profile = interp1(t_ref, curvature, t_ctrl, 'linear')';
v_profile = interp1(t_ref, v_ref, t_ctrl, 'linear')';

%% allocate actual state arrays
x_act = zeros(Tsim+1,1);
y_act = zeros(Tsim+1,1);
psi_act = zeros(Tsim+1,1);
psi_t_act = zeros(Tsim+1,1);
delta_act = zeros(Tsim+1,1);

% initial actual states (tune as needed)
x_act(1) = 1.0; y_act(1) = 1.0; psi_act(1) = 0.1;
psi_t_act(1) = 0.1; delta_act(1) = 0.05;

%% initial error in Frenet frame (time-synced)
dx0 = x_act(1) - x_ref(1); dy0 = y_act(1) - y_ref(1);
e_x0 =  cos(psi_ref_i(1))*dx0 + sin(psi_ref_i(1))*dy0;
e_y0 = -sin(psi_ref_i(1))*dx0 + cos(psi_ref_i(1))*dy0;
x_err = [ e_y0;
          wrapToPi(psi_act(1)-psi_ref_i(1));
          wrapToPi(psi_t_act(1)-psi_ref_i(1));
          delta_act(1)-delta_r_profile(1) ];

history_err = zeros(4, Tsim+1);
history_err(:,1) = x_err;
u_hist = zeros(Tsim,1);

%% simulation loop (LQR online)
for k=1:Tsim
    vr = v_profile(k);            % use ref speed so timing matches
    kappa = k_profile(k);
    delta_r = delta_r_profile(k);

    % linearize using your analytic linearSys
    [A,B,W] = linearSys(vr, kappa, delta_r, m0, lr1, lt1, tau);
    K = lqr(A,B,Q,R_lqr);

    % LQR on error-state x_err
    u_dev = -K * x_err;
    u_cmd = u_dev + delta_r;
    u_cmd_sat = min(max(u_cmd, umin), umax);
    u_hist(k) = u_cmd_sat;

    % nonlinear propagation (use your nonlinear model)
    psi_dot = (vr * tan(delta_act(k))) / lr1;
    psi_t_dot = (vr/lt1)*sin(psi_act(k)-psi_t_act(k)) ...
              - (m0*vr/(lr1*lt1))*cos(psi_act(k)-psi_t_act(k))*tan(delta_act(k));
    delta_dot = -(delta_act(k) - u_cmd_sat) / tau;

    x_act(k+1) = x_act(k) + vr * cos(psi_act(k)) * dt;
    y_act(k+1) = y_act(k) + vr * sin(psi_act(k)) * dt;
    psi_act(k+1) = psi_act(k) + psi_dot * dt;
    psi_t_act(k+1) = psi_t_act(k) + psi_t_dot * dt;
    delta_act(k+1) = delta_act(k) + delta_dot * dt;

    % compute Frenet error at next step (time-synced)
    idx_next = k+1;
    dxp = x_act(idx_next) - x_ref(idx_next);
    dyp = y_act(idx_next) - y_ref(idx_next);
    psir = psi_ref_i(idx_next);
    e_x =  cos(psir)*dxp + sin(psir)*dyp;
    e_y = -sin(psir)*dxp + cos(psir)*dyp;
    e_psi = wrapToPi(psi_act(idx_next) - psir);
    e_psi_t = wrapToPi(psi_t_act(idx_next) - psi_ref_i(idx_next));
    e_delta = delta_act(idx_next) - delta_r_profile(idx_next);
    x_err = [e_y; e_psi; e_psi_t; e_delta];
    history_err(:, idx_next) = x_err;
end

%% Plot results
figure('Units','normalized','Position',[0.05 0.05 0.9 0.8]);
subplot(2,2,1);
plot(x_ref, y_ref, 'k--','LineWidth',1.5); hold on;
plot(x_act, y_act, 'b-','LineWidth',1.2); axis equal; grid on;
xlabel('x [m]'); ylabel('y [m]'); title('Reference vs Actual Path');

subplot(2,2,2);
plot(t_ctrl, history_err(1,:), 'LineWidth',1.2); grid on; xlabel('t [s]'); ylabel('e_y [m]'); title('Lateral error');

subplot(2,2,3);
plot(t_ctrl, history_err(2,:), 'LineWidth',1.2); hold on;
plot(t_ctrl, history_err(3,:), '--','LineWidth',1.2); grid on;
legend('e_{\psi}','e_{\psi_t}'); xlabel('t [s]'); title('Heading errors');

subplot(2,2,4);
plot(t_ctrl(1:end-1), u_hist, 'LineWidth',1.2); grid on; xlabel('t [s]'); ylabel('\delta [rad]'); title('Steering command');

%% --- helper functions (kept from your original analytic linearization) ---
function [A, B, W] = linearSys(vr, k, delta_r, m0, lr1, lt1, tau)
    cos_delta_squared = cos(delta_r)^2;
    A = [0, vr    , 0      , 0;
         0, 0     , 0      , vr/(lr1*cos_delta_squared);
         0, vr/lt1, -vr/lt1, -m0*vr/(lr1*lt1*cos_delta_squared);
         0, 0     , 0      , -1/tau];

    B = [0; 0; 0; 1/tau];

    w1 = (vr/lr1)*(tan(delta_r) - delta_r/cos_delta_squared);
    W = [0; -k*vr + w1; (m0/lt1)*(k*vr - w1); 0];
end

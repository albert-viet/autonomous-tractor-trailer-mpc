% Autonomous car-pulling trailer using linear model predictive control
clear; close all; clc;

%% --- model parameters (same as yours) ---
tau = 0.3; lr1 = 2.9; lt1 = 1.8; m0 = 0.8;

%% sampling
dt = 0.03; umax = 0.52; umin = -umax;
%% MPC tuning
Np = 50;   % prediction horizon
Qx = diag([1, 1, 10, 1]);   % state weights (same ordering e_y,e_psi,e_psi_t,delta)
Ru = 0.1;                   % control move weight (on u_dev)
regularization = 1e-6;
u_prev = 0.0;
max_rate = 1.25;            % maximum steering rate (rad/s)
%% Reference generation (same as LQR script)
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

% interpolate to control grid
t_ctrl = (0:dt:timeSpan)'; Tsim = length(t_ctrl)-1;
x_ref = interp1(t_ref, x_a_r, t_ctrl, 'linear')';
y_ref = interp1(t_ref, y_a_r, t_ctrl, 'linear')';
psi_ref_i = interp1(t_ref, psi_ref, t_ctrl, 'linear')';
delta_r_profile = interp1(t_ref, delta_ref, t_ctrl, 'linear')';
k_profile = interp1(t_ref, curvature, t_ctrl, 'linear')';
v_profile = interp1(t_ref, v_ref, t_ctrl, 'linear')';

plot(t_ctrl,v_profile, 'r-','LineWidth',1.5);
title('Reference and Actual Trajectory'); xlabel('t [s]'); ylabel('v [m/s]'); axis equal; grid on; legend('v_{profile}');

%% Check if quadprog exists
if ~exist('quadprog','file')
    error(['quadprog not found. Please install Optimization Toolbox (Add-Ons -> Get Add-Ons -> Optimization Toolbox).', ...
          ' (Or install an alternative QP solver such as OSQP for MATLAB.)']);
end

%% allocate states
% tractor states
x_act = zeros(Tsim+1,1); y_act = zeros(Tsim+1,1);
psi_act = zeros(Tsim+1,1); psi_t_act = zeros(Tsim+1,1);
delta_act = zeros(Tsim+1,1);
% trailer states
x_t_act = zeros(Tsim+1,1); y_t_act = zeros(Tsim+1,1);

% tractor and trailer initial pose
x_act(1)=0.0; y_act(1)=0.0; psi_act(1)=0.0; psi_t_act(1)=0.0; delta_act(1)=0.0;
x_t_act(1) = x_act(1) - lt1*cos(psi_t_act(1)) - m0*cos(psi_act(1));
y_t_act(1) = y_act(1) - lt1*sin(psi_t_act(1)) - m0*sin(psi_act(1));

% initial error (Frenet)
dx0 = x_act(1) - x_ref(1); dy0 = y_act(1) - y_ref(1);
e_x0 =  cos(psi_ref_i(1))*dx0 + sin(psi_ref_i(1))*dy0;
e_y0 = -sin(psi_ref_i(1))*dx0 + cos(psi_ref_i(1))*dy0;
x_err = [ e_y0; wrapToPi(psi_act(1)-psi_ref_i(1)); wrapToPi(psi_t_act(1)-psi_ref_i(1)); delta_act(1)-delta_r_profile(1) ];

history_err = zeros(4, Tsim+1); history_err(:,1) = x_err;
u_hist = zeros(Tsim,1);

%% Precompute block weights
nx = 4; nu = 1; ny = 3; % output: e_y, e_psi, e_psi_t
Qbar = kron(eye(Np), Qx);
Rbar = kron(eye(Np), Ru);

%% Main MPC loop
for k = 1:Tsim
    vr = v_profile(k);
    kappa = k_profile(k);
    delta_r = delta_r_profile(k);

    % linearize (analytic) and discretize (Tustin)
    [A,B,W] = linearSys(vr, kappa, delta_r, m0, lr1, lt1, tau);
    [Ad, Bd, Cd, Wd] = tustin(A,B,W,dt);

    % Build prediction matrices Phi, Gamma, GammaW
    Phi = zeros(nx*Np, nx);
    Gamma = zeros(nx*Np, nu*Np);
    GammaW = zeros(nx*Np,1);
    Ad_power = eye(nx);
    
    % Build linear inequality constraints lb <= AU <= ub
    Aineq = zeros(Np+Np, nu*Np);
    
    % iterative construction
    for i = 1:Np
        Ad_power = Ad_power * Ad;       % Ad^i
        Phi((i-1)*nx+1:i*nx,:) = Ad_power;
        % Gamma blocks
        for j = 1:i
            block = Ad^(i-j) * Bd;
            Gamma((i-1)*nx+1:i*nx, (j-1)*nu+1:j*nu) = block;
        end
        % W contribution: sum_{j=0}^{i-1} Ad^j * Wd
        Wsum = zeros(nx,1);
        for j = 0:(i-1)
            Wsum = Wsum + Ad^j * Wd;
        end
        GammaW((i-1)*nx+1:i*nx) = Wsum;
    end

    % formulate QP: min_z 1/2 z'H z + f' z , z = [u_dev(0); ...; u_dev(Np-1)]
    % cost: sum_{i} x_i' Qx x_i + sum_{i} u_i' Ru u_i
    H = 2*(Gamma' * Qbar * Gamma + Rbar) + regularization*eye(nu*Np);
    H = (H + H')/2;   % ensure H is symmetric
    f = 2*(Gamma' * Qbar * (Phi * x_err + GammaW));

    % input bounds on u_dev: u_dev ∈ [umin - delta_r, umax - delta_r]
    lb = repmat(umin, Np, 1);
    ub = repmat(umax, Np, 1);
    
    % input bounds on (u-udev)
    bineq = zeros(2*Np, 1);
    
    % defining linear inequality constraints
    for i = 1:Np
        if i == 1
            Aineq(1:2,1) = [1; -1];
            bineq(1:2)   = [u_prev + max_rate*dt;
                            -u_prev + max_rate*dt];
        else
            Aineq(2*i-1:2*i, i-1:i) = [1 -1; -1 1];
            bineq(2*i-1:2*i)        = [max_rate*dt; max_rate*dt];
        end
    end

    opts = optimoptions('quadprog','Display','off','TolFun',1e-6);
    [zopt,~,exitflag] = quadprog(H, f, Aineq, bineq, [], [], lb, ub, [], opts);
    if exitflag ~= 1 && exitflag ~= 0
        % fallback to saturated LQR if QP fails
        disp('OSQP fail')
        warning('quadprog did not converge at step %d (flag=%d). Using saturated LQR fallback.', k, exitflag)
        [A_lin, B_lin, W_lin] = linearSys(vr, kappa, delta_r, m0, lr1, lt1, tau);
        Kfb = lqr(A_lin,B_lin,Qx,Ru);
        u_dev = -Kfb * x_err;
        u_cmd = min(max(u_dev + delta_r, umin), umax);
        u_cmd_sat = u_cmd;
    else
        u_dev_opt = zopt(1:nu);  % only use first control move
        disp(u_dev_opt)
        % assign previous cmd
        u_prev = u_dev_opt;
        
        u_cmd = u_dev_opt + delta_r;
        u_cmd_sat = min(max(u_cmd, umin), umax);
    end
    u_hist(k) = u_cmd_sat;
    
    % propagate nonlinear states
    psi_dot = (vr * tan(delta_act(k))) / lr1;
    psi_t_dot = (vr/lt1)*sin(psi_act(k)-psi_t_act(k)) - (m0*vr/(lr1*lt1))*cos(psi_act(k)-psi_t_act(k))*tan(delta_act(k));
    delta_dot = -(delta_act(k) - u_cmd_sat) / tau;

    x_act(k+1) = x_act(k) + vr * cos(psi_act(k)) * dt;
    y_act(k+1) = y_act(k) + vr * sin(psi_act(k)) * dt;
    psi_act(k+1) = psi_act(k) + psi_dot * dt;
    psi_t_act(k+1) = psi_t_act(k) + psi_t_dot * dt;
    delta_act(k+1) = delta_act(k) + delta_dot * dt;
    
    % calculate the trailer position
    x_t_act(k+1) = x_act(k+1) - lr1*cos(psi_t_act(k+1)) - m0*cos(psi_act(k+1));
    y_t_act(k+1) = y_act(k+1) - lr1*sin(psi_t_act(k+1)) - m0*sin(psi_act(k+1));

    % update error (Frenet) at next timestep
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
plot(x_act,y_act,'b-','LineWidth',1.2); hold on; 
plot(x_t_act, y_t_act, 'r-','LineWidth',1.5);
title('Reference and Actual Trajectory'); xlabel('x [m]'); ylabel('y [m]'); axis equal; grid on; legend('ref','tractor','trailer');

subplot(2,2,2);
plot(t_ctrl, history_err(1,:), 'LineWidth',1.5);
title('Tractor lateral error [m]'); xlabel('t [s]'); ylabel('y [m]'); grid on; legend('e_{y}');

subplot(2,2,3);
plot(t_ctrl, history_err(2,:), 'LineWidth',1.5); hold on; plot(t_ctrl, history_err(3,:),'--','LineWidth',1.5);
title('Tractor and trailer yaw error [rad]'); xlabel('t [s]'); ylabel('rad'); grid on; legend('e_\psi','e_{\psi_t}');

subplot(2,2,4);
hold on
plot(t_ctrl(1:end-1), u_hist, 'LineWidth',1.5);
plot(t_ctrl(1:end-1), umax*ones(size(t_ctrl(1:end-1))), 'r--', 'LineWidth', 1.5);
plot(t_ctrl(1:end-1), umin*ones(size(t_ctrl(1:end-1))), 'r--', 'LineWidth', 1.5);
title('Steering control command [rad]'); xlabel('t [s]'); ylabel('rad'); legend('u_{cmd}','u_{max}','u_{min}'); grid on;

%% --- helper functions ---
function [A, B, W] = linearSys(vr, k, delta_r, m0, lr1, lt1, tau)
    cos_delta_squared = cos(delta_r)^2;
    A = [0, vr    , 0      , 0;
         0, 0     , 0      , vr/(lr1*cos_delta_squared);
         0, vr/lt1, -vr/lt1, -m0*vr/(lr1*lt1*cos_delta_squared);
         0, 0     , 0      , -1/tau];
    B = [0;0;0;1/tau];
    w1 = (vr/lr1)*(tan(delta_r) - delta_r/cos_delta_squared);
    W = [0; -k*vr + w1; (m0/lt1)*(k*vr - w1); 0];
end

function [Ad, Bd, Cd, Wd] = tustin(A,B,W,dt)
    I = eye(size(A));
    Ad = (I - 0.5*dt*A) \ (I + 0.5*dt*A);
    Bd = (I - 0.5*dt*A) \ (B * dt);
    Cd = [1, 0, 0, 0;
          0, 1, 0, 0;
          0, 0, 1, 0];
    Wd = (I - 0.5*dt*A) \ (W * dt);
end

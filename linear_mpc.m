% Autonomous car-pulling trailer using linear model predictive control
clear; close all; clc;

%% --- model parameters ---
tau = 0.3; lr1 = 2.9; lt1 = 1.8; m0 = 0.8;

%% sampling
dt = 0.03; umax = 0.52; umin = -umax;
%% MPC tuning
Np = 50;   % prediction horizon
% Qx = diag([9.3706, 1.2718, 0.10021]);   % state weights (same ordering e_y,e_psi,e_psi_t)
% Ru = 0.3821;                   % control move weight (on u_dev)
Qx = diag([10, 5, 0.1]);
Ru = 0.5;
regularization = 1e-6;
u_prev = 0.0;
max_rate = 1.2;            % maximum steering rate (rad/s)
%% Reference generation
timeSpan = 60; timeStep = 0.005;
t_ref = (0:timeStep:timeSpan)';
% x_a_r = t_ref * 2 * 20 / timeSpan;
% Rparam = 20; alpha_l = 12; c = 10;
% y_a_r = c * (exp(-log(2) * (x_a_r./Rparam).^alpha_l) - 1);

% Allocate reference
x_a_r = zeros(size(t_ref));
y_a_r = zeros(size(t_ref));
psi_ref = zeros(size(t_ref));
v_ref = zeros(size(t_ref));

% ---- Phase 1: t = [0,3] ----
idx1 = t_ref <= 3;
t1 = t_ref(idx1);
vr1 = 0.1 + 0.9 * sin(pi * t1 / 6);           % longitudinal velocity
x1 = cumtrapz(t1, vr1);                        % integrate to get x
y1 = zeros(size(t1));
psi1 = zeros(size(t1));                        % heading constant 0

x_a_r(idx1) = x1;
y_a_r(idx1) = y1;
psi_ref(idx1) = psi1;
v_ref(idx1) = vr1;

% ---- Phase 2: t = [3,43] ----
idx2 = t_ref > 3;
t2 = t_ref(idx2);
x2 = t2 - 1;
y2 = 5 * sin(pi/20 * (t2 - 3) + 1.5*pi) + 5;

dx2 = gradient(x2, timeStep);
dy2 = gradient(y2, timeStep);

psi2 = atan2(dy2, dx2);
vr2 = sqrt(dx2.^2 + dy2.^2);

x_a_r(idx2) = x2;
y_a_r(idx2) = y2;
psi_ref(idx2) = psi2;
v_ref(idx2) = vr2;
%% calculate reference profile

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
x_act(1)=0.2; y_act(1)=0.4; psi_act(1)=0.0; psi_t_act(1)=0.0; delta_act(1)=0.0;
x_t_act(1) = x_act(1) - lt1*cos(psi_t_act(1)) - m0*cos(psi_act(1));
y_t_act(1) = y_act(1) - lt1*sin(psi_t_act(1)) - m0*sin(psi_act(1));

% initial error (Frenet)
% dx0 = x_act(1) - x_ref(1); dy0 = y_act(1) - y_ref(1);
% e_x0 =  cos(psi_ref_i(1))*dx0 + sin(psi_ref_i(1))*dy0;
% e_y0 = -sin(psi_ref_i(1))*dx0 + cos(psi_ref_i(1))*dy0;
% x_err = [ e_y0; wrapToPi(psi_act(1)-psi_ref_i(1)); wrapToPi(psi_t_act(1)-psi_ref_i(1)); delta_act(1)];
[xproj_tr, yproj_tr, psi_ref_tr, k_tr_init, delta_r_tr_init, idx_tr_init] = projectRefPoint(x_act(1), y_act(1), x_ref, y_ref, psi_ref_i, k_profile, delta_r_profile);
[xproj_tl, yproj_tl, psi_ref_tl, k_tl_init, delta_r_tl_init, idx_tl_init] = projectRefPoint(x_t_act(1), y_t_act(1), x_ref, y_ref, psi_ref_i, k_profile, delta_r_profile);

dx0 = x_act(1) - xproj_tr; dy0 = y_act(1) - yproj_tr;
e_y0 = -sin(psi_ref_tr)*dx0 + cos(psi_ref_tr)*dy0;
e_psi0 = wrapToPi(psi_act(1) - psi_ref_tr);
e_psi_t0 = wrapToPi(psi_t_act(1) - psi_ref_tl);

x_err = [e_y0; e_psi0; e_psi_t0; delta_act(1)];
% keep previous projection indices to accelerate next searches (optional)
last_idx_tr = idx_tr_init;
last_idx_tl = idx_tl_init;

history_err = zeros(4, Tsim+1); history_err(:,1) = x_err;
u_hist = zeros(Tsim,1);

%% Precompute block weights
nx = 4; nu = 1; ny = 3; % output: e_y, e_psi, e_psi_t
Qbar = kron(eye(Np), Qx);
Rbar = kron(eye(Np), Ru);

%% Main MPC loop
for k = 1:Tsim
    % project tractor & trailer current positions to reference
    [xproj_tr, yproj_tr, psi_ref_tr, k_truck, delta_r_truck, last_idx_tr] = projectRefPoint(x_act(k), y_act(k), x_ref, y_ref, psi_ref_i, k_profile, delta_r_profile);
    [xproj_tl, yproj_tl, psi_ref_tl, k_trailer, delta_r_tl, last_idx_tl] = projectRefPoint(x_t_act(k), y_t_act(k), x_ref, y_ref, psi_ref_i, k_profile, delta_r_profile);

    vr = v_profile(k); % vehicle speed (still taken from profile)
    % linearize using truck curvature for tractor part and trailer curvature for trailer feedforward
    [A,B,W] = linearSys(vr, k_truck, delta_r_truck, psi_ref_tr, psi_ref_tl, m0, lr1, lt1, tau);
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
    
    Phi_Y = kron(eye(Np), Cd)*Phi;
    Gamma_Y  = kron(eye(Np), Cd) * Gamma;
    GammaW_Y = kron(eye(Np), Cd) * GammaW;

    % formulate QP: min_z 1/2 z'H z + f' z , z = [u_dev(0); ...; u_dev(Np-1)]
    % cost: sum_{i} x_i' Qx x_i + sum_{i} u_i' Ru u_i
%     H = 2*(Gamma' * Qbar * Gamma + Rbar) + regularization*eye(nu*Np);
%     H = (H + H')/2;   % ensure H is symmetric
%     f = 2*(Gamma' * Qbar * (Phi * x_err + GammaW));
    H = 2*(Gamma_Y' * Qbar * Gamma_Y + Rbar) + regularization*eye(nu*Np);
    H = (H + H')/2;   % ensure H is symmetric
    f = 2*(Gamma_Y' * Qbar * (Phi_Y * x_err + GammaW_Y)) - 2*Rbar * kron(ones(Np,1), delta_r_truck);
    
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
        u_opt = zopt(1:nu);  % only use first control move
        disp(u_opt)     
        u_cmd = u_opt;
        u_cmd_sat = min(max(u_cmd, umin), umax);
        % assign previous cmd   
        u_prev = u_opt;
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
    % project new positions (tractor & trailer) onto reference
    [xproj_tr_p, yproj_tr_p, psi_ref_tr_p, ~, ~, last_idx_tr] = projectRefPoint(x_act(k+1), y_act(k+1), x_ref, y_ref, psi_ref_i, k_profile, delta_r_profile);
    [xproj_tl_p, yproj_tl_p, psi_ref_tl_p, ~, ~, last_idx_tl] = projectRefPoint(x_t_act(k+1), y_t_act(k+1), x_ref, y_ref, psi_ref_i, k_profile, delta_r_profile);

    dxp = x_act(k+1) - xproj_tr_p; dyp = y_act(k+1) - yproj_tr_p;
    e_y = -sin(psi_ref_tr_p)*dxp + cos(psi_ref_tr_p)*dyp;
    e_psi = wrapToPi(psi_act(k+1) - psi_ref_tr_p);
    e_psi_t = wrapToPi(psi_t_act(k+1) - psi_ref_tl_p);

    x_err = [e_y; e_psi; e_psi_t; delta_act(k+1)];
    history_err(:, k+1) = x_err;

end

%% tractor-trailer trajectory tracking animation
figure; hold on; axis equal; grid on;
title('Tractor and trailer tracking animation'); 
xlabel('x [m]'); ylabel('y [m]');

% kích thước xe
L_tr = 2.9; W_tr = 2.0;
L_t  = 1.8; W_t = 2.0;

for k = 1:10:Tsim
    cla;
    % reference
    plot(x_ref, y_ref, 'k--', 'LineWidth', 1.5); hold on;
    
    % plot trajectory 
    plot(x_act(1:k), y_act(1:k), 'b-', 'LineWidth', 1.2);
    plot(x_t_act(1:k), y_t_act(1:k), 'r-', 'LineWidth', 1.2);

    % tractor
    tractor_corners = rectangleCorners(x_act(k), y_act(k), psi_act(k), L_tr, W_tr);
    fill(tractor_corners(1,:), tractor_corners(2,:), 'b', 'FaceAlpha',0.3);

    % trailer
    trailer_corners = rectangleCorners(x_t_act(k), y_t_act(k), psi_t_act(k), L_t, W_t);
    fill(trailer_corners(1,:), trailer_corners(2,:), 'r', 'FaceAlpha',0.3);

    % draw hitch line
    hitch_tractor = [x_act(k) - m0*cos(psi_act(k));
                     y_act(k) - m0*sin(psi_act(k))];
    hitch_trailer = [x_t_act(k) + lt1*cos(psi_t_act(k));
                     y_t_act(k) + lt1*sin(psi_t_act(k))];
    plot([hitch_tractor(1), hitch_trailer(1)], ...
         [hitch_tractor(2), hitch_trailer(2)], 'k-', 'LineWidth',2);

    legend('ref','tractor path','trailer path','tractor','trailer','hitch');
    drawnow;
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
function [A, B, W] = linearSys(vr, k, delta_r, psi_ref, psi_t_ref, m0, lr1, lt1, tau)
    % util variables
    cos_delta_squared = cos(delta_r)^2;
    delta_yaw = psi_ref - psi_t_ref;
    e_psi_term = (m0/lr1)*sin(delta_yaw)*tan(delta_r);
    e1 = (2*vr/lt1)*e_psi_term;
    e2 = (-2*vr/lt1)*cos(delta_yaw) - e1;
    
    A = [0, vr    , 0      , 0;
         0, 0     , 0      , vr/(lr1*cos_delta_squared);
         0, e1    , e2    , -m0*vr*cos(delta_yaw)/(lr1*lt1*cos_delta_squared);
         0, 0     , 0      , -1/tau];
    B = [0;0;0;1/tau];
    w1 = (vr/lr1)*(tan(delta_r) - delta_r/cos_delta_squared);
    W = [0; -k*vr + w1; 0; 0];
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

function [xproj, yproj, psi_interp, kappa_interp, delta_interp, idx] = projectRefPoint(xv, yv, x_ref, y_ref, psi_ref_arr, kappa_arr, delta_arr)
    % Project point (xv,yv) onto polyline given by (x_ref,y_ref).
    % Returns projected point coords, interpolated psi (wrapped), kappa, delta, and base index.
    d2 = (x_ref - xv).^2 + (y_ref - yv).^2;
    [~, idx0] = min(d2);
    N = numel(x_ref);
    % choose neighbouring point to form a segment
    if idx0 < N
        i1 = idx0; i2 = idx0+1;
    else
        i1 = idx0-1; i2 = idx0;
    end
    p1 = [x_ref(i1); y_ref(i1)];
    p2 = [x_ref(i2); y_ref(i2)];
    v = p2 - p1;
    if norm(v) < 1e-8
        tproj = 0;
    else
        tproj = dot([xv; yv] - p1, v) / dot(v, v);
        tproj = min(max(tproj, 0), 1);
    end
    proj = p1 + tproj * v;
    xproj = proj(1); yproj = proj(2);
    % interpolate psi carefully (shortest angle)
    psi1 = psi_ref_arr(i1); psi2 = psi_ref_arr(i2);
    dpsi = wrapToPi(psi2 - psi1);
    psi_interp = wrapToPi(psi1 + tproj * dpsi);
    kappa_interp = (1-tproj)*kappa_arr(i1) + tproj*kappa_arr(i2);
    delta_interp = (1-tproj)*delta_arr(i1) + tproj*delta_arr(i2);
    idx = i1;
end


function corners = rectangleCorners(xc, yc, psi, L, W)
    % local coordinates (centered at vehicle center)
    corners_local = [ L/2,  W/2;
                      L/2, -W/2;
                     -L/2, -W/2;
                     -L/2,  W/2]';
    R = [cos(psi), -sin(psi);
         sin(psi),  cos(psi)];
    corners = R * corners_local + [xc; yc];
end

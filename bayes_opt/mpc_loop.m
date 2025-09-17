function [total_rmse, history_err, u_hist, x_act, y_act, x_t_act, y_t_act, psi_act, psi_t_act, delta_act] = ...
    mpc_loop(Qx, Ru, dt, umax, max_rate, Np, regularization, ...
    x_ref, y_ref, psi_ref_i, delta_r_profile, k_profile, v_profile, m0, lr1, lt1, tau)
    % Input: Qx (3x3 diag for e_y, e_psi, e_psi_t), Ru (scalar), params từ code gốc
    % Output: total_rmse (cho BO), history_err, u_hist, states để vẽ
    
    % Allocate states
    Tsim = length(v_profile) - 1;
    x_act = zeros(Tsim+1,1); y_act = zeros(Tsim+1,1);
    psi_act = zeros(Tsim+1,1); psi_t_act = zeros(Tsim+1,1);
    delta_act = zeros(Tsim+1,1);
    x_t_act = zeros(Tsim+1,1); y_t_act = zeros(Tsim+1,1);
    
    % Initial conditions
    x_act(1) = 0.2; y_act(1) = 0.4; psi_act(1) = 0.0; psi_t_act(1) = 0.0; delta_act(1) = 0.0;
    x_t_act(1) = x_act(1) - lt1*cos(psi_t_act(1)) - m0*cos(psi_act(1));
    y_t_act(1) = y_act(1) - lt1*sin(psi_t_act(1)) - m0*sin(psi_act(1));
    
    % Initial error (Frenet)
    dx0 = x_act(1) - x_ref(1); dy0 = y_act(1) - y_ref(1);
    e_x0 = cos(psi_ref_i(1))*dx0 + sin(psi_ref_i(1))*dy0;
    e_y0 = -sin(psi_ref_i(1))*dx0 + cos(psi_ref_i(1))*dy0;
    x_err = [e_y0; wrapToPi(psi_act(1)-psi_ref_i(1)); wrapToPi(psi_t_act(1)-psi_ref_i(1)); delta_act(1)];
    
    history_err = zeros(4, Tsim+1); history_err(:,1) = x_err;
    u_hist = zeros(Tsim,1);
    u_prev = 0.0;
    
    % Precompute block weights
    nx = 4; nu = 1; ny = 3;
    Qbar = kron(eye(Np), Qx);
    Rbar = kron(eye(Np), Ru);
    
    % Main MPC loop
    for k = 1:Tsim
        vr = v_profile(k); kappa = k_profile(k); delta_r = delta_r_profile(k);
        
        % Linearize & discretize
        [A,B,W] = linearSys(vr, kappa, delta_r, m0, lr1, lt1, tau);
        [Ad, Bd, Cd, Wd] = tustin(A,B,W,dt);
        
        % Build prediction matrices
        Phi = zeros(nx*Np, nx); Gamma = zeros(nx*Np, nu*Np); GammaW = zeros(nx*Np,1);
        Ad_power = eye(nx);
        for i = 1:Np
            Ad_power = Ad_power * Ad;
            Phi((i-1)*nx+1:i*nx,:) = Ad_power;
            for j = 1:i
                block = Ad^(i-j) * Bd;
                Gamma((i-1)*nx+1:i*nx, (j-1)*nu+1:j*nu) = block;
            end
            Wsum = zeros(nx,1);
            for j = 0:(i-1)
                Wsum = Wsum + (Ad^j) * Wd;
            end
            GammaW((i-1)*nx+1:i*nx) = Wsum;
        end
        
        % Output prediction matrices
        Phi_Y = kron(eye(Np), Cd)*Phi;
        Gamma_Y = kron(eye(Np), Cd)*Gamma;
        GammaW_Y = kron(eye(Np), Cd)*GammaW;
        
        % QP setup
        H = 2*(Gamma_Y' * Qbar * Gamma_Y + Rbar) + regularization*eye(nu*Np);
        H = (H + H')/2;
        f = 2*(Gamma_Y' * Qbar * (Phi_Y * x_err + GammaW_Y));
        
        % Constraints
        lb = repmat(-umax, Np, 1); ub = repmat(umax, Np, 1);
        Aineq = zeros(2*Np, nu*Np); bineq = zeros(2*Np, 1);
        for i = 1:Np
            if i == 1
                Aineq(1:2,1) = [1; -1];
                bineq(1:2) = [u_prev + max_rate*dt; -u_prev + max_rate*dt];
            else
                Aineq(2*i-1:2*i, i-1:i) = [1 -1; -1 1];
                bineq(2*i-1:2*i) = [max_rate*dt; max_rate*dt];
            end
        end
        
        opts = optimoptions('quadprog','Display','off','TolFun',1e-6);
        [zopt,~,exitflag] = quadprog(H, f, Aineq, bineq, [], [], lb, ub, [], opts);
        
        if exitflag ~= 1 && exitflag ~= 0
            warning('quadprog failed at step %d (flag=%d). Using LQR.', k, exitflag);
            [A_lin, B_lin, ~] = linearSys(vr, kappa, delta_r, m0, lr1, lt1, tau);
            Kfb = lqr(A_lin, B_lin, Qx, Ru);
            u_dev = -Kfb * x_err;
            u_cmd = min(max(u_dev + delta_r, -umax), umax);
        else
            u_dev_opt = zopt(1);
            u_prev = u_dev_opt;
            u_cmd = u_dev_opt + delta_r;
            u_cmd = min(max(u_cmd, -umax), umax);
        end
        u_hist(k) = u_cmd;
        
        % Propagate nonlinear states
        psi_dot = (vr * tan(delta_act(k))) / lr1;
        psi_t_dot = (vr/lt1)*sin(psi_act(k)-psi_t_act(k)) - (m0*vr/(lr1*lt1))*cos(psi_act(k)-psi_t_act(k))*tan(delta_act(k));
        delta_dot = -(delta_act(k) - u_cmd) / tau;
        
        x_act(k+1) = x_act(k) + vr * cos(psi_act(k)) * dt;
        y_act(k+1) = y_act(k) + vr * sin(psi_act(k)) * dt;
        psi_act(k+1) = psi_act(k) + psi_dot * dt;
        psi_t_act(k+1) = psi_t_act(k) + psi_t_dot * dt;
        delta_act(k+1) = delta_act(k) + delta_dot * dt;
        
        x_t_act(k+1) = x_act(k+1) - lr1*cos(psi_t_act(k+1)) - m0*cos(psi_act(k+1));
        y_t_act(k+1) = y_act(k+1) - lr1*sin(psi_t_act(k+1)) - m0*sin(psi_act(k+1));
        
        % Update error (Frenet)
        idx_next = k+1;
        dxp = x_act(idx_next) - x_ref(idx_next);
        dyp = y_act(idx_next) - y_ref(idx_next);
        psir = psi_ref_i(idx_next);
        e_x = cos(psir)*dxp + sin(psir)*dyp;
        e_y = -sin(psir)*dxp + cos(psir)*dyp;
        e_psi = wrapToPi(psi_act(idx_next) - psir);
        e_psi_t = wrapToPi(psi_t_act(idx_next) - psi_ref_i(idx_next));
        x_err = [e_y; e_psi; e_psi_t; delta_act(idx_next)];
        history_err(:, idx_next) = x_err;
    end
    
    % Calculate RMSE for BO
    rmse_y = sqrt(mean(history_err(1,:).^2));
    rmse_psi = sqrt(mean(history_err(2,:).^2));
    rmse_psi_t = sqrt(mean(history_err(3,:).^2));
    total_rmse = rmse_y + 0.5*rmse_psi + 0.5*rmse_psi_t;  % Weighted sum
end

% Helper functions (từ code gốc)
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
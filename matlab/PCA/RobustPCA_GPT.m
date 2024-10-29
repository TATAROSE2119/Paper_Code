function [L, S] = RobustPCA_GPT(X, lambda, tol, max_iter)
    % 初始化
    [m, n] = size(X);
    L = zeros(m, n);
    S = zeros(m, n);
    Y = zeros(m, n);
    mu = 1 / norm(X, 2); % ADMM 参数
    iter = 0;

    while iter < max_iter
        % 更新 L
        L = svd_threshold(X - S + (1 / mu) * Y, 1 / mu);

        % 更新 S
        S = shrink(X - L + (1 / mu) * Y, lambda / mu);

        % 更新 Y
        Y = Y + mu * (X - L - S);

        % 计算收敛条件
        criterion = norm(X - L - S, 'fro') / norm(X, 'fro');
        
        % 判断是否达到收敛
        if criterion < tol
            break;
        end

        % 更新 ADMM 参数 mu
        mu = mu * 1.1;

        iter = iter + 1;
    end
end

% 奇异值软阈值函数
function X = svd_threshold(X, tau)
    [U, S, V] = svd(X, 'econ');
    S = diag(S);
    S = max(S - tau, 0);
    X = U * diag(S) * V';
end

% 软阈值函数
function X = shrink(X, tau)
    X = sign(X) .* max(abs(X) - tau, 0);
end

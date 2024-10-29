function [L, S] = RobustPCA_GPT(X, lambda, tol, max_iter)
    % ��ʼ��
    [m, n] = size(X);
    L = zeros(m, n);
    S = zeros(m, n);
    Y = zeros(m, n);
    mu = 1 / norm(X, 2); % ADMM ����
    iter = 0;

    while iter < max_iter
        % ���� L
        L = svd_threshold(X - S + (1 / mu) * Y, 1 / mu);

        % ���� S
        S = shrink(X - L + (1 / mu) * Y, lambda / mu);

        % ���� Y
        Y = Y + mu * (X - L - S);

        % ������������
        criterion = norm(X - L - S, 'fro') / norm(X, 'fro');
        
        % �ж��Ƿ�ﵽ����
        if criterion < tol
            break;
        end

        % ���� ADMM ���� mu
        mu = mu * 1.1;

        iter = iter + 1;
    end
end

% ����ֵ����ֵ����
function X = svd_threshold(X, tau)
    [U, S, V] = svd(X, 'econ');
    S = diag(S);
    S = max(S - tau, 0);
    X = U * diag(S) * V';
end

% ����ֵ����
function X = shrink(X, tau)
    X = sign(X) .* max(abs(X) - tau, 0);
end

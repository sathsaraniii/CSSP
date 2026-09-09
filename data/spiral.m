function [A, X] = spiral(n, bandwidth)
    % robspiral from Ethan Epperly's github code
    times = linspace(0, 2, n);
    times = times.^6;
    times = flip(times);
    x = exp(0.2 * times) .* cos(times);
    y = exp(0.2 * times) .* sin(times);
    X = [x(:), y(:)];
    A = zeros(n);
    for i = 1:n
        A(i,:) = exp(-vecnorm(X(i,:)'*ones(1,n) - X').^2/(2*bandwidth^2));
    end
    % plot(X(:,1), X(:,2), '.')
    % axis('square')
end
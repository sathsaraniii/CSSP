function [A, X] = smile(N, bandwidth, varargin)
    % smiley from Ethan Epperly's github code    
    % Set eye_points and mouth_points
    eye_points = ceil(N^(1.0/2));
    mouth_points = ceil(N / 10.0);
    face_points = N - 2 * eye_points - mouth_points;

    X = zeros(N, 2);
    idx = 1;

    % Eyes
    for x_shift = [-4.0, 4.0]
        for i = 1:eye_points
            while true
                x = 2 * rand() - 1;
                y = 2 * rand() - 1;
                if x^2 + y^2 <= 1.0
                    X(idx, 1) = x + x_shift;
                    X(idx, 2) = y + 4.0;
                    idx = idx + 1;
                    break;
                end
            end
        end
    end

    % Mouth
    for x = linspace(-5.0, 5.0, mouth_points)
        X(idx, 1) = x;
        X(idx, 2) = x^2 / 16.0 - 5.0;
        idx = idx + 1;
    end

    % Face
    for theta = linspace(0, 2 * pi, face_points)
        X(idx, 1) = 10.0 * cos(theta);
        X(idx, 2) = 10.0 * sin(theta);
        idx = idx + 1;
    end

    % Return the KernelMatrix
    A = zeros(N);
    for i = 1:N
        A(i,:) = exp(-vecnorm(X(i,:)'*ones(1,N) - X').^2/(2*bandwidth^2));
    end
    % plot(X(:,1), X(:,2), '.')
    % axis('square')
end
function [fval, gradfval ] = logistic_loss(alpha, K, y)

    f = K*alpha;
    fval = -sum(log((1+exp(-y.*f)).^-1)) + 0.5 * alpha' * K * alpha;
    gradfval = -K*(((1+exp(y.*f)).^-1).*y) + f;
    
end

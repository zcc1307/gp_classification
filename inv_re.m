function [ ub ] = inv_re( p, d )
% function [ ub ] = inv_re( p, d )
% return the inverted Bernoulli relative entropy function value

    % for numerical stability reasons
    p = p + 0.001;
    ub = 1;
    lb = p;

    while abs(ub - lb) > 0.001
        m = (ub + lb)/2;
        if p*log(p/m) + (1-p)*log((1-p)/(1-m)) > d
            ub = m;
        else
            lb = m;
        end
    end

end


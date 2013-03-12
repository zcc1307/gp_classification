function [ fval, fgradval ] = log_z_noisepar(y, mu, zeta, b_lambda)
%LOG_Z_NOISEPAR Summary of this function goes here
%   calculate the function value and the partial derivatives wrt (b, lambda)

    b = b_lambda(1);
    lambda = b_lambda(2);
    
    c = y ./ sqrt(lambda^-2 + zeta);
    u = c .* (mu + b);


    fval = -sum(log(normcdf(u)));
    
    g = c .* normpdf(u) ./ normcdf(u); 
    G = -0.5 * g .* c .* u;
    
    Z_b = g;
    Z_lambda = -2*lambda^-3 * G;
    
    
    fgradval = -[sum(Z_b); sum(Z_lambda)];

end


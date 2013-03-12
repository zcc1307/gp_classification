function [ mu, sigma, zo_err, gibbs_err ] = get_err_dir( diag_K, K_D_I, C, alpha, Y)
% function [ mu, sigma, zo_err, gibbs_err ] = get_err( diag_K, K_D_I, L, w, alpha, Y)
% return the zero one err and Gibbs error, mean, stddev of the posterior of examples' f
% value 
    
    data_len = size(K_D_I, 1);
    
    mu = K_D_I * alpha;
    sigma = zeros(data_len, 1);
    zo_err = zeros(data_len, 1);
    gibbs_err = zeros(data_len, 1);
    
    for i = 1:data_len
        sigma(i) = sqrt( diag_K(i) + K_D_I(i,:) * C *K_D_I(i,:)' );
        pred = -Y(i) * mu(i) / sigma(i);
        zo_err(i) = (pred >= 0);
        gibbs_err(i) = normcdf(pred); 
    end
end
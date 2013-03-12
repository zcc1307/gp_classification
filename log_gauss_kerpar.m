function [ fval ] = log_gauss_kerpar(m, beta, X, w_C)
%LOG_GAUSS_KERPAR Summary of this function goes here
%  calculate the log likelihood and the derivatives with respect to kernel
%  parameters (w, C)

    w = w_C(1);
    C = w_C(2);
    d = size(X,2);
    params = [];
    params.kernel = 'Gauss';
    params.kernel_parameter1 = sqrt(d/w);
    params.kernel_parameter2 = C;
    K = ker_matrix(X, X, params);
    Sigma = K + diag(beta.^-1);
    
    %Sigma
    
    fval = 0.5 * (log(det(Sigma)) + m'*(Sigma\m));
    
    
    iSigma = inv(Sigma);
    iSigma_m = Sigma \ m;
    
    f_Sigma = 0.5 * (-iSigma + iSigma_m * iSigma_m');
    Sigma_w = (log(K / C) / w) .* K;
    Sigma_C = K / C;
    
    gradfval = -[sum(dot(f_Sigma,Sigma_w)); sum(dot(f_Sigma,Sigma_C))]; 



end


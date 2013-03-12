function [coverage, err_mv, err_m, gibbs_test_err, ub] = sogp(data_train, data_test, params)
% function [coverage, err_mv, err_m, gibbs_test_err, ub] = sogp(data_train, data_test, params)
% the sparse online Gaussian Process implementation
% based on training examples, and testing examples, get the RC curve and
% the verification of the Gibbs test error and its estimated upper bound


    %% basic settings
    train_len = params.train_len;
    test_len = params.test_len;

    X_train = data_train.X;
    Y_train = data_train.Y;
    X_test = data_test.X;
    Y_test = data_test.Y;


    I = [];
    d = 10;

    C = zeros(d,d);
    Q = zeros(d,d);
    alpha = zeros(d,1);
    sigma_0 = 1;
    epsilon = 0.0001;
    i_e = 0; % #effective i, always equal to sum(I);
    
    
    
    %only perform extending update first
    for i = 0:train_len*3-1
        i_p = mod(i,train_len) + 1;
        x = X_train(i_p,:);
        y = Y_train(i_p);
        k_x_s = ker_matrix(x, x, params);
        
        
        if (i_e == 0)
            sigma_x = sqrt(sigma_0^2 + k_x_s);
            f_x = 0;
            gamma_x = k_x_s;
        else
            k_x = ker_matrix(x, X_train(I,:), params)';
            sigma_x = sqrt(sigma_0*sigma_0 + k_x_s + k_x' * C(1:i_e,1:i_e) * k_x);
            f_x = alpha(1:i_e)' * k_x;
            e_x = Q(1:i_e,1:i_e) * k_x;
            gamma_x = k_x_s - k_x' * (Q(1:i_e,1:i_e) * k_x);
        end
               
        m = y * f_x / sigma_x;
        erf_m = normcdf(m);
        erf_m_p = normpdf(m);
        erf_m_pp = normpdf(m) * (-m);
        q = ( y * erf_m_p ) / (sigma_x * erf_m);
        r = ( erf_m_pp / erf_m - (erf_m_p / erf_m)^2 ) / (sigma_x^2);
        
        
        % perform a reduced update only, will not be touched in first round
        if gamma_x < epsilon
            %'reduced'
            
            s = C(1:i_e,1:i_e)*k_x + e_x;
            alpha(1:i_e) = alpha(1:i_e) + q * s;
            C(1:i_e,1:i_e) = C(1:i_e,1:i_e) + r * (s * s');
        else
            %'extended'
            
            if (i_e == 0)
                alpha(i_e+1) = q;
                C(i_e+1,i_e+1) = r;
                Q(i_e+1,i_e+1) = gamma_x^-1;
            else
                s = [C(1:i_e,1:i_e)*k_x; 1];
                C(1:i_e+1,1:i_e+1) = C(1:i_e+1,1:i_e+1) + r * (s * s');
                alpha(1:i_e+1) = alpha(1:i_e+1) + q * s;         
                s2 = [e_x;-1];
                Q(1:i_e+1, 1:i_e+1) = Q(1:i_e+1, 1:i_e+1) + gamma_x^-1 * (s2 * s2'); 
            end

            I = [I i_p];
            i_e = i_e + 1;
        end
        
        % exceed sparsity level
        if i_e > d - 1         
           %'clean' 
           
           epsilon = abs(alpha) ./ diag(Q);
           [~, idx] = min(epsilon);
           I = [I(1:idx-1) I(idx+1:d)];
           i_e = d - 1;
           
           sel = false(1,d);
           sel(idx) = true;        
           
           alpha_s = alpha(sel);
           c_s = C(sel, sel);
           q_s = Q(sel, sel);
           C_s = C(~sel, sel);
           Q_s = Q(~sel, sel);
           
           alpha_t = alpha(~sel);
           C_t = C(~sel, ~sel);
           Q_t = Q(~sel, ~sel);
                   
           alpha(1:i_e) = alpha_t - alpha_s * Q_s / q_s;
           C(1:i_e,1:i_e) = C_t + c_s * (Q_s * Q_s') / (q_s * q_s) - ( Q_s * C_s' + C_s * Q_s' )/q_s;
           Q(1:i_e,1:i_e) = Q_t - (Q_s * Q_s') / q_s;
           
           %alpha(1:i_e) = alpha_t - alpha_s * C_s / c_s;
           %C(1:i_e,1:i_e) = C_t - (C_s * C_s') / c_s;
           
           alpha(i_e+1) = 0;
           C(:,i_e+1) = 0;
           C(i_e+1,:) = 0;
           Q(:,i_e+1) = 0;
           Q(i_e+1,:) = 0;     
        end
        
    end
    
   
    X_I = X_train(I,:);
    K_train_I = ker_matrix(X_train, X_I, params);
    K_test_I = ker_matrix(X_test, X_I, params);
    diag_K = zeros(train_len,1);
    diag_K_test = zeros(test_len,1);
    for i = 1:train_len
        diag_K(i) = ker_matrix(X_train(i,:), X_train(i,:), params);
    end
    for i = 1:test_len
        diag_K_test(i) = ker_matrix(X_test(i,:), X_test(i,:), params);
    end
    
    
    C_p = C(1:i_e,1:i_e);
    alpha_p = alpha(1:i_e);
    
    [ mu_tr, sigma_tr, zo_err_tr, gibbs_err_tr ] = get_err_dir( diag_K, K_train_I, C_p, alpha_p, Y_train);
    [ mu_te, sigma_te, zo_err_te, gibbs_err_te ] = get_err_dir( diag_K_test, K_test_I, C_p, alpha_p, Y_test);
    
    K_I_I = ker_matrix(X_train(I,:), X_train(I,:), params);
    B = eye(i_e) - C_p * K_I_I;
    

    
    
    %% present the results
    zo_train_err = sum(zo_err_tr) / train_len
    zo_test_err = sum(zo_err_te) / test_len
    gibbs_train_err = sum(gibbs_err_tr) / train_len
    gibbs_test_err = sum(gibbs_err_te) / test_len
    RE = 0.5*(-log(det(B)) + trace(B) + alpha_p' * K_I_I * alpha_p - d);
    ub = inv_re(gibbs_train_err, (RE + log((train_len+1)/0.5)) / train_len)
 
    [mu_te sigma_te]
    
    [err_mv, coverage] = selective(mu_te, sigma_te, Y_test);
    [err_m, coverage] = selective(mu_te, ones(test_len,1), Y_test);
    
    
end


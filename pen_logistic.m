function [coverage, err_mv, err_m, gibbs_test_err, ub] = pen_logistic(data_train, data_test, params)
% function [coverage, err_mv, err_m, gibbs_test_err, ub] = pen_logistic(data_train, data_test, params)
% the traditional logistic + Gaussian process implementation
% based on training examples, and testing examples, get the RC curve and
% the verification of the Gibbs test error and its estimated upper bound

    train_len = params.train_len;
    test_len = params.test_len;

    X_train = data_train.X;
    Y_train = data_train.Y;
    X_test = data_test.X;
    Y_test = data_test.Y;

    K = ker_matrix(X_train, X_train, params);
    c = (Y_train'+1)/2;

    %% training
%     cvx_begin
%         variables b(train_len)
%         maximize ( -quad_form(b, 0.5*K) + c*K*b - sum(log_sum_exp([zeros(1,train_len); b'*K])))
%     cvx_end

    b_0 = zeros(train_len,1);
    f = @(b)logistic_loss(b, K, Y_train);
    options = optimset('LargeScale','on');
    options = optimset('GradObj','on');
    [b, fval] = fminunc(f, b_0, options);
    fval


    %% post processing
    K_test_train = ker_matrix(X_test, X_train, params);
    K_test_test = ker_matrix(X_test, X_test, params);

    train_pred = (K*b).*sign(Y_train);
    test_pred = (K_test_train*b).*sign(Y_test);


    u_hat = K*b;
    b_2 = Y_train .* (1+exp(Y_train.*u_hat)).^-1;
    w = ((1+exp(Y_train.*u_hat)) .* (1+exp(-Y_train.*u_hat))).^-1;
    A = eye(train_len) + diag(w.^0.5)*K*diag(w.^0.5);
    M = diag(w.^0.5)*A*diag(w.^0.5);
    SI_K = diag(w.^0.5)*A*diag(w.^-0.5);
    L = chol(A,'lower');

    [ mu_tr, sigma_tr, zo_err_tr, gibbs_err_tr ] = get_err( diag(K), K, L, w, b, Y_train);
    [ mu_te, sigma_te, zo_err_te, gibbs_err_te ] = get_err( diag(K_test_test), K_test_train, L, w, b, Y_test);


    %% present the results
    zo_train_err = sum(zo_err_tr) / train_len
    zo_test_err = sum(zo_err_te) / test_len
    gibbs_train_err = sum(gibbs_err_tr) / train_len
    gibbs_test_err = sum(gibbs_err_te) / test_len
    RE = (log(det(A)) + trace(inv(A)) + b'*K*b - train_len)/2;
    ub = inv_re(gibbs_train_err, (RE + log((train_len+1)/0.5)) / train_len);

    [mu_te sigma_te]
    
    [err_mv, coverage] = selective(mu_te, sigma_te, Y_test);
    [err_m, coverage] = selective(mu_te, ones(test_len,1), Y_test);

end
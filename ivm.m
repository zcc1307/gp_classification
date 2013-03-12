function [coverage, err_mv, err_m, gibbs_test_err, ub, internal_val] = ivm(data_train, data_test, params)
% function [coverage, err_mv, err_m, gibbs_test_err, ub] = ivm(data_train, data_test, params)
% the IVM implementation
% based on training examples, and testing examples, get the RC curve and
% the verification of the Gibbs test error and its estimated upper bound


    %% basic settings
    train_len = params.train_len;
    test_len = params.test_len;

    X_train = data_train.X;
    Y_train = data_train.Y;
    X_test = data_test.X;
    Y_test = data_test.Y;


    I = false(train_len,1);
    d = 30;
    b = 0;
    lambda = 1;

    nu = zeros(train_len,1);
    g = zeros(train_len,1);
    G = zeros(train_len,1);
    m = zeros(train_len,1);
    beta = zeros(train_len,1);
    mu = zeros(train_len,1);
    u = zeros(train_len,1);
    c = zeros(train_len,1);
    Delta = zeros(train_len,1);
    zeta = zeros(train_len,1);
    for i = 1:train_len
        zeta(i) = ker_matrix(X_train(i,:), X_train(i,:),params);
    end

    M = zeros(d,train_len);
    L = zeros(d,d);



    %% training 

    for round = 1:d

        %round

        % calculating the nu values
        for i = 1:train_len
            if ~I(i)
                c(i) = Y_train(i) / sqrt(lambda^(-2) + zeta(i));          
                u(i) = c(i)*(mu(i) + b);
                g(i) = c(i) * normpdf(u(i)) / normcdf(u(i));
                G(i) = -0.5 * g(i) * u(i) * c(i);
                nu(i) = g(i) * g(i) - 2 * G(i);
                m(i) = g(i) / nu(i) + mu(i);
                beta(i) = nu(i) / (1 - nu(i) * zeta(i));
            end
        end


        % selecting the data point which has the maximum entropy decrease
        for i = 1:train_len
            if I(i)
                Delta(i) = -100000;
            else
                Delta(i) = -0.5 * log(1 - zeta(i)*nu(i));
            end
        end

        if round < 3
            while true
                idx_max = ceil(rand * train_len);
                if ~I(idx_max)
                    break
                end
            end
        else
            [~, idx_max] = max(Delta);
        end

        %idx_max

        I(idx_max) = true;


        %s = Sigma(:,idx_max);
        %mu = mu + g(idx_max) * s;


        if round == 1
            s = ker_matrix(X_train, X_train(idx_max,:),params);
        else
            s = ker_matrix(X_train, X_train(idx_max,:),params) - M(1:round-1,:)' * M(1:round-1,idx_max);
        end

        %[s Sigma(:,idx_max)]

        if round == 1
            L(1,1) = (1 - nu(idx_max)*zeta(idx_max))^(-0.5);
        else 
            L(round,1:round) = (1 - nu(idx_max)*zeta(idx_max))^(-0.5)*[nu(idx_max)^0.5 * M(1:round-1,idx_max)' 1];
        end


        zeta = zeta - nu(idx_max) * (s.^2);
        mu = mu + g(idx_max) * s;
        M(round,:) = sqrt(nu(idx_max)) * s';
        %max(abs(K - Sigma - M(1:round,:)'*M(1:round,:)))
    end


    %% post processing
    X_I = X_train(I,:);
    K_I = ker_matrix(X_I, X_I, params);
    w = beta(I);
    B = eye(d) + diag(w.^0.5)*K_I*diag(w.^0.5);
    L = chol(B,'lower');

    b = diag(w.^0.5)*(B\(diag(w.^0.5)*m(I)));
    %b = (K_I + diag(w.^-1))\m(I);

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


    [ mu_tr, sigma_tr, zo_err_tr, gibbs_err_tr ] = get_err( diag_K, K_train_I, L, w, b, Y_train);
    [ mu_te, sigma_te, zo_err_te, gibbs_err_te ] = get_err( diag_K_test, K_test_I, L, w, b, Y_test);


    %% present the results
    zo_train_err = sum(zo_err_tr) / train_len
    zo_test_err = sum(zo_err_te) / test_len
    gibbs_train_err = sum(gibbs_err_tr) / train_len
    gibbs_test_err = sum(gibbs_err_te) / test_len
    RE = (log(det(B)) + trace(inv(B)) + b'*K_I*b - d)/2;
    ub = inv_re(gibbs_train_err, (RE + log((train_len+1)/0.5)) / train_len)
 
    [mu_te sigma_te]
    
    [err_mv, coverage] = selective(mu_te, sigma_te, Y_test);
    [err_m, coverage] = selective(mu_te, ones(test_len,1), Y_test);

    internal_val = [];
    internal_val.mu = mu;
    internal_val.zeta = zeta;
    internal_val.m = m;
    internal_val.beta = beta;
    internal_val.I = I;
    
    
end
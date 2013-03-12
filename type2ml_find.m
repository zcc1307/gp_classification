%% the test script for comparison of IVM and SVM
params = [];
params.train_len = 300;
params.test_len = 200;
params.filename = 'pima2.mat';
params.pca = 'pca';
params.kernel = 'Gauss';
params.task = '2v3';
params.merge_rate = 0.5;
params.reduced_dim = 20;
params.rounds = 100;
params.svm_C_asc = 10000;

% kernel parameter
params.kernel_parameter1 = 1;
params.kernel_parameter2 = 1;
% noise parameter
params.b = 0;
params.lambda = 1;


options = optimset('LargeScale','on');
%options = optimset('GradObj','on');
options = optimset('GradObj','off');

[data_train, data_test] = gen_data_from_len(params);


for i = 1:params.rounds
    
    
    d = size(data_train.X, 2); 

    [~, ~, ~, te, ~, internal_val] = ivm(data_train, data_test, params);
    te
    mu = internal_val.mu; 
    zeta = internal_val.zeta; 
    m = internal_val.m;
    beta = internal_val.beta;
    I = internal_val.I;
    
    m(I)
    expect_K_I_I = m(I)*m(I)'
    beta(I)
    data_train.X(I,:)
    K_I_I = ker_matrix(data_train.X(I,:), data_train.X(I,:), params) %+ diag(beta(I).^-1)
    
    
    f = @(w_C)log_gauss_kerpar(m(I), beta(I), data_train.X(I,:), w_C);
    
    [w_C, ~] = fmincon(f, [1; 1], -eye(2), [-1; -1]);
    
%     f = @(b_lambda)log_z_noisepar(data_train.Y, mu, zeta, b_lambda);
%     
%     [b_lambda, ~] = fminunc(f, [0; 1], options);
%     
%     [w_C;b_lambda]

%     if (w_C(1) < 0)
%         w_C(1) = d;
%     end
%     if (w_C(2) < 0)
%         w_C(2) = 1;
%     end
    
%     params.b = b_lambda(1);
%     params.labmda = b_lambda(2);
    params.kernel_parameter1 = sqrt(d/w_C(1));
    params.kernel_parameter2 = w_C(2);
    
%     [w_C;b_lambda]
    w_C
    
    
    
end





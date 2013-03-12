%% the test script for comparison of IVM and SVM
params = [];
params.train_len = 300;
params.test_len = 200;
params.filename = 'mnist_all.mat';
params.pca = 'pca';
params.kernel = 'Gauss';
params.task = '2v3';
params.merge_rate = 0.5;
params.reduced_dim = 20;
params.kernel_parameter1 = 1;
params.kernel_parameter2 = 1;
params.rounds = 20;
params.svm_C_asc = 10000;

w_cand = [0.25 0.5 1 2 4 8 16 32 64 128 256];
C_cand = [0.25 0.5 1 2 4 8 16 32 64 128 256];

errs = zeros(11,11);



for i = 1:11
    for j = 1:11
        for k = 1:params.rounds
            [data_train, data_test] = gen_data_from_len(params);
            d = size(data_train.X, 2);
            w = w_cand(i);
            C = C_cand(j);
            
            params.kernel_parameter1 = sqrt(d/w);
            params.kernel_parameter2 = C;
            
            %[coverage, err_mv, err_m, gt(i), ub(i)] = pen_logistic(data_train, data_test, params);
            [~, err_mv, ~, gt, ~] = ivm(data_train, data_test, params); 
            %[coverage, err_mv, err_m, gt(i), ub(i)] = sogp(data_train, data_test, params);
            %[coverage_s, err_s] = margin_selective(data_train, data_test, params);

            errs(i,j) = errs(i,j) + err_mv(size(err_mv,2)) / params.rounds
        end
    end
end

errs
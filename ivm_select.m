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
params.rounds = 1;
params.svm_C_asc = 10000;


err_mv_avg = zeros(1, params.test_len);
err_m_avg = zeros(1, params.test_len);
err_s_avg = zeros(1, params.test_len);
gt = zeros(1, params.test_len);
ub = zeros(1, params.test_len);


for i = 1:params.rounds
    [data_train, data_test] = gen_data_from_len(params);
    [coverage, err_mv, err_m, gt(i), ub(i)] = pen_logistic(data_train, data_test, params);
    %[coverage, err_mv, err_m, gt(i), ub(i)] = ivm(data_train, data_test, params); 
    %[coverage, err_mv, err_m, gt(i), ub(i)] = sogp(data_train, data_test, params);
    [coverage_s, err_s] = margin_selective(data_train, data_test, params);

    err_mv_avg = err_mv_avg + err_mv / params.rounds;
    err_m_avg = err_m_avg + err_m / params.rounds;
    err_s_avg = err_s_avg + err_s / params.rounds;
end

figure;
hold on;
plot(coverage, err_m_avg, 'b');
plot(coverage, err_mv_avg, 'r');
plot(coverage, err_s_avg, 'k');
set(gca, 'XLim', [0, get(gca, 'XLim') * [0; 1]])
set(gca, 'YLim', [0, get(gca, 'YLim') * [0; 1]])
legend('GPC w/o var','GPC w/ var','SVM')

figure;
hold on;
plot(gt, ub, '.b')
set(gca, 'XLim', [0, get(gca, 'XLim') * [0; 1]])
set(gca, 'YLim', [0, get(gca, 'YLim') * [0; 1]])
plot(0:0.01:1,0:0.01:1);
function [err coverage] = selective(mu, sigma, Y)
%function [err coverage] = selective(mu, sigma, Y)
%given the mean and stddev, sort the examples according to
%abs(mean/stddev), then outuput the RC curve

    len = size(Y,1);
    conf = abs(mu)./sigma;
    
    sorted = sortrows([conf mu Y], [-1]);

    coverage = 1:len;
    err = zeros(1,len);
    for i = 1:len
        if i == 1
            err(i) = (sorted(i,2)*sorted(i,3)<=0);
        else
            err(i) = err(i-1) + (sorted(i,2)*sorted(i,3)<=0);
        end
    end
    err = err ./ coverage; 
    coverage = coverage / len;   

end

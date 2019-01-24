%
% ENEL 525 Lab 4.
%     Design a predictor based on feed forward neural network with
% multiple neurons using the Backpropogation learning algorithm.
%

% Load the given input data.
load('LAB4_p2.mat')

% Randomly initialize weights and biases.
W1 = rand(5,2);
W2 = rand(1,5);
b1 = rand(5,1);
b2 = rand();

lr = 0.05;    % alpha
et = 0.00002; % convergence threshold

iter = 1;
MSE = 1;
k = 1;
err = [];

% Backprop algorithm
while MSE(iter) >= et
    a1 = logsig(W1 * [p(k+1) p(k)]' + b1);
    a2 = W2 * a1 + b2;
    err(k) = p(k+2) - a2;

    vals = [a1(1)*(1-a1(1)) a1(2)*(1-a1(2)) a1(3)*(1-a1(3)) a1(4)*(1-a1(4)) a1(5)*(1-a1(5))];
    F1 = diag(vals);
    F2 = 1;
    s2 = -2 * F2*err(k);
    s1 = F1*W2'*s2;

    W2 = W2 - lr*s2*(a1)';
    b2 = b2 - lr*s2;
    W1 = W1 - lr*s1*[p(k+1) p(k)];
    b1 = b1 - lr*s1;

    k = k + 1;
    if k > 168
        k = 1;
        iter = iter + 1;
        MSE(iter) = mse(err);
    end
end

results = [];
for i = 1:10
    a1 = logsig(W1 * [p(i+1+169) p(i+169)]' + b1);
    results(i) = W2 * a1 + b2;
end

% Plot learning rate
figure, semilogy(MSE)
hold on;

% Plot predicted and expected results.
figure, plot(results);
hold on;
plot(p(171:180));


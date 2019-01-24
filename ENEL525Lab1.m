%
% ENEL 525 Lab 1.
%     Classify input vectors using the perceptron learning rule.
%


% Initial conditions
p1 = [ 1; 1]; t1 = [0;0];
p2 = [ 0; 1]; t2 = [0;1];
p3 = [-1; 2]; t3 = [1;0];
p4 = [-1;-1]; t4 = [1;1];
p = [p1 p2 p3 p4]; % The input vector
t = [t1 t2 t3 t4]; % The target vector

% Iniital weight and bias
w0 = [1 0; 0 1];
b0 = [1;1];

flags = [1,1,1,1];
while(any(flags))
    flags = [1, 1, 1, 1];
    for k = 1 : 4
        curP = [p(1,k); p(2,k)];
        curT = [t(1,k); t(2,k)];

        a = hardlim(w0 * curP + b0);
        e = curT - a;

        if(e(1,1) == 0 && e(2,1) == 0)
            flags(k) = 0;
        end
        w0 = w0 + e * (transpose(curP));
        b0 = b0 + e;
    end
end

% Display results
disp(w0);
disp(b0);

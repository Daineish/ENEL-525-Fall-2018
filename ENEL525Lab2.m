%
% ENEL 525 Lab 2.
%     Use the Hebbian Learning Rule and Pseudo-Inverse Rule to
% perform simple face recognition.
%


% Read images.
image0 = double(imread('AudreyHepburn.jpg'));
image1 = double(rgb2gray(imread('MrWhite.jpg')));
image2 = double(rgb2gray(imread('SheldonCooper.jpg')));
image3 = double(rgb2gray(imread('TaylorSwift.jpg')));
image4 = double(rgb2gray(imread('BillGates.jpg')));

%%%%% Original Images %%%%%
image0orig = reshape(image0, numel(image0), 1);
image1orig = reshape(image1, numel(image1), 1);
image2orig = reshape(image2, numel(image2), 1);
image3orig = reshape(image3, numel(image3), 1);
image4orig = reshape(image4, numel(image4), 1);

%%%%% Normalized Original Images %%%%%
image0orignorm = normc(image0orig);
image1orignorm = normc(image1orig);
image2orignorm = normc(image2orig);
image3orignorm = normc(image3orig);
image4orignorm = normc(image4orig);

%%%%% Noisy Images %%%%%
image0noise = awgn(image0orig, 20, 'measured');
image1noise = awgn(image1orig, 20, 'measured');
image2noise = awgn(image2orig, 20, 'measured');
image3noise = awgn(image3orig, 20, 'measured');
image4noise = awgn(image4orig, 20, 'measured');

%%%%% Normalized Noisy Images %%%%%
image0noisenorm = normc(image0noise);
image1noisenorm = normc(image1noise);
image2noisenorm = normc(image2noise);
image3noisenorm = normc(image3noise);
image4noisenorm = normc(image4noise);

% figure, imshow(reshape(image0noise, [75 75]), [0 255]);
% figure, imshow(reshape(image1noise, [75 75]), [0 255]);
% figure, imshow(reshape(image2noise, [75 75]), [0 255]);
% figure, imshow(reshape(image3noise, [75 75]), [0 255]);
% figure, imshow(reshape(image4noise, [75 75]), [0 255]);

T = [image0orig  image1orig  image2orig  image3orig  image4orig];
P = [image0orignorm image1orignorm image2orignorm image3orignorm image4orignorm];

%%%%% Hebbian Learning %%%%%
Wh = T*(transpose(P));

n0h = Wh * image0noisenorm;
n1h = Wh * image1noisenorm;
n2h = Wh * image2noisenorm;
n3h = Wh * image3noisenorm;
n4h = Wh * image4noisenorm;

% figure, imshow(mat2gray(reshape(n0h, [75 75])));
% figure, imshow(mat2gray(reshape(n1h, [75 75])));
% figure, imshow(mat2gray(reshape(n2h, [75 75])));
% figure, imshow(mat2gray(reshape(n3h, [75 75])));
% figure, imshow(mat2gray(reshape(n4h, [75 75])));

% Correlation table
fprintf('          |Output 1|Output 2|Output 3|Output 4|Output 5\n');
fprintf('-------------------------------------------------------\n');
fprintf('Pattern 1 |%*f|%f|%f|%f|%f\n', 8,abs(corr2(image0orig,n0h)),abs(corr2(image0orig,n1h)),abs(corr2(image0orig,n2h)),abs(corr2(image0orig,n3h)),abs(corr2(image0orig,n4h)));
fprintf('Pattern 2 |%*f|%f|%f|%f|%f\n', 8,abs(corr2(image1orig,n0h)),abs(corr2(image1orig,n1h)),abs(corr2(image1orig,n2h)),abs(corr2(image1orig,n3h)),abs(corr2(image1orig,n4h)));
fprintf('Pattern 3 |%*f|%f|%f|%f|%f\n', 8,abs(corr2(image2orig,n0h)),abs(corr2(image2orig,n1h)),abs(corr2(image2orig,n2h)),abs(corr2(image2orig,n3h)),abs(corr2(image2orig,n4h)));
fprintf('Pattern 4 |%*f|%f|%f|%f|%f\n', 8,abs(corr2(image3orig,n0h)),abs(corr2(image3orig,n1h)),abs(corr2(image3orig,n2h)),abs(corr2(image3orig,n3h)),abs(corr2(image3orig,n4h)));
fprintf('Pattern 5 |%*f|%f|%f|%f|%f\n', 8,abs(corr2(image4orig,n0h)),abs(corr2(image4orig,n1h)),abs(corr2(image4orig,n2h)),abs(corr2(image4orig,n3h)),abs(corr2(image4orig,n4h)));
fprintf('\n\n');


%%%%% Pseudo-Inverse %%%%%
Pinv = inv(transpose(P)*P)*transpose(P);
Wp = T*Pinv;
n0i = Wp * image0noisenorm;
n1i = Wp * image1noisenorm;
n2i = Wp * image2noisenorm;
n3i = Wp * image3noisenorm;
n4i = Wp * image4noisenorm;

figure, imshow(reshape(n0i, [75 75]), [0 255]);
figure, imshow(reshape(n1i, [75 75]), [0 255]);
figure, imshow(reshape(n2i, [75 75]), [0 255]);
figure, imshow(reshape(n3i, [75 75]), [0 255]);
figure, imshow(reshape(n4i, [75 75]), [0 255]);

% Correlation table
fprintf('          |Output 1|Output 2|Output 3|Output 4|Output 5\n');
fprintf('-------------------------------------------------------\n');
fprintf('Pattern 1 |%*f|%f|%f|%f|%f\n', 8,abs(corr2(image0orig,n0i)),abs(corr2(image0orig,n1i)),abs(corr2(image0orig,n2i)),abs(corr2(image0orig,n3i)),abs(corr2(image0orig,n4i)));
fprintf('Pattern 2 |%*f|%f|%f|%f|%f\n', 8,abs(corr2(image1orig,n0i)),abs(corr2(image1orig,n1i)),abs(corr2(image1orig,n2i)),abs(corr2(image1orig,n3i)),abs(corr2(image1orig,n4i)));
fprintf('Pattern 3 |%*f|%f|%f|%f|%f\n', 8,abs(corr2(image2orig,n0i)),abs(corr2(image2orig,n1i)),abs(corr2(image2orig,n2i)),abs(corr2(image2orig,n3i)),abs(corr2(image2orig,n4i)));
fprintf('Pattern 4 |%*f|%f|%f|%f|%f\n', 8,abs(corr2(image3orig,n0i)),abs(corr2(image3orig,n1i)),abs(corr2(image3orig,n2i)),abs(corr2(image3orig,n3i)),abs(corr2(image3orig,n4i)));
fprintf('Pattern 5 |%*f|%f|%f|%f|%f\n', 8,abs(corr2(image4orig,n0i)),abs(corr2(image4orig,n1i)),abs(corr2(image4orig,n2i)),abs(corr2(image4orig,n3i)),abs(corr2(image4orig,n4i)));

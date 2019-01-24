%
% ENEL 525 Lab 3.
%     Design a single layer network with the multiple neurons and
% the linear transfer function using LMS supervised learning algorithm
% to recognize characters.
%


% Read all letters.
N = 26;
ima = double(imread('char1_a.bmp')); ima = reshape(ima, numel(ima), 1);
imb = double(imread('char1_b.bmp')); imb = reshape(imb, numel(imb), 1);
imc = double(imread('char1_c.bmp')); imc = reshape(imc, numel(imc), 1);
imd = double(imread('char1_d.bmp')); imd = reshape(imd, numel(imd), 1);
ime = double(imread('char1_e.bmp')); ime = reshape(ime, numel(ime), 1);
imf = double(imread('char1_f.bmp')); imf = reshape(imf, numel(imf), 1);
img = double(imread('char1_g.bmp')); img = reshape(img, numel(img), 1);
imh = double(imread('char1_h.bmp')); imh = reshape(imh, numel(imh), 1);
imi = double(imread('char1_i.bmp')); imi = reshape(imi, numel(imi), 1);
imj = double(imread('char1_j.bmp')); imj = reshape(imj, numel(imj), 1);
imk = double(imread('char1_k.bmp')); imk = reshape(imk, numel(imk), 1);
iml = double(imread('char1_l.bmp')); iml = reshape(iml, numel(iml), 1);
imm = double(imread('char1_m.bmp')); imm = reshape(imm, numel(imm), 1);
imn = double(imread('char1_n.bmp')); imn = reshape(imn, numel(imn), 1);
imo = double(imread('char1_o.bmp')); imo = reshape(imo, numel(imo), 1);
imp = double(imread('char1_p.bmp')); imp = reshape(imp, numel(imp), 1);
imq = double(imread('char1_q.bmp')); imq = reshape(imq, numel(imq), 1);
imr = double(imread('char1_r.bmp')); imr = reshape(imr, numel(imr), 1);
ims = double(imread('char1_s.bmp')); ims = reshape(ims, numel(ims), 1);
imt = double(imread('char1_t.bmp')); imt = reshape(imt, numel(imt), 1);
imu = double(imread('char1_u.bmp')); imu = reshape(imu, numel(imu), 1);
imv = double(imread('char1_v.bmp')); imv = reshape(imv, numel(imv), 1);
imw = double(imread('char1_w.bmp')); imw = reshape(imw, numel(imw), 1);
imx = double(imread('char1_x.bmp')); imx = reshape(imx, numel(imx), 1);
imy = double(imread('char1_y.bmp')); imy = reshape(imy, numel(imy), 1);
imz = double(imread('char1_z.bmp')); imz = reshape(imz, numel(imz), 1);

%%%%% Normalized Original Images%%%%%
imna = normc(ima); imnb = normc(imb); imnc = normc(imc); imnd = normc(imd); imne = normc(ime);
imnf = normc(imf); imng = normc(img); imnh = normc(imh); imni = normc(imi); imnj = normc(imj);
imnk = normc(imk); imnl = normc(iml); imnm = normc(imm); imnn = normc(imn); imno = normc(imo);
imnp = normc(imp); imnq = normc(imq); imnr = normc(imr); imns = normc(ims); imnt = normc(imt);
imnu = normc(imu); imnv = normc(imv); imnw = normc(imw); imnx = normc(imx); imny = normc(imy);
imnz = normc(imz);

P = [imna imnb imnc imnd imne imnf imng imnh imni imnj imnk imnl imnm imnn imno imnp imnq imnr imns imnt imnu imnv imnw imnx imny imnz];
T = P;

W = zeros(numel(ima), numel(ima));
b = zeros(numel(ima, 1));
et = 10e-06;
lr = 0.04;
errors = [1];

i = 1;
while errors(i) >= et
    err = zeros(400,26);
    for k = 1 : N
        err(:,k) = T(:,k) - (W*P(:,k) + b);
        W = W + 2*lr*err(:,k)*transpose(P(:,k));
        b = b + 2*lr*err(:,k);
    end
    errors(i+1) = mse(err);
    i = i + 1;
end

% Show learning curve
figure, semilogy(errors)
%figure, imshow(reshape(W*imm, [20 20]), [0 255])

% Correlation table
letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'];
fprintf(' |a        b        b        d        e        f        g        h        i        j        k        l        m        n        o        p        q        r        s        t        u        v        w        x        y        z\n')
fprintf('--------------------------------------------------------------------------------------------------------------------------------\n')
for i = 1: 26
    fprintf('%c', letters(i))
    curimg = P(:,i);
    fprintf('|%*f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f|%f\n',4,abs(corr2(W*curimg,ima)),abs(corr2(W*curimg,imb)),abs(corr2(W*curimg,imc)),abs(corr2(W*curimg,imd)),abs(corr2(W*curimg,ime)),abs(corr2(W*curimg,imf)),abs(corr2(W*curimg,img)),abs(corr2(W*curimg,imh)),abs(corr2(W*curimg,imi)),abs(corr2(W*curimg,imj)),abs(corr2(W*curimg,imk)),abs(corr2(W*curimg,iml)),abs(corr2(W*curimg,imm)),abs(corr2(W*curimg,imn)),abs(corr2(W*curimg,imo)),abs(corr2(W*curimg,imp)),abs(corr2(W*curimg,imq)),abs(corr2(W*curimg,imr)),abs(corr2(W*curimg,ims)),abs(corr2(W*curimg,imt)),abs(corr2(W*curimg,imu)),abs(corr2(W*curimg,imv)),abs(corr2(W*curimg,imw)),abs(corr2(W*curimg,imx)),abs(corr2(W*curimg,imy)),abs(corr2(W*curimg,imz)))
end

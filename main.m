%% Linear inference
addpath('.\Assignment1');
X=[1,1,1,0;
   1,1,1,0;
   1,0,1,1;
   1,1,1,1;
   0,0,0,0;
   0,1,1,1;
   ];
t=[1;0;1;0;0;1];
lamda=0.5;
w=linear_inference(t,X,lamda);
y1=[1,1,0,0]*w;
%% Gaussian kernel regression 
addpath('.\Assignment2');
sigma=0.5; lamda=0.4;
K=gram_matrix(X,X,sigma);
alpha=(K+lamda.*eye(size(X,1)))\t;
y3=0;
for i=1:size(X,1)
  y3=y3+alpha(i)'*gaussian_kernel(X(i,:),[1,1,0,0],sigma);
end
%% Gaussian process Regression
addpath('.\Assignment3');
theta=ones(4,1); precision=0.001;
[K,C]=gaussian_process_gram(X,theta,precision,t);
mu=gaussian_process_kernel(X,[1,1,0,0],theta)'*pinv(C)*t;
xn=[X;[1,1,0,0]];
c=gaussian_process_kernel([1,1,0,0],[1,1,0,0],theta)+precision;
sigma=c-gaussian_process_kernel(X,[1,1,0,0],theta)'*pinv(C)*gaussian_process_kernel(X,[1,1,0,0],theta);
%p=mvnpdf([1,1,0,0,],mu,sigma);
%% SVM 
addpath('.\Assignment2');
X=[];Y=[];
for i=1:3000
    if(labelMat(i,1)==1)
        X=[X;featureMat(i,:)];
        Y=[Y,1];
    elseif (labelMat(i,2)==1)
        X=[X;featureMat(i,:)];
        Y=[Y,-1];
    end
end
Y=Y';
%training
tic
sigma=0.5;
K=gram_matrix(X,X,sigma);
for i=1:size(X,1)
    for j=1:i
        
    end
end
N=size(X,1);
C=1e-3;
cvx_begin
    cvx_precision best
    variable a_m(N);
    minimize (0.5.*quad_form(Y.*a_m,K) - ones(N,1)'*(a_m));
    subject to
        a_m >= 0;
        Y'*(a_m) == 0;
        a_m <= C;
cvx_end
a_m(a_m<10^-5)=0;
X_m = X(a_m>0,:);
Y_m = Y(a_m>0);
a_m = a_m(a_m>0);
K=gram_matrix(X,X_m,sigma);
b_m = mean(Y-K*(a_m.*Y_m));
toc
time=toc
% prediction
XT=[];YT=[];
for i=3001:size(featureMat,1)
    if(labelMat(i,1)==1)
        XT=[XT;featureMat(i,:)];
        YT=[YT,1];
    elseif labelMat(i,2)==1
        XT=[XT;featureMat(i,:)];
        YT=[YT,-1];
    end
end
YT=YT';
K1=gram_matrix(XT,X_m,sigma);
YP = K1*(a_m.*Y_m)-b_m>0;
YT(YT==-1)=0;
YT=YT-YP;
accuracy=(size(YT,1)-sum(sum(YT~=0))/2)/size(YT,1);
%% 1-layer ELM
X=[1,1,1,0;
   1,1,1,0;
   1,0,1,1;
   1,1,1,1;
   0,0,0,0;
   0,1,1,1;
   ];
T=[1;0;1;0;0;1];
neurons=3; input_weights1=rand(neurons,size(X,2))*2-1;
H1=radbas(input_weights1*X'); %LT and NLT
B=pinv(H1*H1'+0.05*ones(size(H1*H1')))*H1*T;
% prediction
H1s=radbas(input_weights1*[1,0,0,1]');
Yls=(H1s'*B);
%% 3-layer ELM
X=[1,1,1,0;
   1,1,1,0;
   1,0,1,1;
   1,1,1,1;
   0,0,0,0;
   0,1,1,1;
   ];
T=[1;0;1;0;0;1];
neurons=3; input_weights1=rand(neurons,size(X,2))*2-1;
input_weights2=rand(2*neurons,neurons)*2-1;% Second layer has more neurons.
input_weights3=rand(neurons,2*neurons)*2-1;
H1=radbas(input_weights1*X'); %LT and NLT
H2=radbas(input_weights2*H1); %LT and NLT
H3=radbas(input_weights3*H2); %LT and NLT
B=pinv(H3*H3'+0.05*ones(size(H3*H3')))*H3*T;
% prediction
H1s=radbas(input_weights1*[1,0,0,1]');
H2s=radbas(input_weights2*H1s);
H3s=radbas(input_weights3*H2s);
Ys=(H3s'*B);
%% Sparse Representation
addpath('.\Assignment4');
re1=zeros(100,1); re2=zeros(100,1); re3=zeros(100,1); 
X=[rand(5,1); zeros(495,1)];
for m=1:100
for repe=1:30
A=rand(1,500);
b=A*X;
xx1(:,repe)=orth_match(b,A,m);
xx2(:,repe)=sub_pursuit(b,A,m);
xx3(:,repe)=basis_pursuit(b,A,10e-4);
end
re1(m)=calNMSE(mean(xx1,2),X);
re2(m)=calNMSE(mean(xx2,2),X);
re3(m)=calNMSE(mean(xx3,2),X);
%test=O*exp(W*[1,1,0,0]'+B);
end
%% Plot and Comparison of SParse Representation
xlen=1:100;
plot(xlen/500,re1(xlen));
hold on
plot(1./xlen,re2(xlen));
plot(xlen/500,re3(xlen));
legend('OMP','SP','BP')
hold off
title('The Reconstruction Error versus Sparsity using OMP and BP','FontSize', 24)
xlabel('Sparsity level n/m','FontSize', 16)
ylabel('Normalized Mean Square Error','FontSize', 16)
%% Markov Random Field De-noising
clear;clc;
addpath('.\Assignment5');
addpath('.\noise');
img=imread('.\Assignment5\lena.bmp');
img=im2double(img);
figure; imshow(img);
%additive white noise to original img, evaluation of PSNR, WPSNR, 
noise_img=test_awgn(img,0.01,1);
%noise_img=img;
%noise_img(1:5:end,1:5:end)=test_awgn(img(1:5:end,1:5:end),0.01,1);
noise_wpsnr=WPSNR(noise_img,img);
noise_psnr=PSNR(noise_img,img);
noise_simm=ssim(noise_img,img);
noise_nmse=mse(noise_img,img);
figure; imshow(noise_img);
para=[1,0.5,1,std(std(noise_img))];
%energy of the noise image
hidd=noise_img;
total_energy=img_energy(noise_img,noise_img,para);
fprintf('total energy before denoising:%f \n',total_energy);
%ICM de-noising
iter=1;it=1;
energy_old=total_energy;
while it<iter+1 && total_energy<=energy_old
   energy_old=total_energy;
   for i=1 :size(hidd,1)
      for j=1:size(hidd,2)
          %[hidd,total_energy]=icm_update(noise_img,hidd,i,j,total_energy,para);
          [hidd,total_energy]=deri_update(noise_img,hidd,i,j,total_energy,para);
      end
   end
   it=it+1;
   fprintf('total energy after denoising:%f \n',total_energy);
end
hidd_wpsnr=WPSNR(hidd,img);
hidd_psnr=PSNR(hidd,img);
hidd_simm=ssim(hidd,img);
hidd_nmse=mse(hidd,img);
figure; imshow(hidd);
%% SP/MS De-noising Algorithm
clear;clc;
addpath('.Assignment5\noise');
addpath('.Assignment6');
img=imread('.\lena.bmp');
img=im2double(img);
figure; imshow(img);
%additive white noise to original img, evaluation of PSNR, WPSNR, 
noise_img=test_awgn(img,0.01,1);
%noise_img=img;
%noise_img(200:400,200:400)=test_awgn(img(200:400,200:400),0.01,1);
hidd=noise_img;
para=[sqrt(0.01)  sqrt(var(noise_img(:)))];
for i=1:size(img,1)
    for j=1:size(img,2)
        hidd(i,j)=max_sum(hidd,noise_img,i,j,para);
    end
end
hidd_wpsnr=WPSNR(hidd,img);
hidd_psnr=PSNR(hidd,img);
hidd_simm=ssim(hidd,img);
hidd_nmse=mse(hidd,img);
figure; imshow(hidd);
%% EM Algorithm and EM+MRF
addpath('.\Assignment5\noise');
addpath('.\Assignment5');
img=imread('.\lena.bmp');
img=im2double(img);
imgre=img(:);
noisy_img=img;
noisy_img(200:400,200:400)=test_awgn(img(200:400,200:400),0.3,1);
noise_wpsnr=WPSNR(noisy_img,img);
noise_psnr=PSNR(noisy_img,img);
noise_simm=ssim(noisy_img,img);
noise_nmse=mse(noisy_img,img);
noisy_img=reshape(noisy_img,[size(img,1)*size(img,2) 1]);
% Parameter Initialization
alpha=[0.5 0.5];
cov=[std(imgre) std(noisy_img)];
mu=[mean(imgre) mean(noisy_img)];
mu_old=0;
iter = 1000;
it=1;
while it <iter && abs(mu(1)-mu_old)>0.001
    mu_old=mu(1);
    for j = 1 : length(alpha)
        w(:,j)=alpha(j)*mvnpdf(noisy_img,mu(j),cov(j));
    end   
    w=w./repmat(sum(w,2),1,size(w,2));

% Maximum: 
    alpha = sum(w,1)./size(w,1); 
    
    mu = w'*noisy_img; 
    %mu= mu./repmat((sum(w,1))',1,size(mu,2));
    mu=mu/sum(w(:,1));
        
    for j = 1 : length(alpha)
        vari = sqrt(repmat(w(:,j),1,size(noisy_img,2))).*(noisy_img- mu(j)); 
        cov(j) = (vari'*vari)/sum(w(:,j),1);      
    end
    it=it+1;
end
%  L_log = log(cov(2)) - log(cov(1))-0.5.*((((log(noisy_img) - log(mu(1))).^2)/cov(1)^2)- (((log(noisy_img) - log(mu(2))).^2)/cov(2)^2));
%  map=exp(L_log);
%  map=1./(map+1);
%hidd=norminv(w(:,1),mu(1),cov(1));
hidd_select=w(:,2);
%hidd=mvnpdf(noisy_img,mu(1),cov(1))./(mvnpdf(noisy_img,mu(1),cov(1))+mvnpdf(noisy_img,mu(2),cov(2)));
hidd_select(hidd_select>0.5)=1; hidd_select(hidd_select<0.5)=0;
hidd_select=reshape(hidd_select,[size(img,1) size(img,2)]);
figure;imshow(hidd_select)
noisy_img=reshape(noisy_img,[size(img,1) size(img,2)]);
para=[1,0.5,0.2,std(std(noisy_img))];
hidd=noisy_img;
%hidd=noisy_img-norminv(w(:,2),mu(2),cov(2));
%hidd=noisy_img-w(:,2).*noisy_img;
%hidd(isinf(hidd))=1; hidd(isnan(hidd))=0; hidd(hidd<0)=0; hidd(hidd>1)=1; 
%hidd=map.*noisy_img;
%[c estimate] = max(w,[],2);
%hidd = find(estimate==1); 
total_energy=img_energy(noisy_img,noisy_img,para);
fprintf('total energy before denoising:%f \n',total_energy);
%ICM de-noising
iter=10;it=1;
energy_old=total_energy;
while it<iter+1 && total_energy<=energy_old
   energy_old=total_energy;
   for i=1 :size(hidd,1)
      for j=1:size(hidd,2)
          if(hidd_select(i,j)==1)
            [hidd,total_energy]=icm_update(noisy_img,hidd,i,j,total_energy,para);
          %[hidd,total_energy]=deri_update(noisy_img,hidd,i,j,total_energy,para);
          %hidd(i,j)=img(i,j);
          end
      end
   end
   it=it+1;
   fprintf('total energy after denoising:%f \n',total_energy);
end
hidd=reshape(hidd,[size(img,1) size(img,2)]);
hidd_wpsnr=WPSNR(hidd,img);
hidd_psnr=PSNR(hidd,img);
hidd_simm=ssim(hidd,img);
hidd_nmse=mse(hidd,img);
figure;imshow(hidd)
%% EM + MP/MS
addpath('.Assignment5\noise');
addpath('.Assignment6');
img=imread('.\lena.bmp');
img=im2double(img);
imgre=img(:);
noisy_img=img;
noisy_img(200:400,200:400)=test_awgn(img(200:400,200:400),0.3,1);
noisy_img=reshape(noisy_img,[size(img,1)*size(img,2) 1]);
% Parameter Initialization
alpha=[0.5 0.5];
cov=[std(imgre) std(noisy_img)];
mu=[mean(imgre) mean(noisy_img)];
mu_old=0;
iter = 1000;
it=1;
while it <iter && abs(mu(1)-mu_old)>0.001
    mu_old=mu(1);
    for j = 1 : length(alpha)
        w(:,j)=alpha(j)*mvnpdf(noisy_img,mu(j),cov(j));
    end   
    w=w./repmat(sum(w,2),1,size(w,2));

% Maximum: 
    alpha = sum(w,1)./size(w,1); 
    
    mu = w'*noisy_img; 
    %mu= mu./repmat((sum(w,1))',1,size(mu,2));
    mu=mu/sum(w(:,1));
        
    for j = 1 : length(alpha)
        vari = sqrt(repmat(w(:,j),1,size(noisy_img,2))).*(noisy_img- mu(j)); 
        cov(j) = (vari'*vari)/sum(w(:,j),1);      
    end
    it=it+1;
end
hidd_select=w(:,2);
hidd_select(hidd_select>0.5)=1; hidd_select(hidd_select<0.5)=0;
hidd_select=reshape(hidd_select,[size(img,1) size(img,2)]);
figure;imshow(hidd_select)
noisy_img=reshape(noisy_img,[size(img,1) size(img,2)]);
hidd=noisy_img;
para=cov.^2;
for i=1:size(img,1)
    for j=1:size(img,2)
        if(hidd_select(i,j)==1)
           hidd(i,j)=max_sum(hidd,noisy_img,i,j,para);
        end
    end
end
hidd_wpsnr=WPSNR(hidd,img);
hidd_psnr=PSNR(hidd,img);
hidd_simm=ssim(hidd,img);
hidd_nmse=mse(hidd,img);
figure; imshow(hidd);
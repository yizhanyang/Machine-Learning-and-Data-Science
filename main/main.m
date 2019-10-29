%% Odd/Even Training phase
clear;
x=[1,1,1,0;
   1,1,1,0;
   1,0,1,1;
   1,1,1,1;
   0,0,0,0;
   0,1,1,1;
   ];
t=[1;0;1;0;0;1];
lamda=0.5;
%% Odd/Even testing phase
w=linear_inference(t,x,lamda);
y1=[1,1,0,0]*w;
%% Gradient Descent
theta=ones(4,1); alpha=0.01; lamda=0.1;num_iters=100;
[theta, J_history] = gradientDescentMulti(x, t, theta, alpha,lamda,num_iters);
y2=[1,1,0,0]*theta;
%% Gaussian kernel regression 
addpath('D:\Courses\MLDS\AS\Assignment2');
sigma=0.5; lamda=0.4;
K=gram_matrix(x,sigma);
alpha=(K+lamda.*eye(size(x,1)))\t;
y3=0;
for i=1:size(x,1)
  y3=y3+alpha(i)'*gaussian_kernel(x(i,:),[1,1,0,0],sigma);
end
%% Gaussian process Regression
addpath('D:\Courses\MLDS\AS\Assignment3');
theta=ones(4,1); precision=0.001;
[K,C]=gaussian_process_gram(x,theta,precision,t);
mu=gaussian_process_kernel(x,[1,1,0,0],theta)'*pinv(C)*t;
xn=[x;[1,1,0,0]];
c=gaussian_process_kernel([1,1,0,0],[1,1,0,0],theta)+precision;
sigma=c-gaussian_process_kernel(x,[1,1,0,0],theta)'*pinv(C)*gaussian_process_kernel(x,[1,1,0,0],theta);
%% SVM 

%% 1-layer ELM
%Assume nodes,
n=3; learning_rate_w=0.001; learning_rate_o=0.001;
W=rand(n,size(x,2)); B=rand(n,1); O=rand(1,n);
for index=1:size(x,1)
    z=W*x(index,:)'+B;
    y=exp(z);
    if index==1
        O=t(index)*y'*pinv(y*y'+lamda*eye(size(y,1)));
    end
    W=W+learning_rate_w.*O'*(t(index)-O*y).*(1./z*x(index)');
    O=O+learning_rate_o.*(t(index)-O*y)*y'-lamda*O;
end
test=O*exp(W*[1,1,0,0]'+B);
%% 3-layer ELM

%% Sparse Representation
addpath('E:\KTH\P1\Machine Learning\Assignment4');
re1=zeros(100,1); re2=zeros(100,1); re3=zeros(100,1); 
x=[rand(5,1);zeros(495,1)];
for m=1:100
A=rand(m,500);
b=A*x;
xx1=orth_match(b,A,m);
xx2=sub_pursuit(b,A,m);
xx3=basis_pursuit(b,A,10e-4);
re1(m)=calNMSE(xx1,x);
re2(m)=calNMSE(xx2,x);
re3(m)=calNMSE(xx3,x);
end
%% Plot and Comparison
xlen=1:100;
plot(xlen/500,re1(xlen));
hold on
%plot(1./xlen,re2(xlen));
plot(xlen/500,re3(xlen));
legend('OMP','BP')
hold off
title('The Reconstruction Error versus Sparsity using OMP and BP','FontSize', 24)
xlabel('Sparsity level n/m','FontSize', 16)
ylabel('Normalized Mean Square Error','FontSize', 16)
%% Markov Random Field De-noising
clear;clc;
addpath('E:\KTH\P1\Machine Learning\Assignment5');
addpath('E:\KTH\P1\Machine Learning\Assignment5\noise');
img=imread('E:\KTH\P1\Machine Learning\Assignment5\lena.bmp');
img=im2double(img);
figure; imshow(img);
%additive white noise to original img, evaluation of PSNR, WPSNR, 
noise_img=test_awgn(img,0.01,1);
noise_wpsnr=WPSNR(noise_img,img);
noise_psnr=PSNR(noise_img,img);
noise_simm=ssim(noise_img,img);
figure; imshow(noise_img);
para=[1,0.5,0.02];
%energy of the noise image
hidd=noise_img;
total_energy=img_energy(noise_img,noise_img,para);
fprintf('total energy before denoising:%f \n',total_energy);
%ICM de-noising
for iter=1:10
   for i=1 :size(hidd,1)
      for j=1:size(hidd,2)
          [hidd,total_energy]=icm_update(noise_img,hidd,i,j,total_energy,para);
      end
   end
   fprintf('total energy after denoising:%f \n',total_energy);
end
hidd_wpsnr=WPSNR(hidd,img);
hidd_psnr=PSNR(hidd,img);
hidd_simm=ssim(hidd,img);
figure; imshow(hidd);




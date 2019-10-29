addpath('D:\Courses\MLDS\AS\Assignment4');
%% Vowel Odd/Even testing phase
clear;clc;
lamda = 0.1;
load("D:\Courses\MLDS\AS\main\vw.mat");
n=5; learning_rate_w=0.01; learning_rate_o=0.01;%hand-crafted
fMat = featureMat(1:528,:);
lMat = labelMat(1:528,:);
indices = crossvalind('Kfold', 528, 5);
for i = 1:5
    tic
    test = (indices == i);
    train = ~test;
    x = fMat(train, :);
    t = lMat(train, :);
    x1 = fMat(test, :);
    t1 = lMat(test, :);
    W=rand(n,size(x,2)); B=rand(n,1); O=rand(size(t,2),n);
    for index=1:size(x,1)
        z=W*x(index,:)'+B;
        y=exp(z);
%             if index==1
%                 O=t(index,:)'*y'*pinv(y*y'+lamda*eye(size(y,1)));
%             end
        O=O+learning_rate_o.*(t(index,:)'-O*y)*y'-lamda*O;
        W=W+learning_rate_w.*O'*(t(index,:)'-O*y).*(y*x(index,:));
    end
    y1=zeros(size(t1,1),size(t1,2));
    for p = 1:size(x1,1)
        y1(p,:)=O*exp(W*x1(p,:)'+B);
    end
    time(i) = toc;
    acc(i) = compute_accuracy(t1,y1);
    nme(i) = compute_nme(t1,y1);
end
train_nme = sum(nme)/max(i);
train_time = sum(time);
[~,ind]=max(acc);

x2 = featureMat(529:end,:);
t2 = labelMat(529:end,:);
y2=O*exp(W*x2'+B);
accuracy = compute_accuracy(t2,y2');
test_nme = compute_nme(t2,y2');
%% Satimage Odd/Even testing phase
clear;clc;
lamda = 0.1;
load("D:\Courses\MLDS\AS\main\st.mat");
n=5; learning_rate_w=0.01; learning_rate_o=0.01;%hand-crafted
fMat = featureMat(1:4430,:);
lMat = labelMat(1:4430,:);
indices = crossvalind('Kfold', 528, 5);
for i = 1:5
    tic
    test = (indices == i);
    train = ~test;
    x = fMat(train, :);
    t = lMat(train, :);
    x1 = fMat(test, :);
    t1 = lMat(test, :);
    W=rand(n,size(x,2)); B=rand(n,1); O=rand(size(t,2),n);
    for index=1:size(x,1)
        z=W*x(index,:)'+B;
        y=log(z);
%             if index==1
%                 O=t(index,:)'*y'*pinv(y*y'+lamda*eye(size(y,1)));
%             end
        O=O+learning_rate_o.*(t(index,:)'-O*y)*y'-lamda*O;
        W=W+learning_rate_w.*O'*(t(index,:)'-O*y).*(1./y*x(index,:));
    end
    y1=zeros(size(t1,1),size(t1,2));
    for p = 1:size(x1,1)
        y1(p,:)=O*log(W*x1(p,:)'+B);
    end
    time(i) = toc;
    acc(i) = compute_accuracy(t1,y1);
    nme(i) = compute_nme(t1,y1);
end
train_nme = sum(nme)/max(i);
train_time = sum(time);
[~,ind]=max(acc);

x2 = featureMat(4431:end,:);
t2 = labelMat(4431:end,:);
y2=O*log(W*x2'+B);
accuracy = compute_accuracy(t2,y2');
test_nme = compute_nme(t2,y2');

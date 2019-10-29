addpath('D:\Courses\MLDS\AS\Assignment2');
%% Vowel
sigma=0.5; lamda=0.05;
whole = csvread("C:\Users\Yizhan\Documents\datasets\vw_train.csv");
indices = crossvalind('Kfold', 528, 4);
for i = 1:4
    tic
    test = (indices == i);
    train = ~test;
    train = whole(train, :);
    test = whole(test, :);
    x= train(:,1:10); 
    t=train(:,11:end);
    x1= test(:,1:10); 
    t1=test(:,11:end);
    %w(i,:,:) = linear_inference(t,x,lamda);
    K = gram_matrix(x,sigma);
    alpha=(K+lamda.*eye(size(x,1)))\t;
    alpha_train(i,:,:) = reshape(alpha,1,size(alpha,1),size(alpha,2));
    y1=zeros(size(t1,1),size(t1,2));
    for p = 1:size(x1,1)
        for k=1:size(x,1)
            y1(p,:)=y1(p,:)+(alpha(k,:)'*gaussian_kernel(x(k,:),x1(p,:),sigma))';
        end %train
    end
    time(i) = toc;
    acc(i) = compute_accuracy(t1,y1);
    nme(i) = compute_nme(t1,y1);
end  
train_nme = sum(nme)/max(i);
train_time = sum(time);

test = csvread("D:\Courses\MLDS\AS\dataset\vw_test.csv");
[~,ind]=max(acc);
x2= test(:,1:10); 
t2=test(:,11:end);
alpha_test = reshape(alpha(ind,:,:),size(alpha,2),size(alpha,3));
y2=zeros(size(t2,1),size(t2,2));
for q = 1: 10
    for p = 1:size(x2,1)
        for k=1:size(x,1)
            y2(p,:)=y2(p,:)+(alpha(k,:)'*gaussian_kernel(x(k,:),x2(p,:),sigma))';
        end
    end
    accuracy(q) = compute_accuracy(t2,y2);
    test_nme = compute_nme(t2,y2);
end
facc = mean(accuracy);
fdev = std(accuracy);
%% Yale
clear;clc;
sigma=0.5; lamda=0.05;
load("D:\Courses\MLDS\AS\main\ExtendedYaleB.mat");
fMat = featureMat(1:1600,:);
lMat = labelMat(1:1600,:);
indices = crossvalind('Kfold', 1600, 5);
for i = 1:5
    tic
    test = (indices == i);
    train = ~test;
    x = fMat(train, :);
    t = lMat(train, :);
    x1 = fMat(test, :);
    t1 = lMat(test, :);

    %w(i,:,:) = linear_inference(t,x,lamda);
    K = gram_matrix(x,sigma);
    alpha=(K+lamda.*eye(size(x,1)))\t;
    alpha_train(i,:,:) = reshape(alpha,1,size(alpha,1),size(alpha,2));
    y1=zeros(size(t1,1),size(t1,2));
    for p = 1:size(x1,1)
        for k=1:size(x,1)
            y1(p,:)=y1(p,:)+(alpha(k,:)'*gaussian_kernel(x(k,:),x1(p,:),sigma))';
        end %train
    end
    time(i) = toc;
    acc(i) = compute_accuracy(t1,y1);
    nme(i) = compute_nme(t1,y1);
end  
train_nme = sum(nme)/max(i);
train_time = sum(time);

x2 = featureMat(1601:end,:);
t2 = labelMat(1601:end,:);
[~,ind]=max(acc);
alpha_test = reshape(alpha(ind,:,:),size(alpha,2),size(alpha,3));
%test
y2=zeros(size(t2,1),size(t2,2));
for p = 1:size(x2,1)
    for k=1:size(x,1)
        y2(p,:)=y2(p,:)+(alpha(k,:)'*gaussian_kernel(x(k,:),x2(p,:),sigma))';
    end
end
accuracy = compute_accuracy(t2,y2);
test_nme = compute_nme(t2,y2);
facc = mean(accuracy);
fdev = std(acc);
%% AR
clear;clc;
sigma=0.5; lamda=0.05;
load("D:\Courses\MLDS\AS\main\AR.mat");
fMat = featureMat(1:1800,:);
lMat = labelMat(1:1800,:);
indices = crossvalind('Kfold', 1800, 5);
for i = 1:5
    tic
    test = (indices == i);
    train = ~test;
    x = fMat(train, :);
    t = lMat(train, :);
    x1 = fMat(test, :);
    t1 = lMat(test, :);

    %w(i,:,:) = linear_inference(t,x,lamda);
    K = gram_matrix(x,sigma);
    alpha=(K+lamda.*eye(size(x,1)))\t;
    alpha_train(i,:,:) = reshape(alpha,1,size(alpha,1),size(alpha,2));
    y1=zeros(size(t1,1),size(t1,2));
    for p = 1:size(x1,1)
        for k=1:size(x,1)
            y1(p,:)=y1(p,:)+(alpha(k,:)'*gaussian_kernel(x(k,:),x1(p,:),sigma))';
        end %train
    end
    time(i) = toc;
    acc(i) = compute_accuracy(t1,y1);
    nme(i) = compute_nme(t1,y1);
end  
train_nme = sum(nme)/max(i);
train_time = sum(time);

x2 = featureMat(1801:end,:);
t2 = labelMat(1801:end,:);
[~,ind]=max(acc);
alpha_test = reshape(alpha(ind,:,:),size(alpha,2),size(alpha,3));
%test
y2=zeros(size(t2,1),size(t2,2));
for p = 1:size(x2,1)
    for k=1:size(x,1)
        y2(p,:)=y2(p,:)+(alpha(k,:)'*gaussian_kernel(x(k,:),x2(p,:),sigma))';
    end
end
accuracy = compute_accuracy(t2,y2);
test_nme = compute_nme(t2,y2);
facc = mean(accuracy);
fdev = std(acc);
%% Satimage
clear;clc;
sigma=0.5; lamda=0.05;
load("D:\Courses\MLDS\AS\main\st.mat");
fMat = featureMat(1:4430,:);
lMat = labelMat(1:4430,:);
indices = crossvalind('Kfold', 4430, 5);
for i = 1:5
    tic
    test = (indices == i);
    train = ~test;
    x = fMat(train, :);
    t = lMat(train, :);
    x1 = fMat(test, :);
    t1 = lMat(test, :);

    %w(i,:,:) = linear_inference(t,x,lamda);
    K = gram_matrix(x,sigma);
    alpha=(K+lamda.*eye(size(x,1)))\t;
    alpha_train(i,:,:) = reshape(alpha,1,size(alpha,1),size(alpha,2));
    y1=zeros(size(t1,1),size(t1,2));
    for p = 1:size(x1,1)
        for k=1:size(x,1)
            y1(p,:)=y1(p,:)+(alpha(k,:)'*gaussian_kernel(x(k,:),x1(p,:),sigma))';
        end %train
    end
    time(i) = toc;
    acc(i) = compute_accuracy(t1,y1);
    nme(i) = compute_nme(t1,y1);
end  
train_nme = sum(nme)/max(i);
train_time = sum(time);

x2 = featureMat(4431:end,:);
t2 = labelMat(4431:end,:);
[~,ind]=max(acc);
alpha_test = reshape(alpha(ind,:,:),size(alpha,2),size(alpha,3));
%test
y2=zeros(size(t2,1),size(t2,2));
for p = 1:size(x2,1)
    for k=1:size(x,1)
        y2(p,:)=y2(p,:)+(alpha(k,:)'*gaussian_kernel(x(k,:),x2(p,:),sigma))';
    end
end
accuracy = compute_accuracy(t2,y2);
test_nme = compute_nme(t2,y2);
facc = mean(accuracy);
fdev = std(acc);
%% Scene15
clear;clc;
sigma=0.5; lamda=0.05;
load("D:\Courses\MLDS\AS\main\Scene15.mat");
fMat = featureMat(1:3000,:);
lMat = labelMat(1:3000,:);
indices = crossvalind('Kfold', 3000, 5);
for i = 1:5
    tic
    test = (indices == i);
    train = ~test;
    x = fMat(train, :);
    t = lMat(train, :);
    x1 = fMat(test, :);
    t1 = lMat(test, :);

    %w(i,:,:) = linear_inference(t,x,lamda);
    K = gram_matrix(x,sigma);
    alpha=(K+lamda.*eye(size(x,1)))\t;
    alpha_train(i,:,:) = reshape(alpha,1,size(alpha,1),size(alpha,2));
    y1=zeros(size(t1,1),size(t1,2));
    for p = 1:size(x1,1)
        for k=1:size(x,1)
            y1(p,:)=y1(p,:)+(alpha(k,:)'*gaussian_kernel(x(k,:),x1(p,:),sigma))';
        end %train
    end
    time(i) = toc;
    acc(i) = compute_accuracy(t1,y1);
    nme(i) = compute_nme(t1,y1);
end  
train_nme = sum(nme)/max(i);
train_time = sum(time);

x2 = featureMat(3001:end,:);
t2 = labelMat(3001:end,:);
[~,ind]=max(acc);
alpha_test = reshape(alpha(ind,:,:),size(alpha,2),size(alpha,3));
%test
y2=zeros(size(t2,1),size(t2,2));
for p = 1:size(x2,1)
    for k=1:size(x,1)
        y2(p,:)=y2(p,:)+(alpha(k,:)'*gaussian_kernel(x(k,:),x2(p,:),sigma))';
    end
end
accuracy = compute_accuracy(t2,y2);
test_nme = compute_nme(t2,y2);
facc = mean(accuracy);
fdev = std(acc);
%% Caltech101
clear;clc;
sigma=0.5; lamda=0.05;
load("D:\Courses\MLDS\AS\dataset\Caltech101.mat");
fMat = featureMat(1:6000,:);
lMat = labelMat(1:6000,:);
indices = crossvalind('Kfold', 6000, 5);
for i = 1:5
    tic
    test = (indices == i);
    train = ~test;
    x = fMat(train, :);
    t = lMat(train, :);
    x1 = fMat(test, :);
    t1 = lMat(test, :);

    %w(i,:,:) = linear_inference(t,x,lamda);
    K = gram_matrix(x,sigma);
    alpha=(K+lamda.*eye(size(x,1)))\t;
    alpha_train(i,:,:) = reshape(alpha,1,size(alpha,1),size(alpha,2));
    y1=zeros(size(t1,1),size(t1,2));
    for p = 1:size(x1,1)
        for k=1:size(x,1)
            y1(p,:)=y1(p,:)+(alpha(k,:)'*gaussian_kernel(x(k,:),x1(p,:),sigma))';
        end %train
    end
    time(i) = toc;
    acc(i) = compute_accuracy(t1,y1);
    nme(i) = compute_nme(t1,y1);
end  
train_nme = sum(nme)/max(i);
train_time = sum(time);

x2 = featureMat(6001:end,:);
t2 = labelMat(6001:end,:);
[~,ind]=max(acc);
%% 
alpha_test = reshape(alpha_train(5,:,:),size(alpha_train,2),size(alpha_train,3));
for i = 1:5
    tic
    test = (indices == i);
    train = ~test;
    x = fMat(train, :);
    t = lMat(train, :);
    x1 = fMat(test, :);
    t1 = lMat(test, :);
end
%test
y2=zeros(size(t2,1),size(t2,2));
for p = 1:size(x2,1)
    for k=1:size(x,1)
        y2(p,:)=y2(p,:)+(alpha_test(k,:)'*gaussian_kernel(x(k,:),x2(p,:),sigma))';
    end
end
accuracy = compute_accuracy(t2,y2);
test_nme = compute_nme(t2,y2);
facc = mean(accuracy);
fdev = std(acc);
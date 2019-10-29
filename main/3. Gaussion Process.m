addpath('D:\Courses\MLDS\AS\Assignment3');
%% Vowel Odd/Even testing phase
clear;clc;
whole = csvread("D:\Courses\MLDS\AS\dataset\vw_train.csv");
lamda=0.5;
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
    theta=ones(size(x,2),1); precision=0.001;
    [K,C]=gaussian_process_gram(x,theta,precision,t);
    C_train(i,:,:) = reshape(C,1,size(C,1),size(C,2));
    y1 = [];
    for p = 1:size(x1,1)
        yp=gaussian_process_kernel(x,x1(p,:),theta)'*pinv(C)*t;
        y1 = [y1;yp];
    end

    time(i) = toc;
    acc(i) = compute_accuracy(t1,y1);
    nme(i) = compute_nme(t1,y1);
end
train_nme = sum(nme)/max(i);
train_time = sum(time);

[~,ind]=max(acc);
test = csvread("D:\Courses\MLDS\AS\dataset\vw_test.csv");
x2= test(:,1:10); 
t2=test(:,11:end);
y2 = [];
C_test = reshape(C_train(ind,:,:),size(C_train,2),size(C_train,3));
for p = 1:size(x2,1)
     yp=gaussian_process_kernel(x,x2(p,:),theta)'*pinv(C_test)*t;
     y2 = [y2;yp];
     c(p,:)=gaussian_process_kernel(x2(p,:),x2(p,:),theta)+precision;
     sigma(p,:)=c(p,:)-gaussian_process_kernel(x,x2(p,:),theta)'*pinv(C)*gaussian_process_kernel(x,x2(p,:),theta);
end
accuracy = compute_accuracy(t2,y2);
test_nme = compute_nme(t2,y2);
faccdev = std(acc);
%% ExtendedYaleB dataset Odd/Even testing phase
clear;clc;
load("D:\Courses\MLDS\AS\main\ExtendedYaleB.mat");
lamda=1;
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
    theta=ones(size(x,2),1); precision=0.001;
    [K,C]=gaussian_process_gram(x,theta,precision,t);
    C_train(i,:,:) = reshape(C,1,size(C,1),size(C,2));
    y1 = [];
    for p = 1:size(x1,1)
        yp=gaussian_process_kernel(x,x1(p,:),theta)'*pinv(C)*t;
        y1 = [y1;yp];
    end
    time(i) = toc;
    acc(i) = compute_accuracy(t1,y1);
    nme(i) = compute_nme(t1,y1);
[~,ind]=max(acc);
x2 = featureMat(1601:end,:);
t2 = labelMat(1601:end,:);
y2 = [];
C_test = reshape(C_train(ind,:,:),size(C_train,2),size(C_train,3));
for p = 1:size(x2,1)
     yp=gaussian_process_kernel(x,x2(p,:),theta)'*pinv(C_test)*t;
     y2 = [y2;yp];
 end
accuracy = compute_accuracy(t2,y2);
test_nme = compute_nme(t2,y2);
faccdev = std(accuracy);
end
%% AR dataset Odd/Even testing phase
clear;clc;
load("D:\Courses\MLDS\AS\main\AR.mat");
lamda=1;
fMat = featureMat(1:1800,:);
lMat = labelMat(1:1800,:);
indices = crossvalind('Kfold',1800, 5);
for i = 1:5
    tic
    test = (indices == i);
    train = ~test;
    x = fMat(train, :);
    t = lMat(train, :);
    x1 = fMat(test, :);
    t1 = lMat(test, :);
    theta=ones(size(x,2),1); precision=0.001;
    [K,C]=gaussian_process_gram(x,theta,precision,t);
    C_train(i,:,:) = reshape(C,1,size(C,1),size(C,2));
    y1 = [];
    for p = 1:size(x1,1)
        yp=gaussian_process_kernel(x,x1(p,:),theta)'*pinv(C)*t;
        y1 = [y1;yp];
    end
    time(i) = toc;
    acc(i) = compute_accuracy(t1,y1);
    nme(i) = compute_nme(t1,y1);
[~,ind]=max(acc);
x2 = featureMat(1801:end,:);
t2 = labelMat(1801:end,:);
y2 = [];
C_test = reshape(C_train(ind,:,:),size(C_train,2),size(C_train,3));
for p = 1:size(x2,1)
     yp=gaussian_process_kernel(x,x2(p,:),theta)'*pinv(C_test)*t;
     y2 = [y2;yp];
 end
accuracy = compute_accuracy(t2,y2);
test_nme = compute_nme(t2,y2);
faccdev = std(accuracy);
end
%% Satimage dataset Odd/Even testing phase
clear;clc;
load("D:\Courses\MLDS\AS\main\st.mat");
lamda=0.5;
fMat = featureMat(1:4430,:);
lMat = labelMat(1:4430,:);
indices = crossvalind('Kfold',4430, 5);
for i = 1:5
    tic
    test = (indices == i);
    train = ~test;
    x = fMat(train, :);
    t = lMat(train, :);
    x1 = fMat(test, :);
    t1 = lMat(test, :);
    theta=ones(size(x,2),1); precision=0.001;
    [K,C]=gaussian_process_gram(x,theta,precision,t);
    C_train(i,:,:) = reshape(C,1,size(C,1),size(C,2));
    y1 = [];
    for p = 1:size(x1,1)
        yp=gaussian_process_kernel(x,x1(p,:),theta)'*pinv(C)*t;
        y1 = [y1;yp];
    end
    time(i) = toc;
    acc(i) = compute_accuracy(t1,y1);
    nme(i) = compute_nme(t1,y1);
[~,ind]=max(acc);
x2 = featureMat(4431:end,:);
t2 = labelMat(4431:end,:);
y2 = [];
C_test = reshape(C_train(ind,:,:),size(C_train,2),size(C_train,3));
for p = 1:size(x2,1)
     yp=gaussian_process_kernel(x,x2(p,:),theta)'*pinv(C_test)*t;
     y2 = [y2;yp];
 end
accuracy = compute_accuracy(t2,y2);
test_nme = compute_nme(t2,y2);
faccdev = std(accuracy);
end
%% Scene15
clear;clc;
load("D:\Courses\MLDS\AS\main\Scene.mat");
lamda=0.5;
fMat = featureMat(1:3000,:);
lMat = labelMat(1:3000,:);
indices = crossvalind('Kfold',4430, 5);
for i = 1:5
    tic
    test = (indices == i);
    train = ~test;
    x = fMat(train, :);
    t = lMat(train, :);
    x1 = fMat(test, :);
    t1 = lMat(test, :);
    theta=ones(size(x,2),1); precision=0.001;
    [K,C]=gaussian_process_gram(x,theta,precision,t);
    C_train(i,:,:) = reshape(C,1,size(C,1),size(C,2));
    y1 = [];
    for p = 1:size(x1,1)
        yp=gaussian_process_kernel(x,x1(p,:),theta)'*pinv(C)*t;
        y1 = [y1;yp];
    end
    time(i) = toc;
    acc(i) = compute_accuracy(t1,y1);
    nme(i) = compute_nme(t1,y1);
[~,ind]=max(acc);
x2 = featureMat(3001:end,:);
t2 = labelMat(3001:end,:);
y2 = [];
C_test = reshape(C_train(ind,:,:),size(C_train,2),size(C_train,3));
for p = 1:size(x2,1)
     yp=gaussian_process_kernel(x,x2(p,:),theta)'*pinv(C_test)*t;
     y2 = [y2;yp];
 end
accuracy = compute_accuracy(t2,y2);
test_nme = compute_nme(t2,y2);
faccdev = std(accuracy); 
end
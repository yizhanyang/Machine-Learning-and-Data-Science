addpath('D:\Courses\MLDS\AS\Assignment1');
%% Vowel Odd/Even testing phase
clear;clc;
whole = csvread("D:\Courses\MLDS\AS\dataset\vw_train.csv");
lamda=0.1;
indices = crossvalind('Kfold', 528, 5);
for j = 1:20
    for i = 1:5
        tic
        test = (indices == i);
        train = ~test;
        train = whole(train, :);
        test = whole(test, :);
        x= train(:,1:10); 
        t=train(:,11:end);
        w(i,:,:) = linear_inference(t,x,lamda);
        x1= test(:,1:10); 
        t1=test(:,11:end);
        w1 = reshape(w(i,:,:) ,size(w,2),size(w,3));
        y1=x1*w1;
        time(i) = toc;
        acc(i) = compute_accuracy(t1,y1);
        nme(i) = compute_nme(t1,y1);
    end
train_nme(j) = sum(nme)/max(i);
train_time(j) = sum(time);
[~,ind]=max(acc);
wtest=reshape(w(ind,:,:),size(w,2),size(w,3));
test = csvread("D:\Courses\MLDS\AS\dataset\vw_test.csv");
x2= test(:,1:10); 
t2=test(:,11:end);
y2=x2*wtest;
accuracy(j) = compute_accuracy(t2,y2);
test_nme(j) = compute_nme(t2,y2);
end
facc = mean(accuracy(j));
faccdev = std(accuracy);
ftrain_nme = mean(train_nme(j));
ftest_nme = mean(test_nme(j));
ftrain_time = mean(train_time(j));
%% ExtendedYaleB dataset Odd/Even testing phase
clear;clc;
load("D:\Courses\MLDS\AS\dataset\ExtendedYaleB.mat");
fMat = featureMat(1:1600,:);
lMat = labelMat(1:1600,:);
lamda=0.1;
indices = crossvalind('Kfold', 1600, 5);
for j = 1:1
    for i = 1:5
        tic
        test = (indices == i);
        train = ~test;
        train_x = fMat(train, :);
        train_t = lMat(train, :);
        test_x = fMat(test, :);
        test_t = lMat(test, :);

        w(i,:,:) = linear_inference(train_t,train_x,lamda);
        w1 = reshape(w(i,:,:) ,size(w,2),size(w,3));
        x1 = test_x;
        y1=x1*w1;
        t1 = test_t;
        time(i) = toc;
        acc(i) = compute_accuracy(t1,y1);
        nme(i) = compute_nme(t1,y1);
    end
train_nme(j) = sum(nme)/max(i);
train_time(j) = sum(time);
[~,ind]=max(acc);
wtest=reshape(w(ind,:,:),size(w,2),size(w,3));
x2 = featureMat(1601:end,:);
t2 = labelMat(1601:end,:);
y2=x2*wtest;
accuracy(j) = compute_accuracy(t2,y2);
test_nme(j) = compute_nme(t2,y2);
end
facc = mean(accuracy(j));
faccdev = std(accuracy(j));
ftrain_nme = mean(train_nme(j));
ftest_nme = mean(test_nme(j));
ftrain_time = mean(train_time(j));
%% AR dataset Odd/Even testing phase
clear;clc;
load("D:\Courses\MLDS\AS\dataset\AR.mat");
fMat = featureMat(1:1800,:);
lMat = labelMat(1:1800,:);
lamda=0.05;
indices = crossvalind('Kfold', 1800, 5);
for j = 1:1
    for i = 1:5
        tic
        test = (indices == i);
        train = ~test;
        train_x = featureMat(train, :);
        train_t = labelMat(train, :);
        test_x = featureMat(test, :);
        test_t = labelMat(test, :);

        w(i,:,:) = linear_inference(train_t,train_x,lamda);
        w1 = reshape(w(i,:,:) ,size(w,2),size(w,3));
        y1=test_x*w1;
        time(i) = toc;
        acc(i) = compute_accuracy(test_t,y1);
        nme(i) = compute_nme(test_t,y1);
    end
train_nme(j) = sum(nme)/max(i);
train_time(j) = sum(time);
[~,ind]=max(acc);
wtest=reshape(w(ind,:,:),size(w,2),size(w,3));
load("C:\Users\Yizhan\Documents\datasets\AR.mat");
featureMat = featureMat';
labelMat = labelMat';
x2 = featureMat(1801:end,:);
t2 = labelMat(1801:end,:);
y2=x2*wtest;
accuracy(j) = compute_accuracy(t2,y2);
test_nme(j) = compute_nme(t2,y2);
end
facc = mean(accuracy(j));
faccdev = std(accuracy(j));
ftrain_nme = mean(train_nme(j));
ftest_nme = mean(test_nme(j));
ftrain_time = mean(train_time(j));
%% %%Satimage dataset Odd/Even testing phase
clear;clc;
whole = csvread("D:\Courses\MLDS\AS\dataset\st.csv");
whole = whole(1:4430,:);
lamda=0.1;
indices = crossvalind('Kfold', 4430, 5);
for j = 1:20
    for i = 1:5
        tic
        test = (indices == i);
        train = ~test;
        train = whole(train, :);
        test = whole(test, :);
        x= train(:,1:36); 
        t=train(:,37:end);
        w(i,:,:) = linear_inference(t,x,lamda);
        x1= test(:,1:36); 
        t1=test(:,37:end);
        w1 = reshape(w(i,:,:) ,size(w,2),size(w,3));
        y1=x1*w1;
        time(i) = toc;
        acc(i) = compute_accuracy(t1,y1);
        nme(i) = compute_nme(t1,y1);
    end
train_nme(j) = sum(nme)/max(i);
train_time(j) = sum(time);
[~,ind]=max(acc);
wtest=reshape(w(ind,:,:),size(w,2),size(w,3));
whole = csvread("C:\Users\Yizhan\Documents\datasets\st.csv");
test = whole(4431:end,:);
x2= test(:,1:36); 
t2=test(:,37:end);
y2=x2*wtest;
accuracy(j) = compute_accuracy(t2,y2);
test_nme(j) = compute_nme(t2,y2);
end
facc = mean(accuracy(j));
faccdev = std(accuracy(j));
ftrain_nme = mean(train_nme(j));
ftest_nme = mean(test_nme(j));
ftrain_time = mean(train_time(j));
%% Scene15 dataset Odd/Even testing phase
clear;clc;
load("C:\Users\Yizhan\Documents\datasets\Scene15.mat");
featureMat = featureMat';
labelMat = labelMat';
dataMat = cat(2,featureMat,labelMat);
dataMat_i = randperm(size(dataMat,1));
dataMat = dataMat(dataMat_i,:); %random partition
featureMat = dataMat(:,1:504);
labelMat = dataMat(:,505:end);
fMat = featureMat(1:3000,:);
lMat = labelMat(1:3000,:);
lamda=0.1;
indices = crossvalind('Kfold', 3000, 5);
for j = 1:20
    for i = 1:5
        tic
        test = (indices == i);
        train = ~test;
        train_x = fMat(train, :);
        train_t = lMat(train, :);
        test_x = fMat(test, :);
        test_t = lMat(test, :);

        w(i,:,:) = linear_inference(train_t,train_x,lamda);
        w1 = reshape(w(i,:,:) ,size(w,2),size(w,3));
        x1 = test_x;
        y1=x1*w1;
        t1 = test_t;
        time(i) = toc;
        acc(i) = compute_accuracy(t1,y1);
        nme(i) = compute_nme(t1,y1);
    end
train_nme(j) = sum(nme)/max(i);
train_time(j) = sum(time);
[~,ind]=max(acc);
wtest=reshape(w(ind,:,:),size(w,2),size(w,3));
x2 = featureMat(3001:end,:);
t2 = labelMat(3001:end,:);
y2=x2*wtest;
accuracy(j) = compute_accuracy(t2,y2);
test_nme(j) = compute_nme(t2,y2);
end
facc = mean(accuracy(j));
faccdev = std(accuracy(j));
ftrain_nme = mean(train_nme(j));
ftest_nme = mean(test_nme(j));
ftrain_time = mean(train_time(j));
%% Caltech101 dataset Odd/Even testing phase
clear;clc;
load("D:\Courses\MLDS\AS\dataset\Caltech101.mat");
fMat = featureMat(1:6000,:);
lMat = labelMat(1:6000,:);
lamda=0.1;
indices = crossvalind('Kfold', 6000, 5);
for j = 1:1
    for i = 1:5
        tic
        test = (indices == i);
        train = ~test;
        train_x = fMat(train, :);
        train_t = lMat(train, :);
        test_x = fMat(test, :);
        test_t = lMat(test, :);

        w(i,:,:) = linear_inference(train_t,train_x,lamda);
        w1 = reshape(w(i,:,:) ,size(w,2),size(w,3));
        x1 = test_x;
        y1=x1*w1;
        t1 = test_t;
        time(i) = toc;
        acc(i) = compute_accuracy(t1,y1);
        nme(i) = compute_nme(t1,y1);
    end
train_nme(j) = sum(nme)/max(i);
train_time(j) = sum(time);
[~,ind]=max(acc);
wtest=reshape(w(ind,:,:),size(w,2),size(w,3));
x2 = featureMat(6001:end,:);
t2 = labelMat(6001:end,:);
y2=x2*wtest;
accuracy(j) = compute_accuracy(t2,y2);
test_nme(j) = compute_nme(t2,y2);
end
facc = mean(accuracy(j));
faccdev = std(accuracy(j));
ftrain_nme = mean(train_nme(j));
ftest_nme = mean(test_nme(j));
ftrain_time = mean(train_time(j));

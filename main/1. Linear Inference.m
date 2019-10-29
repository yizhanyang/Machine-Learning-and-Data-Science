addpath('D:\Courses\MLDS\AS\Assignment1');
%% Vowel Odd/Even testing phase
clear;clc;
sigma=0.5; lamda=0.05;
load("D:\Courses\MLDS\AS\main\vw.mat");
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
        w(i,:,:) = linear_inference(t,x,lamda);
        w1 = reshape(w(i,:,:) ,size(w,2),size(w,3));
        y1=x1*w1;
        time(i) = toc;
        acc(i) = compute_accuracy(t1,y1);
        nme(i) = compute_nme(t1,y1);
    end
train_nme = sum(nme)/max(i);
train_time= sum(time);
[~,ind]=max(acc);
wtest=reshape(w(ind,:,:),size(w,2),size(w,3));
x2 = featureMat(529:end,:);
t2 = labelMat(529:end,:);
y2=x2*wtest;
accuracy = compute_accuracy(t2,y2);
test_nme = compute_nme(t2,y2);
fdev = std(acc);
%% ExtendedYaleB dataset Odd/Even testing phase
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
        w(i,:,:) = linear_inference(t,x,lamda);
        w1 = reshape(w(i,:,:) ,size(w,2),size(w,3));
        y1=x1*w1;
        time(i) = toc;
        acc(i) = compute_accuracy(t1,y1);
        nme(i) = compute_nme(t1,y1);
    end
train_nme = sum(nme)/max(i);
train_time= sum(time);
[~,ind]=max(acc);
wtest=reshape(w(ind,:,:),size(w,2),size(w,3));
x2 = featureMat(1601:end,:);
t2 = labelMat(1601:end,:);
y2=x2*wtest;
accuracy = compute_accuracy(t2,y2);
test_nme = compute_nme(t2,y2);
fdev = std(acc);
%% AR dataset Odd/Even testing phase
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
        w(i,:,:) = linear_inference(t,x,lamda);
        w1 = reshape(w(i,:,:) ,size(w,2),size(w,3));
        y1=x1*w1;
        time(i) = toc;
        acc(i) = compute_accuracy(t1,y1);
        nme(i) = compute_nme(t1,y1);
    end
train_nme = sum(nme)/max(i);
train_time= sum(time);
[~,ind]=max(acc);
wtest=reshape(w(ind,:,:),size(w,2),size(w,3));
x2 = featureMat(1801:end,:);
t2 = labelMat(1801:end,:);
y2=x2*wtest;
accuracy = compute_accuracy(t2,y2);
test_nme = compute_nme(t2,y2);
fdev = std(acc);
%% %%Satimage dataset Odd/Even testing phase
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
        w(i,:,:) = linear_inference(t,x,lamda);
        w1 = reshape(w(i,:,:) ,size(w,2),size(w,3));
        y1=x1*w1;
        time(i) = toc;
        acc(i) = compute_accuracy(t1,y1);
        nme(i) = compute_nme(t1,y1);
    end
train_nme = sum(nme)/max(i);
train_time= sum(time);
[~,ind]=max(acc);
wtest=reshape(w(ind,:,:),size(w,2),size(w,3));
x2 = featureMat(4431:end,:);
t2 = labelMat(4431:end,:);
y2=x2*wtest;
accuracy = compute_accuracy(t2,y2);
test_nme = compute_nme(t2,y2);
fdev = std(acc);
%% Scene15 dataset Odd/Even testing phase
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
        w(i,:,:) = linear_inference(t,x,lamda);
        w1 = reshape(w(i,:,:) ,size(w,2),size(w,3));
        y1=x1*w1;
        time(i) = toc;
        acc(i) = compute_accuracy(t1,y1);
        nme(i) = compute_nme(t1,y1);
    end
train_nme = sum(nme)/max(i);
train_time= sum(time);
[~,ind]=max(acc);
wtest=reshape(w(ind,:,:),size(w,2),size(w,3));
x2 = featureMat(3001:end,:);
t2 = labelMat(3001:end,:);
y2=x2*wtest;
accuracy = compute_accuracy(t2,y2);
test_nme = compute_nme(t2,y2);
fdev = std(acc);
%% Caltech101 dataset Odd/Even testing phase
clear;clc;
sigma=0.5; lamda=0.05;
load("D:\Courses\MLDS\AS\main\Caltech101.mat");
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
        w(i,:,:) = linear_inference(t,x,lamda);
        w1 = reshape(w(i,:,:) ,size(w,2),size(w,3));
        y1=x1*w1;
        time(i) = toc;
        acc(i) = compute_accuracy(t1,y1);
        nme(i) = compute_nme(t1,y1);
    end
train_nme = sum(nme)/max(i);
train_time= sum(time);
[~,ind]=max(acc);
wtest=reshape(w(ind,:,:),size(w,2),size(w,3));
x2 = featureMat(6001:end,:);
t2 = labelMat(6001:end,:);
y2=x2*wtest;
accuracy = compute_accuracy(t2,y2);
test_nme = compute_nme(t2,y2);
fdev = std(acc);
%% Vowel
clear;clc;
whole = csvread("D:\Courses\MLDS\AS\dataset\vw.csv");
X= whole(:,1:10); 
Y=whole(:,11:end);
for i = 1:size(Y,2)
    Y(Y(:,i)==1) = i;
end
Y = Y(:,1);
    
t = templateSVM('Standardize',true);
for j=1:20
    tic
    Mdl = fitcecoc(X,Y,'Learners',t);
    time(j)=toc;
    CVMdl = crossval(Mdl);
    genError = kfoldLoss(CVMdl);
    acc(j) = 1- genError;
end
facc=mean(acc);
fstd=std(acc);
ftime=mean(time);
%% ExtendYaleB
clear;clc;
load("D:\Courses\MLDS\AS\dataset\ExtendedYaleB.mat");
X= featureMat; 
Y=labelMat;
for i = 1:size(Y,2)
    Y(Y(:,i)==1) = i;
end
Y = Y(:,1);
t = templateSVM('Standardize',true);
for j=1:1
    tic
    Mdl = fitcecoc(X,Y,'Learners',t);
    time(j)=toc;
    CVMdl = crossval(Mdl);
    genError = kfoldLoss(CVMdl);
    acc(j) = 1- genError;
end
facc=mean(acc);
fstd=std(acc);
ftime=mean(time);
%% AR
clear;clc;
load("D:\Courses\MLDS\AS\dataset\AR.mat");
X= featureMat; 
Y=labelMat;
for i = 1:size(Y,2)
    Y(Y(:,i)==1) = i;
end
Y = Y(:,1);
t = templateSVM('Standardize',true);
for j=1:20
    tic
    Mdl = fitcecoc(X,Y,'Learners',t);
    time(j)=toc;
    CVMdl = crossval(Mdl);
    genError = kfoldLoss(CVMdl);
    acc(j) = 1- genError;
end
facc=mean(acc);
fstd=std(acc);
ftime=mean(time);

%% Satimage
clear;clc;
whole =  csvread("D:\Courses\MLDS\AS\dataset\st.csv");
X= whole(:,1:36); 
Y=whole(:,37:end);
for i = 1:size(Y,2)
    Y(Y(:,i)==1) = i;
end
Y = Y(:,1);
    
t = templateSVM('Standardize',true);
for j=1:20
    tic
    Mdl = fitcecoc(X,Y,'Learners',t);
    time(j)=toc;
    CVMdl = crossval(Mdl);
    genError = kfoldLoss(CVMdl);
    acc(j) = 1- genError;
end
facc=mean(acc);
fstd=std(acc);
ftime=mean(time);
%% Scene15
clear;clc;
load("D:\Courses\MLDS\AS\dataset\Scene15.mat");
X= featureMat; 
Y=labelMat;
for i = 1:size(Y,2)
    Y(Y(:,i)==1) = i;
end
Y = Y(:,1);
t = templateSVM('Standardize',true);
tic
Mdl = fitcecoc(X,Y,'Learners',t);
time=toc;
CVMdl = crossval(Mdl);
genError = kfoldLoss(CVMdl);
acc = 1- genError;

%% Caltech101
clear;clc;
load("D:\Courses\MLDS\AS\dataset\Caltech101.mat");
X= featureMat; 
Y=labelMat;
for i = 1:size(Y,2)
    Y(Y(:,i)==1) = i;
end
Y = Y(:,1);
for j=1:20
    tic
    Mdl = fitcecoc(X,Y);
    time(j)=toc;
    CVMdl = crossval(Mdl);
    genError = kfoldLoss(CVMdl);
    acc(j) = 1- genError;
end
facc=mean(acc);
fstd=std(acc);
ftime=mean(time);
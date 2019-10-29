%% Vowel
whole = csvread("D:\Courses\MLDS\AS\dataset\vw.csv");
featureMat= whole(:,1:10); 
labelMat=whole(:,11:end);
save('vw.mat','featureMat','labelMat');
%% ExtendedYaleB dataset
clear;clc;
load("C:\Users\Yizhan\Documents\datasets\ExtendedYaleB.mat");
featureMat = featureMat';
labelMat = labelMat';
dataMat = cat(2,featureMat,labelMat);
dataMat_i = randperm(size(dataMat,1));
dataMat = dataMat(dataMat_i,:); %random partition
featureMat = dataMat(:,1:504);
labelMat = dataMat(:,505:end);
save('ExtendedYaleB.mat','featureMat','labelMat');
%% AR dataset
clear;clc;
load("C:\Users\Yizhan\Documents\datasets\AR.mat");
featureMat = featureMat';
labelMat = labelMat';
dataMat = cat(2,featureMat,labelMat);
dataMat_i = randperm(size(dataMat,1));
dataMat = dataMat(dataMat_i,:); %random partition
featureMat = dataMat(:,1:540);
labelMat = dataMat(:,541:end);
featureMat = normalize(featureMat);
save('AR.mat','featureMat','labelMat');
%% Satimage
whole = csvread("D:\Courses\MLDS\AS\dataset\st.csv");
featureMat= whole(:,1:36); 
labelMat=whole(:,37:end);
featureMat = normalize(featureMat);
save('st.mat','featureMat','labelMat');
%% Scene15 dataset
clear;clc;
load("C:\Users\Yizhan\Documents\datasets\Scene15.mat");
featureMat = featureMat';
labelMat = labelMat';
dataMat = cat(2,featureMat,labelMat);
dataMat_i = randperm(size(dataMat,1));
dataMat = dataMat(dataMat_i,:); %random partition
featureMat = dataMat(:,1:3000);
labelMat = dataMat(:,3001:end);
save('Scene15.mat','featureMat','labelMat');
%% Caltech101
clear;clc;
load("C:\Users\Yizhan\Documents\datasets\Caltech101.mat");
featureMat = featureMat';
labelMat = labelMat';
dataMat = cat(2,featureMat,labelMat);
dataMat_i = randperm(size(dataMat,1));
dataMat = dataMat(dataMat_i,:); %random partition
featureMat = dataMat(:,1:3000);
labelMat = dataMat(:,3001:end);
save('Caltech101.mat','featureMat','labelMat');
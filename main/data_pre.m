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
save('AR.mat','featureMat','labelMat');
%% Scene15 dataset
clear;clc;
load("C:\Users\Yizhan\Documents\datasets\Scene15.mat");
featureMat = featureMat';
labelMat = labelMat';
dataMat = cat(2,featureMat,labelMat);
dataMat_i = randperm(size(dataMat,1));
dataMat = dataMat(dataMat_i,:); %random partition
featureMat = dataMat(:,1:504);
labelMat = dataMat(:,505:end);
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
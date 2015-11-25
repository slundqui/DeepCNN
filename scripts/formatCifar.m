baseDir = '~/workspace/DeepCNNData/cifar/';
rawDir = [baseDir, '/raw/'];
outDir = [baseDir, '/formatted/'];

trainNames = {...
   [rawDir, 'data_batch_1.mat']; ...
   [rawDir, 'data_batch_2.mat']; ...
   [rawDir, 'data_batch_3.mat']; ...
   [rawDir, 'data_batch_4.mat']; ...
   [rawDir, 'data_batch_5.mat']; ...
};

testName = [rawDir, 'test_batch.mat'];

trainData = [];
trainLabels = [];

for batch_i = 1:length(trainNames)
   matName = trainNames{batch_i};
   load(matName);
   [numEx, dataSize] = size(data);
   assert(dataSize == 3072);
   %Data stored as image, x, y, color
   data = reshape(data, [numEx, 32, 32, 3]);
   %DeepCNN expects x, y, color, image
   data = permute(data, [2, 3, 4, 1]);
   trainData = cat(4, trainData, data);
   trainLabels = cat(1, trainLabels, labels);
end

%Normalize and cast to float
trainData = single(trainData);
trainData = (trainData - mean(trainData(:)))/std(trainData(:));

%Bin labels into 10 categories
outTrainLabels = single(zeros(1, 1, 10, length(trainLabels)));

for(ti = 1:length(trainLabels))
   idx = trainLabels(ti) + 1;
   outTrainLabels(:, :, idx, ti) = 1;
end

%Write out to mat
data = trainData;
save([outDir, 'trainData.mat'], 'data', '-v6');
data = outTrainLabels;
save([outDir, 'trainLabels.mat'], 'data', '-v6');

%Read test data
load(testName);

[numEx, dataSize] = size(data);
assert(dataSize == 3072);
%Data stored as image, x, y, color
data = reshape(data, [numEx, 32, 32, 3]);
%DeepCNN expects x, y, color, image
data = permute(data, [2, 3, 4, 1]);
data = single(data);
data = (data - mean(data(:)))/std(data(:));

save([outDir, 'testData.mat'], 'data', '-v6');

outTrainLabels = single(zeros(1, 1, 10, length(labels)));
for(ti = 1:length(labels))
   idx = labels(ti) + 1;
   outTrainLabels(:, :, idx, ti) = 1;
end

data = outTrainLabels;
save([outDir, 'testLabels.mat'], 'data', '-v6');


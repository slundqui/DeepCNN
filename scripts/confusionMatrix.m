dataFilename = '~/workspace/DeepCNNData/cifar/out/testEst_time6400.txt';

dataFile = fopen(dataFilename, 'r');
if(dataFilename< 0)
   disp(['Cost file ', dataFilename, ' does not exist']);
   fflush(stdout);
   keyboard
end

exampleIdx = [];
gt = [];
est = [];
conf = [];

line = fgetl(dataFile);
while(ischar(line))
   split = strsplit(line, ',');
   keyboard
   exampleIdx = [exapleIdx, str2num(split{1})];
   gt = [gt, str2num(split{2})];
   est = [est, str2num(split{3})];
   conf = [conf, str2num(split{4})];
   line = fgetl(dataFile);
end




basedir = '/home/sheng/workspace/DeepCNNData/cifar/out/';

trainCostFilename =    [basedir, '/train_totalCost.txt'];
testCostFilename =    [basedir, '/test_totalCost.txt'];
costOutfilename = [basedir, '/totalCost.png'];

trainCostFile = fopen(trainCostFilename, 'r');
if(trainCostFile < 0)
   disp(['Cost file ', trainCostFilename, ' does not exist']);
   fflush(stdout);
   keyboard
end

testCostFile = fopen(testCostFilename, 'r');
if(testCostFile < 0)
   disp(['Cost file ', testCostFilename, ' does not exist']);
   fflush(stdout);
   keyboard
end

trainTimestep = [];
trainCost = [];
testTimestep = [];
testCost = [];

line = fgetl(trainCostFile);
while(ischar(line))
   split = strsplit(line, ',');
   trainTimestep = [trainTimestep, str2num(split{1})];
   trainCost = [trainCost, str2num(split{2})];
   line = fgetl(trainCostFile);
end

line = fgetl(testCostFile);
while(ischar(line))
   split = strsplit(line, ',');
   testTimestep = [testTimestep, str2num(split{1})];
   testCost = [testCost, str2num(split{2})];
   line = fgetl(testCostFile);
end

h = figure;
hold on
plot(trainTimestep, trainCost, 'b', 'lineWidth', 5);
plot(testTimestep, testCost, 'r', 'lineWidth', 5);
hold off

title('Energy vs timestep', 'FontSize', 28);
xlabel('timestep', 'FontSize', 28);
ylabel('Energy', 'FontSize', 28);

l = legend('Train', 'Test');
set(l, 'FontSize', 16);

saveas(h, costOutfilename);

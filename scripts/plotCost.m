basedir = '/home/sheng/workspace/DeepCNNData/xor';
costFilename =    [basedir, '/totalCost.txt'];
costOutfilename = [basedir, '/totalCost.png'];

costFile = fopen(costFilename, 'r');
if(costFile < 0)
   disp(['Cost file ', costFile, ' does not exist']);
   keyboard
end

timestep = [];
cost = [];


line = fgetl(costFile);
while(ischar(line))
   split = strsplit(line, ',');
   timestep = [timestep, str2num(split{1})];
   cost = [cost, str2num(split{2})];
   line = fgetl(costFile);
end

h = plot(timestep, cost);
saveas(h, costOutfilename);
   

% Set hyper parameters and initialize in this script
% Training in 'rbmtrain' function
% Load MNIST data
train_x = []; test_x = [];
load('../data/mnist_uint8.mat');
train_x = double(train_x)' / 255;
test_x = double(test_x)' / 255;

num_train = 10000;  % size(train_x, 2);
num_test  = 50000;  % size(test_x, 2);

x = [train_x test_x];
y = [train_y;test_y];

train_x = x(:, 1:num_train);
train_y = y(1:num_train, :);
test_x  = x(:, num_train+1:num_train+num_test);
test_y  = y(num_train+1:num_train+num_test, :);

load('../data/background_image', 'T');
back = double(T')/255;

test_back = back(:, num_train+1:num_train+num_test);
back = back(:, 1:num_train);

%------------ hyper parameters ----------------
mix   = 0.5;
pbias = 0.05;
model.sizes = [784, 1200];
opts.numepochs = 30;
opts.batchsize = 200;
opts.alpha     = 0.02;
opts.num_cd    = 1;
opts.initialmomentum = 0.5;
opts.finalmomentum   = 0.9;
opts.plambda   = 0.3;
opts.pbias     = pbias;
opts.l2penalty = 0.0001;
opts.noiseLevel = 0.0;      % noise background
opts.imageLevel = mix;      % image background
%         if mix == 0.5
%             mchar = 'hf';
%         end
%         filename = ['../weights/small_image_checkPoint_' mchar '_p' num2str(pbias*100)];
%-----------------------------------------------
filename = '../model/pretrain_result.mat';
model = rbmsetup(model, train_x, opts);
evalData.test_x = test_x;
evalData.back = test_back;
model = rbmtrain(model, train_x', back', opts, @evalFunc, evalData, filename);
save(filename, 'model');




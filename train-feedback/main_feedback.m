function main_feedback(weights, name)
% This function is used to set option and initialize
% Training process in train_feedback
%-------------------------- option setting --------------------------------
if ~exist('experiment', 'var')
    experiment  = 'image';
end
fprintf('experiment: %s\n', experiment);

if ~exist('weights', 'var')
    weights = '../model/pretrain_result';       % name of pretrain result
end
fprintf('weights: %s\n', weights);

if ~exist('name', 'var')
    name = ['../model/train_result' '.mat'];    % name of result generated in this function
end
fprintf('save file name: %s\n', name);

%------------------- prepare data and initialize --------------------------
num_train = 10000;
num_test  = 10000;
[t_train_x, train_y, t_test_x, test_y] = prepare_data('../data/mnist_uint8', 1:num_train, num_train+1:num_train+num_test);
if strcmp(experiment, 'noise')
    train_x = max(t_train_x, rand(num_train, 784));
    test_x  = max(t_test_x, rand(num_test, 784));
elseif strcmp(experiment, 'image')
    load('../data/background_image.mat', 'T');
    back = double(T);
    back = back / max(max(back));
    train_x = max(back(1:num_train, :), t_train_x);
    test_x  = max(back(num_train+1:num_train+num_test, :), t_test_x);
end

opts = opts_setup();

m = load(weights, 'model');
model = model_setup(train_x, train_y, opts);
model.W = m.model.W;
model.b{1} = m.model.b;
model.b{2} = m.model.c;

opts = opts_setup();

[model, err_test] = train_feedback(model, train_x, t_train_x, train_y, opts, test_x, t_test_x, test_y, name);
save(name, 'opts', 'model', 'err_test');
end


function opts = opts_setup()
opts.size             =   [1200];
opts.numepochs        =   1;
opts.batchsize        =   200;
opts.alpha            =   12;
opts.ratedecay        =   1;
opts.minalpha         =   1e-6;
opts.mstep            =   100;
opts.mbeginpoint      =   1;
opts.finalmomentum    =   0.9;
opts.initialmomentum  =   0.5;
opts.wdecay           =   0;
end

function model = model_setup(x, y, opts)
v = size(x, 2);
f = size(y, 2);
model.sizes    = [v, opts.size, f];
num_layer     = numel(model.sizes);
model.W  = 0.05 * randn(model.sizes(2), model.sizes(1));
model.U  = 0.05 * randn(model.sizes(3), model.sizes(2));
model.vU = 0.05 * randn(model.sizes(3), model.sizes(2));
for n = 1:num_layer-1
    model.b{n}  = zeros(model.sizes(n), 1);
    model.vb{n} = zeros(model.sizes(n), 1);
end
end

function [ err_rate ] = test_background(experiment, ...
    index, ...          % test cases to run
    threshold, ...      % threshold after reconstruction
    ite, ...            % iterations for each guess
    classifier, ...     % classifier model: mlp or cnn
    denoiser, ...       % denoiser model
    noise_level ...     % background noise level
    )


    %--------------------------- Default Setting ---------------------------
    if ~exist('experiment', 'var')
        experiment = 'rand';
    end
    fprintf('experiment: mnist-background- %s\n', experiment);
    
    if ~exist('index', 'var')
        index = 60001:70000;
    end
    fprintf('contruct %d %s examples from %d MNIST examples\n', floor(2/length(index)), experiment, length(index));

    if ~exist('threshold', 'var')
        threshold = 0.32;
    end
    fprintf('threshold: %f\n', threshold);

    if ~exist('ite', 'var')
        ite = 1;
    end
    fprintf('iteration number: %d\n', ite);

    if ~exist('classifier', 'var')
        classifier = 'mlp';
    end
    fprintf('classifier: %s\n', classifier);

    if ~exist('denoiser', 'var')
        denoiser = '../model/noise_p5_model.mat'; % feedback_hf_p5_model.mat';    
    end
    fprintf('denoiser: %s\n', denoiser);

    if ~exist('noise_level', 'var')
        noise_level = 1.0;    
    end
    fprintf('noise_level: %f\n', noise_level);

    %--------------------------- Prepare Data ---------------------------
    rng(6);

    [t_x, t_y, ~, ~] = prepare_data('../data/mnist_uint8', index, index);
    num  = size(t_x, 1);
    if strcmp(experiment, 'image')
        load('../data/background_image.mat');
        back = double(T);
        back = back / max(max(back));
        back = back(index, :);
        img  = max(t_x, noise_level * back);            % generate mnist-back-img
    elseif strcmp(experiment, 'rand')
        img  = max(t_x, noise_level * rand(size(t_x))); % generate mnist-back-rand
    end
    
    [~, I_y]  = max(t_y, [], 2);  % label of the digit

    figure(1);
    clf;
    visualize(img(1:16, :)');

    %--------------------------- Load Models ---------------------------
     % the classifier
    if strcmp(classifier, 'cnn')
        load('../model/cnn.mat', 'cnn');
        cnet = cnn;
        f_softmax = @cnn_softmax;
    elseif strcmp(classifier, 'mlp')
        load('../model/mlp.mat', 'mlp');
        cnet = mlp;
        f_softmax = @FF_softmax;
        
    end

    % the denoiser
    load(denoiser, 'model');

    %-------------- run iterations for each guess ---------------------

    disp('***** Run iterations....');
    rng(12);

    for guess = 1:10
        fprintf('\tGuess: %d\n', guess);
        y = zeros(num, 10);
        y(:, guess) = 1;
        x = img;
        for i = 1:ite
            [v, ~, ~, ~, ~] = ff(x, y, model);
            x               = img .* (v > threshold);
            vis(guess, i, :, :) = x;
        end
    end
    
    %-------------- run classifier on the iterations ---------------------
    disp('***** Run classifier....');

    vis_reshape = reshape(vis, [10*ite*num, 784]);
    [ent, ind_label] = f_softmax(cnet, vis_reshape);
    [~, I_lab] = max(ind_label, [], 2);
    label = reshape(I_lab-1, [10, ite, num]);
    score = reshape(ent, [10, ite, num]);
    
    %------------------ final classification ----------------------
    match_mat = zeros(10, ite, num);
    for guess = 1:10
        match_mat(guess, :, :) = (label(guess, :, :) == (guess - 1));
    end

    match_score = match_mat .* score;
    match_score(match_score == 0) = Inf;

    for n = 1:ite
        match_score_step = squeeze(match_score(:, n, :));
        [min_score, I_lab]  = min(match_score_step, [], 1);
        I_lab(isinf(min_score)) = 0;
        err(n) = sum(I_lab ~= I_y') / num;
    end
    
    err_rate = err(end);
    fprintf('Final Result:\n');
    fprintf('\t err_rate: %f%%\n', 100*err_rate);

end


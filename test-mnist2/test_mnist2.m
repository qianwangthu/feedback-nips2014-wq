function [ acc_both, acc_one ] = test_mnist2(...
    index, ...          % test cases to run
    threshold, ...      % threshold after reconstruction
    ite, ...            % iterations for each guess
    classifier, ...     % classifier model: mlp or cnn
    denoiser, ...       % denoiser model
    noise_level ...     % background noise level
    )


    %--------------------------- Default Setting ---------------------------
    if ~exist('index', 'var')
        index = 1:10000;
    end
    fprintf('contruct %d MNIST-2 examples from %d MNIST examples\n', floor(2/length(index)), length(index));

    if ~exist('threshold', 'var')
        threshold = 0.3;
    end
    fprintf('threshold: %f\n', threshold);

    if ~exist('ite', 'var')
        ite = 5;
    end
    fprintf('iteration number: %d\n', ite);

    if ~exist('classifier', 'var')
        classifier = 'mlp';
    end
    fprintf('classifier: %s\n', classifier);

    if ~exist('denoiser', 'var')
        denoiser = '../model/feedback_hf_p5_model.mat';    
    end
    fprintf('denoiser: %s\n', denoiser);

    if ~exist('noise_level', 'var')
        noise_level = 1.0;    
    end
    fprintf('noise_level: %f\n', noise_level);

    %--------------------------- Prepare Data ---------------------------
    rng(6);

    [t_x, t_y, ~, ~] = prepare_data('../data/mnist_uint8', index, index);
    num  = floor(size(t_x, 1) / 2);

    img = max(t_x(1:num, :), t_x(num+1:2*num, :));  % combine two digits
    img = max(img, noise_level * rand(size(img)));  % add background noise

    t_y1 = t_y(1:num, :);
    t_y2 = t_y(num+1:2*num, :);
    [~, I_y1]  = max(t_y1, [], 2);  % label of the first digit
    [~, I_y2]  = max(t_y2, [], 2);  % label of the second digit

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

    [first_match_score, first_match_step] = min(match_score, [], 2);
    first_match_score = squeeze(first_match_score);

    [min_score_1, I_lab_1]  = min(first_match_score, [], 1);
    for m = 1:num
        first_match_score(I_lab_1(m), m) = Inf;
    end
    [min_score_2, I_lab_2]  = min(first_match_score, [], 1);
    I_lab_1(isinf(min_score_1)) = 0;
    I_lab_2(isinf(min_score_2)) = 0;

    c_1 = (I_y1 == I_lab_1') | (I_y1 == I_lab_2');  % hit the first digit
    c_2 = (I_y2 == I_lab_1') | (I_y2 == I_lab_2');  % hit the second digit

    acc_both = sum(c_1 & c_2) / num;    % accuracy for hitting both digits
    acc_one  = sum(c_1 | c_2) / num;    % accuracy for hitting at least one digit

    fprintf('Final Result:\n');
    fprintf('\t accuracy_both: %f%%, accuracy_one: %f%%', 100*acc_both, 100*acc_one);

end


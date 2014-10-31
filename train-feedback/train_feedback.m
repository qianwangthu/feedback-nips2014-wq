function [model, err_test] = train_feedback(model, x, t_x, y, opts, test_x, test_tx, test_y, filename)

assert(isfloat(x), 'x must be a float');
m = size(x, 1);

num_layer      = numel(model.sizes);
numbatches     = m / opts.batchsize;
model.alpha    = opts.alpha;
model.momentum = opts.initialmomentum;
assert(rem(numbatches, 1) == 0, 'numbatches not integer');
err_train   = [];
err_test    = [];
for i = 1 : opts.numepochs
    kk = randperm(m);
    if i>opts.mbeginpoint && i < opts.mstep + opts.mbeginpoint
        model.momentum = model.momentum - (1/opts.mstep) * (opts.initialmomentum - opts.finalmomentum);
    end
    if ~mod(i, 20)
        model.alpha = model.alpha * 0.3;    % learning rate
    end
    for l = 1 : numbatches
        batch_x  = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
        batch_y  = y(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
        batch_tx = t_x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
        
        [dU, db, err_batch] = bp(model, batch_x, batch_tx, batch_y, opts);
        disp(['batch ' num2str(l) ' r_err ' num2str(err_batch)]);

        model.vU = model.momentum * model.vU + model.alpha * dU - opts.wdecay * model.U;
        model.U  = model.U + model.vU;
        for n = 1:num_layer - 1
            model.vb{n} = model.momentum * model.vb{n} + model.alpha * db{n} - opts.wdecay * model.b{n};
            model.b{n}  = model.b{n} + model.vb{n};
        end
    end
    [test_vis, ~, ~, ~, ~] = ff(test_x, test_y, model);
    err = sum(sum( (test_vis - test_tx).^2 )) / size(test_tx, 1);
    err_test = [err_test err];
    fprintf('epoch %i/%i. Average reconstruct testing err: %f\n', i, opts.numepochs, err);
    if ~mod(i, 10)
        save([filename ' epoch_' num2str(i)], 'err_test', 'model', 'opts');
    end
end

end
function rbm = rbmtrain(rbm, x, back, opts, evalFunc, evalData, filename)
    assert(isfloat(x), 'x must be a float');
    m = size(x, 1);
    numbatches = m / opts.batchsize;
    numBack = size(back, 1);
    
    assert(rem(numbatches, 1) == 0, 'numbatches not integer');
    avg_h = [];
    errTrace_batch = [];
    spaTrace_batch = [];
    errTrace_epoch = [];
    spaTrace_epoch = [];
   
    for epoch = 1 : opts.numepochs
        kk = randperm(m);
           
        err = 0;
        sparsity = 0;
        if epoch < 5
            momentum = opts.initialmomentum;
        else
            momentum = opts.finalmomentum;
        end
        
        for l = 1 : numbatches
            batch = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
            
            v1 = batch;
            v1 = v1 + opts.noiseLevel*randn(size(batch));
            I = 1:opts.batchsize;
            v1(I, :) = max( v1(I, :), opts.imageLevel*back(randperm(numBack, length(I)), :) );
            % if rem(l, 2) ~= 0
            %     v1 = max( v1, opts.imageLevel*back(randperm(numBack, opts.batchsize), :) );
            % end
            
            h1 = sigm(repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W');
            h1_state = rand(size(h1)) < h1; 
            v2 = sigm(repmat(rbm.b', opts.batchsize, 1) + h1_state * rbm.W);
            h2 = sigm(repmat(rbm.c', opts.batchsize, 1) + v2 * rbm.W');
            
            h1_recon = sigm(repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W');
            recon    = sigm(repmat(rbm.b', opts.batchsize, 1) + h1_recon * rbm.W);

            c1 = h1' * v1;
            c2 = h2' * v2;
            
            dw  = (c1 - c2)     / opts.batchsize;
            dvb = sum(v1 - v2)' / opts.batchsize;
            dhb = sum(h1 - h2)' / opts.batchsize;
            
            if isempty(avg_h)
                avg_h = sum(h1)' / opts.batchsize;
            else
                avg_h = 0.9 * avg_h + 0.1 * sum(h1)' / opts.batchsize;
            end
            
            
            dhb_sparsity = opts.plambda * (opts.pbias - avg_h);
%             dhb_sparsity = 0;

            rbm.vW = momentum * rbm.vW + opts.alpha * (dw - opts.l2penalty * rbm.W);
            rbm.vb = momentum * rbm.vb + opts.alpha * dvb;
            rbm.vc = momentum * rbm.vc + opts.alpha * (dhb + dhb_sparsity);     % sparse constrain

            rbm.W = rbm.W + rbm.vW;
            rbm.b = rbm.b + rbm.vb;
            rbm.c = rbm.c + rbm.vc;

            thisErr = sum(sum((batch - recon).^2))/ opts.batchsize;
            thisSpa = mean(h1(:));
            
            errTrace_batch(end+1) = thisErr;
            spaTrace_batch(end+1) = thisSpa;

            err      = err + thisErr;
            sparsity = sparsity + thisSpa;
        end
        
        err = err / numbatches;
        sparsity = sparsity / numbatches;
        
        errTrace_epoch = [errTrace_epoch err];
        spaTrace_epoch = [spaTrace_epoch sparsity];
        disp(['epoch ' num2str(epoch) '/' num2str(opts.numepochs)  '. Average reconstruction error: '...
            num2str(err) '. sparsity: ' num2str(sparsity)]);
       
        
        model = rbm;
        evalFunc(model, evalData);
        if mod(epoch, 10)==0
            save(sprintf('%s_epoch=%d.mat',filename, epoch), 'model', 'opts', 'errTrace_epoch', 'spaTrace_epoch'); 
        end
    end
end

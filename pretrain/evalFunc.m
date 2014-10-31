function evalFunc(model, evalData)

    test_x = evalData.test_x;
    back = evalData.back;
    
    %----------------------------------
    figure(1);
    clf;
 
    visualize(model.W(1:100,:)');
    title('features');
    
    %----------------------------------
    figure(2);
    clf;
    
    N = 100;
    x0 = test_x(:, 1:N);
    x = x0;
    h = sigm(bsxfun(@plus, model.W *x, model.c));
    hclean = h;
    y = sigm(bsxfun(@plus, model.W'*h, model.b));
    
    err0 = sum(sum((y-x0).^2))/N;
    err1 = sum(sum((y-x).^2))/N;
    spa = mean(h(:));
    fprintf('[clean data] reconstruction err: (%f, %f), sparsity: %f\n', err0, err1, spa );
    
    subplot(1,2,1);
    visualize(x);
    title('input');
    subplot(1,2,2);
    visualize(y);
    title(sprintf('reconstruction: err = (%f, %f)', err0, err1));
    
    %----------------------------------
    figure(3);
    clf;
    
    N = 100;
    x0 = test_x(:, 1:N);
    x  = max(x0, rand(size(x0)));
    h = sigm(bsxfun(@plus, model.W *x, model.c));
    hnoise = h;

    y = sigm(bsxfun(@plus, model.W'*h, model.b));
    
    err0 = sum(sum((y-x0).^2))/N;
    err1 = sum(sum((y-x).^2))/N;
    spa = mean(h(:));
    fprintf('[noisy_back] reconstruction err: (%f, %f), sparsity: %f\n', err0, err1, spa );
    
    subplot(1,2,1);
    visualize(x);
    title('input');
    subplot(1,2,2);
    visualize(y);
    title(sprintf('reconstruction: err = (%f, %f)', err0, err1));
    
    %----------------------------------
    figure(4);
    clf;
    
    N = 100;
    x0 = test_x(:, 1:N);
    x = max(x0, back(:, 1:N));
    h = sigm(bsxfun(@plus, model.W *x, model.c));
    himg = h;
    y = sigm(bsxfun(@plus, model.W'*h, model.b));
    
    err0 = sum(sum((y-x0).^2))/N;
    err1 = sum(sum((y-x).^2))/N;
    spa = mean(h(:));
    fprintf('[image_back] reconstruction err: (%f, %f), sparsity: %f\n', err0, err1, spa );
    
    subplot(1,2,1);
    visualize(x);
    title('input');
    subplot(1,2,2);
    visualize(y);
    title(sprintf('reconstruction: err = (%f, %f)', err0, err1));
    %----------------------------------
    save('hlook', 'hclean', 'hnoise', 'himg');
    drawnow;
    
end
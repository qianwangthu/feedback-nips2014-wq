function rbm = rbmsetup(rbm, x, opts)
    
    % rbm.sizes(1) ~ num_vis
    % rbm.sizes(2) ~ num_hid

    rbm.W  = 0.02 * randn(rbm.sizes(2), rbm.sizes(1));
    rbm.vW = zeros(rbm.sizes(2), rbm.sizes(1));

    rbm.b  = zeros(rbm.sizes(1), 1);
    rbm.vb = zeros(rbm.sizes(1), 1);

    rbm.c  = zeros(rbm.sizes(2), 1);
    rbm.vc = zeros(rbm.sizes(2), 1);
end

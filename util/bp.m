function [dU, db, err] = bp(model, x, t_x, y, opts)
[vis, mask_h, mask_v, hid_r, back] = ff(x, y, model);

err  = sum(sum((t_x - vis).^2)) / opts.batchsize;
dEdv = (t_x - vis) .* mask_v;
dEdo = dEdv * model.W';
dEdh = (dEdo .* back) .* mask_h;
dEdb = dEdo .* hid_r;

dU = y' * (dEdb .* back .* (1-back) ) / opts.batchsize;
db{1}  = mean(dEdv, 1)';
db{2}  = mean(dEdh, 1)';

end


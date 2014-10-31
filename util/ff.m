function [vis, mask_h, mask_v, hid_r, back] = ff(x, y, model)
    hid_r           = bsxfun(@plus, x * model.W', model.b{2}');
    [hid_r, mask_h] = sigmoid(hid_r);
    back            = sigm(y * model.U);
    hid             = back .* hid_r;
    vis             = bsxfun(@plus, hid * model.W, model.b{1}');
    [vis, mask_v]   = sigmoid(vis);
end


function [res, mask] = sigmoid(h)
res  = sigm(h);
mask = res .* (1-res);
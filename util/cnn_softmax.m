function [score, out] = cnn_softmax(cnn, x)
x = reshape(x, [size(x, 1), 28 28]);
x = permute(x, [2 3 1]);
net = cnnff(cnn, x);
out = net.o';
score = -sum((net.o .* log(net.o)));
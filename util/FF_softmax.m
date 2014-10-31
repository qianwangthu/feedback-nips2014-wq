% W{i} is the weight from layer i to layer i+1
% data and target dimension: sample * data
% err: cros-entropy
% dW: dW{i} is the err derivative on 
function [score, out] = FF_softmax(net, data)
    W = net.W;
    b = net.b;
	numLayer = length(W) + 1;
	layerSize = [];
	for i = 1:(numLayer-1)
		layerSize(i) = size(W{i}, 1);
	end
	layerSize(numLayer) = size(W{numLayer-1}, 2);

	outDim = layerSize(numLayer);
	N = size(data,1);

	%----------- Forward Propogation -------------
	x = {};
	x{1} = data;
	for i = 1:(numLayer-2)
		x{i+1} = 1./(1 + exp(-1*( x{i}*W{i} + repmat(b{i},N,1)) ));
	end

	w_class = W{numLayer-1};
    b_class = b{numLayer-1};

	out = exp(x{numLayer-1}*w_class + repmat(b_class,N,1));
	out = out./repmat(sum(out,2),1,outDim);

	%------------ Evaluate Entropy ---------------
	score = -sum((out.* log(out)), 2);
end
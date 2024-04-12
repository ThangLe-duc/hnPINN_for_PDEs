% Model Function
function dlU = model_PStr_2NN_U(parameters,dlX,dlY)

dlXY = [dlX;dlY];
numLayers = numel(fieldnames(parameters));

% First fully connect operation.
weights = parameters.fc1.Weights;
bias = parameters.fc1.Bias;
dlU = fullyconnect(dlXY,weights,bias);

% tanh and fully connect operations for remaining layers.
for i=2:numLayers
    name = "fc" + i;
    dlU = tanh(dlU);
    weights = parameters.(name).Weights;
    bias = parameters.(name).Bias;
    dlU = fullyconnect(dlU, weights, bias);
end
dlU = dlU.*dlX;

end
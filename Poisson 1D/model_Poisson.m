% Model Function
function dlU = model_Poisson(parameters,dlX)

numLayers = numel(fieldnames(parameters));

% First fully connect operation.
weights = parameters.fc1.Weights;
bias = parameters.fc1.Bias;
dlU = fullyconnect(dlX,weights,bias);

% tanh and fully connect operations for remaining layers.
for i=2:numLayers
    name = "fc" + i;
    dlU = tanh(dlU);
    weights = parameters.(name).Weights;
    bias = parameters.(name).Bias;
    dlU = fullyconnect(dlU, weights, bias);
end

end
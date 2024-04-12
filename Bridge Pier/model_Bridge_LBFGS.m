% Model Function
function dlU = model_Bridge_LBFGS(parameters,dlX)

numLayers = numel(fieldnames(parameters))/2;

% First fully connect operation.
weights = parameters.fc1_Weights;
bias = parameters.fc1_Bias;
dlU = fullyconnect(dlX,weights,bias);

% tanh and fully connect operations for remaining layers.
for i=2:numLayers
    name = "fc" + i;
    dlU = tanh(dlU);
    weights = parameters.(name + "_Weights");
    bias = parameters.(name + "_Bias");
    dlU = fullyconnect(dlU, weights, bias);
end

end
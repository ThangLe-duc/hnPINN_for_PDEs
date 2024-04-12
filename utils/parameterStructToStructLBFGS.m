function parametersLBFGS = parameterStructToStructLBFGS(parameters)
% parameterStructToStructLBFGS converts a struct of learnable parameters to a
% struct required by LBFGS algorithm.
parametersLBFGS = struct;

% Parameter names.
parameterNames = fieldnames(parameters);

% Determine parameter sizes.
numFields = numel(parameterNames);
for i = 1:numFields
    name = "fc"+i;
    parametersLBFGS.(name + "_Weights") = parameters.(parameterNames{i}).Weights;
    parametersLBFGS.(name + "_Bias") = parameters.(parameterNames{i}).Bias;
end

end
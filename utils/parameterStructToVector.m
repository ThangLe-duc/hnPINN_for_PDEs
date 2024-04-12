function [parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters)
% parameterStructToVector converts a struct of learnable parameters to a
% vector and also returns the parameter names and sizes.

% Parameter names.
parameterNames = fieldnames(parameters);

% Determine parameter sizes.
numFields = numel(parameterNames);
parameterSizes = cell(1,numFields);
for i = 1:numFields
    parameter = parameters.(parameterNames{i});
    parameterSizes{i} = size(parameter);
end

% Calculate number of elements per parameter.
numParameterElements = cellfun(@prod,parameterSizes);
numParamsTotal = sum(numParameterElements);

% Construct vector
parametersV = zeros(numParamsTotal,1,'like',parameters.(parameterNames{1}));
count = 0;

for i = 1:numFields
    parameter = parameters.(parameterNames{i});
    numElements = numParameterElements(i);
    parametersV(count+1:count+numElements) = parameter(:);
    count = count + numElements;
end

end
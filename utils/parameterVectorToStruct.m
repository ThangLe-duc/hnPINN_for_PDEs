function parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes)
% parameterVectorToStruct converts a vector of parameters with specified
% names and sizes to a struct.

parameters = struct;
numFields = numel(parameterNames);
count = 0;

for i = 1:numFields
    numElements = prod(parameterSizes{i});
    parameter = parametersV(count+1:count+numElements);
    parameter = reshape(parameter,parameterSizes{i});
    parameters.(parameterNames{i}) = parameter;
    count = count + numElements;
end

end
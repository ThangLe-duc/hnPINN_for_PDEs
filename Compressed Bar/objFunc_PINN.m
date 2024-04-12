function [loss,gradientsV] = objFunc_PINN(parametersV,dlX,dlX0BC1,dlU0BC1,dlX0BC2,dlU0BC2,dlXTest,UTest,parameterNames,...
    parameterSizes,E,A,L,g)

global TestErr;
% Convert parameters to structure of dlarray objects.
parametersV = dlarray(parametersV);
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
% Evaluate model gradients and loss.
[gradients,loss] = dlfeval(@modelGradients_AxialBar_PINN,parameters,dlX,dlX0BC1,dlU0BC1,dlX0BC2,dlU0BC2,E,A,L,g);
% Return loss and gradients for fmincon.
gradientsV = parameterStructToVector(gradients);
gradientsV = extractdata(gradientsV);
loss = extractdata(loss);
gradientsV = double(gradientsV);
loss = double(loss);
% Calculate error on test set
dlUPred = model_AxialBar_LBFGS(parameters,dlXTest);
err = norm(extractdata(dlUPred) - extractdata(UTest)) / norm(extractdata(UTest));
TestErr = [TestErr; err];
end
function [loss,gradientsV] = objFunc_nPINN(parametersV,dlX,dlX0BC1,dlU0BC1,dlX0BC2,dlU0BC2,dlXTest,UTest,parameterNames,...
    parameterSizes,E,L,q0,scaleX,scaleU,alpha)

global TestErr

% Convert parameters to structure of dlarray objects.
parametersV = dlarray(parametersV);
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
% Evaluate model gradients and loss.
[gradients,loss] = dlfeval(@modelGradients_Bridge_nPINN,parameters,dlX,dlX0BC1,dlU0BC1,dlX0BC2,dlU0BC2,E,L,q0,scaleX,scaleU,alpha);
% Return loss and gradients for fmincon.
gradientsV = parameterStructToVector(gradients);
gradientsV = extractdata(gradientsV);
loss = extractdata(loss);
gradientsV = double(gradientsV);
loss = double(loss);
% Calculate error on test set
dlXTestInp = dlXTest/scaleX;
dlUPred = model_Bridge_LBFGS(parameters,dlXTestInp);
dlUPred = alpha*scaleU*dlUPred;
err = norm(extractdata(dlUPred) - extractdata(UTest)) / norm(extractdata(UTest));
TestErr = [TestErr; err];
end
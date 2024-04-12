function [loss,gradientsV] = objFunc_Poisson_nPINN(parametersV,dlX,dlX0BC1,dlU0BC1,dlXTest,UTest,...
    parameterNames,parameterSizes,k,scaleX,scaleU,alpha)

global TestErr

% Convert parameters to structure of dlarray objects.
parametersV = dlarray(parametersV);
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
% Evaluate model gradients and loss.
[gradients,loss] = dlfeval(@modelGradients_Poisson_nPINN,parameters,dlX,dlX0BC1,dlU0BC1,k,scaleX,scaleU,alpha);
% Return loss and gradients for fmincon.
gradientsV = parameterStructToVector(gradients);
gradientsV = double(extractdata(gradientsV));
loss = double(extractdata(loss));
% Calculate error.
dlXTestInp = dlXTest/scaleX;
dlUPred = model_Poisson_LBFGS(parameters,dlXTestInp);
dlUPred = alpha*scaleU*dlUPred;
err = norm(extractdata(dlUPred) - extractdata(UTest)) / norm(extractdata(UTest));
TestErr = [TestErr; err];
end
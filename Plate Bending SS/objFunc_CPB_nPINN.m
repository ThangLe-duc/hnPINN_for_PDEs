function [loss,gradientsV] = objFunc_CPB_nPINN(parametersV,dlX,dlY,dlX0BCx,dlY0BCx,dlU0BCx,dlU0BCx2,dlX0BCy,dlY0BCy,dlU0BCy,...
    dlU0BCy2,dlXTest,dlYTest,UTest,parameterNames,parameterSizes,a,b,muy,q0,m,n,D,scaleX,scaleY,scaleU,alpha)

global TestErr

% Convert parameters to structure of dlarray objects.
parametersV = dlarray(parametersV);
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);

% Evaluate model gradients and loss.
[gradients,loss] = dlfeval(@modelGradients_CPB_nPINN,parameters,dlX,dlY,dlX0BCx,dlY0BCx,dlU0BCx,dlU0BCx2,dlX0BCy,dlY0BCy,...
    dlU0BCy,dlU0BCy2,a,b,muy,q0,m,n,D,scaleX,scaleY,scaleU,alpha);

% Return loss and gradients for fmincon.
gradientsV = parameterStructToVector(gradients);
gradientsV = double(extractdata(gradientsV));
loss = double(extractdata(loss));

% Calculate error.
dlXTestInp = dlXTest/scaleX;
dlYTestInp = dlYTest/scaleY;
dlUPred = model_CPB_LBFGS(parameters,dlXTestInp,dlYTestInp);
dlUPred = alpha*scaleU*dlUPred;
err = norm(extractdata(dlUPred) - extractdata(UTest)) / norm(extractdata(UTest));
TestErr = [TestErr; err];
end
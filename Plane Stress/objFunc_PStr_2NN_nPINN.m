function [loss,gradientsUV] = objFunc_PStr_2NN_nPINN(parametersUV,dlX,dlY,dlX0BCx1,dlY0BCx1,dlU0BCx1,dlX0BCy1,dlY0BCy1,dlU0BCy1,dlX0Nxx,dlY0Nxx,dlU0Nxx,...
    dlX0Nyy,dlY0Nyy,dlU0Nyy,dlXTest,dlYTest,UTest,VTest,parameterNamesU,parameterSizesU,parameterNamesV,parameterSizesV,a,b,muy,E,...
    scaleX,scaleY,scaleU,scaleV,alpha_u,alpha_v)

global TestErrU TestErrV

% Convert parameters to structure of dlarray objects.
parametersUV = dlarray(parametersUV);
parametersUvec = parametersUV(1:floor(numel(parametersUV)/2));
parametersVvec = parametersUV(floor(numel(parametersUV)/2)+1:floor(numel(parametersUV)));
parametersU = parameterVectorToStruct(parametersUvec,parameterNamesU,parameterSizesU);
parametersV = parameterVectorToStruct(parametersVvec,parameterNamesV,parameterSizesV);

% Evaluate model gradients and loss.
[gradientsU,gradientsV,loss] = dlfeval(@modelGradients_PStr_2NN_nPINN_Type1,parametersU,parametersV,dlX,dlY,dlX0BCx1,dlY0BCx1,dlU0BCx1,...
    dlX0BCy1,dlY0BCy1,dlU0BCy1,dlX0Nxx,dlY0Nxx,dlU0Nxx,dlX0Nyy,dlY0Nyy,dlU0Nyy,a,b,muy,E,scaleX,scaleY,scaleU,scaleV,alpha_u,alpha_v);

% Return loss and gradients for fmincon.
gradientsUvec = parameterStructToVector(gradientsU);
gradientsUvec = double(extractdata(gradientsUvec));
gradientsVvec = parameterStructToVector(gradientsV);
gradientsVvec = double(extractdata(gradientsVvec));
gradientsUV = [gradientsUvec; gradientsVvec];
loss = double(extractdata(loss));

% Calculate error.
dlXTestInp = dlXTest;
dlYTestInp = dlYTest;
dlUPred = model_PStr_LBFGS_2NN_U(parametersU,dlXTestInp,dlYTestInp);
dlVPred = model_PStr_LBFGS_2NN_V(parametersV,dlXTestInp,dlYTestInp);
dlUPred = alpha_u*scaleU*dlUPred;
dlVPred = alpha_v*scaleV*dlVPred;
errU = norm(extractdata(dlUPred) - UTest) / norm(UTest);
errV = norm(extractdata(dlVPred) - VTest) / norm(VTest);
TestErrU = [TestErrU; errU];
TestErrV = [TestErrV; errV];
end
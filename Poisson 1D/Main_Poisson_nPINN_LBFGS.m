%% Main function for implementing hnPINN to solve 1D Poisson problem by LBFGS optimizer
%% Programmer: Thang Le-Duc
%  Emails: le.duc.thang0312@gmail.com

%% Begin main function
clear all, close all, clc
rng('default')
addpath('./utils')
global TestErr

%% Initially physical model
k = 6;
L = 1;

%% Normalized physical model
scaleX = L;
scaleU = k^2*pi^2*scaleX^2;
% alpha = 1;
alpha = 0.071;

%% Generate Training Data
% Select points to enforce boundary conditions
x0BC = [0 L];
u0BC = [0 0];
% Normalize boundary conditions
X0BC = x0BC/scaleX;
U0BC = u0BC;
% Select points to enforce the network output
numIntColPoints = 1000;
dataX = L*rand(numIntColPoints,1);
dataX = dataX/scaleX;
ds = arrayDatastore(dataX);
% Convert boundary conditions to dlarray
dlX = dlarray(dataX','CB');
dlX0BC = dlarray(X0BC,'CB');
dlU0BC = dlarray(U0BC,'CB');
% Calculate true values.
numPredictions = 1001;
XTest = linspace(0,L,numPredictions);
dlXTest = dlarray(XTest,'CB');
UTest = sin(k*pi*dlXTest);

%% Specify Training Options
MaxFuncEval = 2800;
% Optimize using the fmincon optmizer with the LBFGS algorithm
options = optimoptions('fmincon', ...
    'HessianApproximation','lbfgs', ...
    'MaxIterations',2*MaxFuncEval, ...
    'MaxFunctionEvaluations',MaxFuncEval, ...
    'OptimalityTolerance',1e-16, ...
    'SpecifyObjectiveGradient',true);
options.Display = 'iter';

%% Train PINN model
numTest = 10;
TestErr_LBFGS = struct;
for i = 1:numTest
% Define PINN Model
numLayers = 3;
numNeurons = 40;
parameters = struct;
sz = [numNeurons 1];
parameters.fc1_Weights = initializeGlorot(sz,numNeurons,1);
parameters.fc1_Bias = initializeZeros([numNeurons 1],'double');
for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;
    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parameters.(name + "_Weights") = initializeGlorot(sz,numNeurons,numIn);
    parameters.(name + "_Bias") = initializeZeros([numNeurons 1],'double');
end
sz = [1 numNeurons];
numIn = numNeurons;
parameters.("fc" + numLayers + "_Weights") = initializeGlorot(sz,1,numIn);
parameters.("fc" + numLayers + "_Bias") = initializeZeros([1 1],'double');

%% Train Network
% Convert the parameters to a vector
[parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters);
parametersV = double(extractdata(parametersV));
% Train Network
tstart = tic;
objFun = @(parameters) objFunc_Poisson_nPINN(parameters,dlX,dlX0BC,dlU0BC,dlXTest,UTest,parameterNames,parameterSizes,...
    k,scaleX,scaleU,alpha);
[parametersV,loss] = fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);
toc(tstart)
% Convert the vector of parameters to network structure
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
TestErr_LBFGS.("time"+i) = TestErr;
TestErr = [];

%% Evaluate Model Accuracy
dlXTest = dlXTest/scaleX;
dlUPred = model_Poisson_LBFGS(parameters,dlXTest);
dlUPred = alpha*scaleU*dlUPred;
err = norm(extractdata(dlUPred) - extractdata(UTest)) / norm(extractdata(UTest));
% Plot predictions vs true values.
figure
plot(XTest,extractdata(dlUPred),'-','LineWidth',2);
hold on
plot(XTest, extractdata(UTest), '--','LineWidth',2)
hold off
legend('Predicted','True')
end

%% Statistical results
TestErr_Final = [];
for i=1:numTest
    TestErr_Final = [TestErr_Final TestErr_LBFGS.("time"+i)(end)];
end
meanErr = mean(TestErr_Final);
stdErr = std(TestErr_Final);

%% Save results
save problemInfo.mat k L scaleX scaleU alpha numIntColPoints numLayers numNeurons
save resultsLBFGS_nPINN.mat parameters loss err TestErr_LBFGS TestErr_Final meanErr stdErr
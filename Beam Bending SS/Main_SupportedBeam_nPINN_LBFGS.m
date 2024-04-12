%% Main function for implementing hnPINN to solve simply-supported beam problem by LBFGS optimizer
%% Programmer: Thang Le-Duc
%  Emails: le.duc.thang0312@gmail.com

%% Begin main function
clear all, close all, clc
rng('default')
addpath('./utils')
global TestErr

%% Initially physical model
E = 200e9;
L = 10; H = L/100;
b = 1; I = b*H^3/12;
EI = E*I;
q0 = 1000;

%% Normalized physical model
scaleX = L;
scaleU = q0*L^4/EI;
% alpha = 1;
alpha = 0.02;

%% Generate Training Data
% Select points to enforce boundary conditions
x0BC1_Init = [0 L];
u0BC1_Init = [0 0];
x0BC2_Init = [0 L];
u0BC2_Init = [0 0];
% Normalize boundary conditions
X0BC1 = x0BC1_Init/scaleX;
U0BC1 = u0BC1_Init;
X0BC2 = x0BC2_Init/scaleX;
U0BC2 = u0BC2_Init;
% Select points to enforce the network output
numIntColPoints = 1000;
dataX = L*rand(numIntColPoints,1);
dataX = dataX/scaleX;
ds = arrayDatastore(dataX);

%% Specify Training Options
MaxFuncEval = 600;
% Optimize using LBFGS algorithm
options = optimoptions('fmincon', ...
    'HessianApproximation','lbfgs', ...
    'MaxIterations',2*MaxFuncEval, ...
    'MaxFunctionEvaluations',MaxFuncEval, ...
    'OptimalityTolerance',1e-6, ...
    'SpecifyObjectiveGradient',true);
options.Display = 'iter';
% Calculate true values.
numPredictions = 1001;
XTest = linspace(0,L,numPredictions);
dlXTest = dlarray(XTest,'CB');
UTest = q0*L^4/(24*EI)*(dlXTest/L - 2*(dlXTest/L).^3 + (dlXTest/L).^4);

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
% Convert network parameters to a vector
[parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters);
parametersV = double(extractdata(parametersV));
% Convert boundary conditions to dlarray
dlX = dlarray(dataX','CB');
dlX0BC1 = dlarray(X0BC1,'CB');
dlU0BC1 = dlarray(U0BC1,'CB');
dlX0BC2 = dlarray(X0BC2,'CB');
dlU0BC2 = dlarray(U0BC2,'CB');
% Train Network
tstart = tic;
objFun = @(parameters) objFunc_SupportedBeam_nPINN(parameters,dlX,dlX0BC1,dlU0BC1,dlX0BC2,dlU0BC2,dlXTest,UTest,...
    parameterNames,parameterSizes,EI,q0,L,scaleX,scaleU,alpha);
parametersV = fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);
toc(tstart)
% Convert the vector of parameters to network structure
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
TestErr_LBFGS.("time"+i) = TestErr;
TestErr = [];

%% Evaluate Model Accuracy
dlXTestInp = dlXTest/scaleX;
dlUPred = model_SupportedBeam_LBFGS(parameters,dlXTestInp);
dlUPred = alpha*scaleU*dlUPred;
err = norm(extractdata(dlUPred) - extractdata(UTest)) / norm(extractdata(UTest));
figure
% Plot predictions vs true values.
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
save problemInfo.mat E L EI q0 numIntColPoints numLayers numNeurons MaxFuncEval
save resultsLBFGS_nPINN.mat parameters TestErr TestErr_LBFGS TestErr_Final meanErr stdErr
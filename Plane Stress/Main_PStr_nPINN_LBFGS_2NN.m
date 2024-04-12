%% Main function for implementing hnPINN to solve inplane-plate deformation problem by LBFGS optimizer
%% Programmer: Thang Le-Duc
%  Emails: le.duc.thang0312@gmail.com

%% Begin main function
clear all, close all, clc
rng('default')
addpath('./utils')

%% Initially physical model
a = 10;
b = 10;
h = 1;
E = 70;
muy = 0.3;

%% Normalized physical model
scaleX = a;
scaleY = b;
scaleU = (1-muy^2)*scaleX^2/E;
scaleV = (1-muy^2)*scaleY^2/E;
% alpha_u = 1; alpha_v = 1;
alpha_u = 0.01; alpha_v = 0.01;

%% Generate Training Data
numBCPoints = [50 50];
% Select points along x to enforce each of the boundary conditions
x0BCx1 = 0*ones(1,numBCPoints(1));
x0BCx2 = a*ones(1,numBCPoints(1));
x0BCx1 = x0BCx1/scaleX;
x0BCx2 = x0BCx2/scaleX;

y0BCx1 = b*linspace(0,1,numBCPoints(1));
y0BCx2 = b*linspace(0,1,numBCPoints(1));
y0BCx1 = y0BCx1/scaleY;
y0BCx2 = y0BCx2/scaleY;

u0BCx1 = zeros(1,numBCPoints(1));
u0BCx2 = h*cos(scaleY*y0BCx2/(2*b)*pi)/alpha_u;

% Select points along y to enforce each of the boundary conditions
x0BCy1 = a*linspace(0,1,numBCPoints(2));
x0BCy2 = a*linspace(0,1,numBCPoints(2));
x0BCy1 = x0BCy1/scaleX;
x0BCy2 = x0BCy2/scaleX;

y0BCy1 = 0*ones(1,numBCPoints(2));
y0BCy2 = b*ones(1,numBCPoints(2));
y0BCy1 = y0BCy1/scaleY;
y0BCy2 = y0BCy2/scaleY;

u0BCy1 = zeros(1,numBCPoints(2));
u0BCy2 = zeros(1,numBCPoints(2));

% Group together the data for boundary conditions
X0BCx1 = x0BCx1;
Y0BCx1 = y0BCx1;
U0BCx1 = u0BCx1;

X0BCy1 = x0BCy1;
Y0BCy1 = y0BCy1;
U0BCy1 = u0BCy1;

X0Nxx = x0BCx2;
Y0Nxx = y0BCx2;
U0Nxx = u0BCx2;

X0Nyy = x0BCy2;
Y0Nyy = y0BCy2;
U0Nyy = u0BCy2;

% Select points to enforce the network output
numIntColPoints = 1000;
pointsX = a*rand(numIntColPoints,1);
pointsY = b*rand(numIntColPoints,1);
dataX = pointsX;
dataY = pointsY;
dataX = dataX/scaleX;
dataY = dataY/scaleY;
ds = arrayDatastore([dataX dataY]);

% Convert boundary conditions to dlarray
dlX = dlarray(dataX','CB');
dlY = dlarray(dataY','CB');

dlX0BCx1 = dlarray(X0BCx1,'CB');
dlY0BCx1 = dlarray(Y0BCx1,'CB');
dlU0BCx1 = dlarray(U0BCx1,'CB');

dlX0BCy1 = dlarray(X0BCy1,'CB');
dlY0BCy1 = dlarray(Y0BCy1,'CB');
dlU0BCy1 = dlarray(U0BCy1,'CB');

dlX0Nxx = dlarray(X0Nxx,'CB');
dlY0Nxx = dlarray(Y0Nxx,'CB');
dlU0Nxx = dlarray(U0Nxx,'CB');

dlX0Nyy = dlarray(X0Nyy,'CB');
dlY0Nyy = dlarray(Y0Nyy,'CB');
dlU0Nyy = dlarray(U0Nyy,'CB');

% Calculate true values.
load FEM_Results.mat
numelements = 1000;
indices = randperm(length(nodes),numelements);
XTest = nodes(1,indices);
YTest = nodes(2,indices);
dlXTest = dlarray(XTest,'CB')/scaleX;
dlYTest = dlarray(YTest,'CB')/scaleY;
UTest = XDisp(indices)';
VTest = YDisp(indices)';
dlUTest = dlarray(UTest);
dlVTest = dlarray(VTest);
clear nodes XDisp YDisp

%% Specify Training Options
MaxFuncEval = 3000;
% Optimize using the fmincon optmizer with the LBFGS algorithm
options = optimoptions('fmincon', ...
    'HessianApproximation','lbfgs', ...
    'MaxIterations',2*MaxFuncEval, ...
    'MaxFunctionEvaluations',MaxFuncEval, ...
    'OutputFcn',@outfun,...
    'OptimalityTolerance',1e-8, ...
    'SpecifyObjectiveGradient',true);
options.Display = 'iter';

%% Train PINN model
numTest = 10;
TestErr_LBFGS = struct;
for i = 1:numTest
% Define PINN Model
numLayers = 3;
numNeurons = 40;
% Define network for predicting u
parametersU = struct;
sz = [numNeurons 2];
parametersU.fc1_Weights = initializeGlorot(sz,numNeurons,1);
parametersU.fc1_Bias = initializeZeros([numNeurons 1],'double');
for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;
    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parametersU.(name + "_Weights") = initializeGlorot(sz,numNeurons,numIn);
    parametersU.(name + "_Bias") = initializeZeros([numNeurons 1],'double');
end
sz = [1 numNeurons];
numIn = numNeurons;
parametersU.("fc" + numLayers + "_Weights") = initializeGlorot(sz,1,numIn);
parametersU.("fc" + numLayers + "_Bias") = initializeZeros([1 1],'double');

% Define network for predicting v
parametersV = struct;
sz = [numNeurons 2];
parametersV.fc1_Weights = initializeGlorot(sz,numNeurons,1);
parametersV.fc1_Bias = initializeZeros([numNeurons 1],'double');
for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;
    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parametersV.(name + "_Weights") = initializeGlorot(sz,numNeurons,numIn);
    parametersV.(name + "_Bias") = initializeZeros([numNeurons 1],'double');
end
sz = [1 numNeurons];
numIn = numNeurons;
parametersV.("fc" + numLayers + "_Weights") = initializeGlorot(sz,1,numIn);
parametersV.("fc" + numLayers + "_Bias") = initializeZeros([1 1],'double');

%% Train Network
global TestErrU TestErrV
% Convert network parameters to vectors
[parametersUvec,parameterNamesU,parameterSizesU] = parameterStructToVector(parametersU);
parametersUvec = extractdata(parametersUvec);
[parametersVvec,parameterNamesV,parameterSizesV] = parameterStructToVector(parametersV);
parametersVvec = extractdata(parametersVvec);
parametersUV = double([parametersUvec; parametersVvec]);
% Train Network
tstart = tic;
objFun = @(parameters) objFunc_PStr_2NN_nPINN(parameters,dlX,dlY,dlX0BCx1,dlY0BCx1,dlU0BCx1,dlX0BCy1,dlY0BCy1,dlU0BCy1,...
    dlX0Nxx,dlY0Nxx,dlU0Nxx,dlX0Nyy,dlY0Nyy,dlU0Nyy,dlXTest,dlYTest,UTest,VTest,parameterNamesU,parameterSizesU,...
    parameterNamesV,parameterSizesV,a,b,muy,E,scaleX,scaleY,scaleU,scaleV,alpha_u,alpha_v);
parametersUV = fmincon(objFun,parametersUV,[],[],[],[],[],[],[],options);
toc(tstart)
% Convert the vector of parameters to network structure
parametersUvec = parametersUV(1:floor(numel(parametersUV)/2));
parametersVvec = parametersUV(floor(numel(parametersUV)/2)+1:floor(numel(parametersUV)));
parametersU = parameterVectorToStruct(parametersUvec,parameterNamesU,parameterSizesU);
parametersV = parameterVectorToStruct(parametersVvec,parameterNamesV,parameterSizesV);

%% Evaluate Model Accuracy
% Make predictions.
numPredictions = 1001;
XCol = linspace(0,a,numPredictions);
YCol = linspace(0,b,numPredictions);
[XInp,YInp] = meshgrid(XCol,YCol);
XInpScaled = XInp/scaleX;
YInpScaled = YInp/scaleY;
for k=1:size(XInp,1)
    dlXInp = dlarray(XInpScaled(k,:),'CB');
    dlYInp = dlarray(YInpScaled(k,:),'CB');
    dlUPred(k,:) = scaleU*alpha_u*model_PStr_LBFGS_2NN_U(parametersU,dlXInp,dlYInp);
    dlVPred(k,:) = scaleV*alpha_v*model_PStr_LBFGS_2NN_V(parametersV,dlXInp,dlYInp);
end
UPredDat = extractdata(dlUPred);
VPredDat = extractdata(dlVPred);
DisPred = sqrt(UPredDat.^2 + VPredDat.^2);

figure()
contourf(XInp,YInp,UPredDat)
view(2)
colorbar
colormap(jet)
legend('u_x')

figure()
contourf(XInp,YInp,VPredDat)
view(2)
colorbar
colormap(jet)
legend('u_y')

TestErr_LBFGS_U.("time"+i) = TestErrU;
TestErr_LBFGS_V.("time"+i) = TestErrV;
TestErrU = [];
TestErrV = [];
end

%% Statistical results
TestErr_Final_U = [];
TestErr_Final_V = [];
for i=1:numTest
    TestErr_Final_U = [TestErr_Final_U TestErr_LBFGS_U.("time"+i)(end)];
    TestErr_Final_V = [TestErr_Final_V TestErr_LBFGS_V.("time"+i)(end)];
end
meanErr_U = mean(TestErr_Final_U);
stdErr_U = std(TestErr_Final_U);
meanErr_V = mean(TestErr_Final_V);
stdErr_V = std(TestErr_Final_V);

%% Save results
save problemInfo.mat a b h E muy alpha_u alpha_v numBCPoints numLayers numNeurons numPredictions numEpochs miniBatchSize lrSchedule initialLearnRate
save resultsLBFGS_nPINN.mat parametersU parametersV TestErr_LBFGS_U TestErr_LBFGS_V TestErr_Final_U TestErr_Final_V meanErr_U meanErr_V stdErr_U stdErr_V
%% Main function for implementing hnPINN to solve supported-plate problem by LBFGS optimizer
%% Programmer: Thang Le-Duc
%  Emails: le.duc.thang0312@gmail.com

%% Begin main function
clear all, close all, clc
rng('default')
addpath('./utils')
global TestErr

%% Initially physical model
a = 10; b = 5; h = 0.05;
q0 = 1000; m = 3; n = 3;
E = 2.1*10^10;
muy = 0.3;
D = E*h^3/(12*(1-muy^2));

%% Normalized physical model
scaleX = a;
scaleY = b;
scaleU = q0*scaleX^2*scaleY^2/D;
% alpha = 1;      % ndPINN
alpha = 0.001;    % hnPINN

%% Generate Training Data
% Select points along x to enforce each of the boundary conditions
numBCxPoints = [50 50];

x0BCx1 = 0*ones(1,numBCxPoints(1));
x0BCx2 = a*ones(1,numBCxPoints(2));
x0BCx1 = x0BCx1/scaleX;
x0BCx2 = x0BCx2/scaleX;

y0BCx1 = b*linspace(0,1,numBCxPoints(1));
y0BCx2 = b*linspace(0,1,numBCxPoints(2));
y0BCx1 = y0BCx1/scaleY;
y0BCx2 = y0BCx2/scaleY;

u0BCx1 = zeros(1,numBCxPoints(1));
u0BCx2 = zeros(1,numBCxPoints(2));
u0BCx1 = u0BCx1/(scaleU*alpha);
u0BCx2 = u0BCx2/(scaleU*alpha);
u0BCx3 = zeros(1,numBCxPoints(1));
u0BCx4 = zeros(1,numBCxPoints(2));
u0BCx3 = u0BCx3/(scaleU*alpha);
u0BCx4 = u0BCx4/(scaleU*alpha);

% Select points along y to enforce each of the boundary conditions
numBCyPoints  = [50 50];

x0BCy1 = a*linspace(0,1,numBCyPoints(1));
x0BCy2 = a*linspace(0,1,numBCyPoints(2));
x0BCy1 = x0BCy1/scaleX;
x0BCy2 = x0BCy2/scaleX;

y0BCy1 = 0*ones(1,numBCyPoints(1));
y0BCy2 = b*ones(1,numBCyPoints(2));
y0BCy1 = y0BCy1/scaleY;
y0BCy2 = y0BCy2/scaleY;

u0BCy1 = zeros(1,numBCyPoints(1));
u0BCy2 = zeros(1,numBCyPoints(2));
u0BCy1 = u0BCy1/(scaleU*alpha);
u0BCy2 = u0BCy2/(scaleU*alpha);
u0BCy3 = zeros(1,numBCxPoints(1));
u0BCy4 = zeros(1,numBCxPoints(2));
u0BCy3 = u0BCy3/(scaleU*alpha);
u0BCy4 = u0BCy4/(scaleU*alpha);

% Group together the data for boundary conditions
X0BCx = [x0BCx1 x0BCx2];
Y0BCx = [y0BCx1 y0BCx2];
U0BCx = [u0BCx1 u0BCx2];
U0BCx2 = [u0BCx3 u0BCx4];

X0BCy = [x0BCy1 x0BCy2];
Y0BCy = [y0BCy1 y0BCy2];
U0BCy = [u0BCy1 u0BCy2];
U0BCy2 = [u0BCy3 u0BCy4];

% Select points to enforce the network output
numIntColPoints = 1000;
pointsX = a*rand(numIntColPoints,1);
pointsY = b*rand(numIntColPoints,1);
dataX = pointsX;
dataX = dataX/scaleX;
dataY = pointsY;
dataY = dataY/scaleY;
ds = arrayDatastore([dataX dataY]);

% Convert boundary conditions to dlarray
dlX = dlarray(dataX','CB');
dlY = dlarray(dataY','CB');

dlX0BCx = dlarray(X0BCx,'CB');
dlY0BCx = dlarray(Y0BCx,'CB');
dlU0BCx = dlarray(U0BCx,'CB');
dlU0BCx2 = dlarray(U0BCx2,'CB');

dlX0BCy = dlarray(X0BCy,'CB');
dlY0BCy = dlarray(Y0BCy,'CB');
dlU0BCy = dlarray(U0BCy,'CB');
dlU0BCy2 = dlarray(U0BCy2,'CB');

%% Specify Training Options
MaxFuncEval = 10000;
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
XTest = linspace(0,a,numPredictions);
YTest = linspace(0,b,numPredictions);
dlXTest = dlarray(XTest,'CB');
dlYTest = dlarray(YTest,'CB');
UTest = q0/(pi^4*D)*(((m/a)^2 + (n/b)^2)^(-2))*(sin(m*pi*dlXTest/a).*sin(n*pi*dlYTest/b));

%% Train PINN model
numTest = 10;
TestErr_LBFGS = struct;
for i = 1:numTest
% Define PINN Model
numLayers = 3;
numNeurons = 40;
parameters = struct;
sz = [numNeurons 2];
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
% Convert network parameters to vectors
[parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters);
parametersV = double(extractdata(parametersV));
% Train Network
tstart = tic;
objFun = @(parameters) objFunc_CPB_nPINN(parameters,dlX,dlY,dlX0BCx,dlY0BCx,dlU0BCx,dlU0BCx2,dlX0BCy,dlY0BCy,dlU0BCy,dlU0BCy2,...
    dlXTest,dlYTest,UTest,parameterNames,parameterSizes,a,b,muy,q0,m,n,D,scaleX,scaleY,scaleU,alpha);
[parametersV,loss] = fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);
toc(tstart)
% Convert the vector of parameters to network structure
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
TestErr_LBFGS.("time"+i) = TestErr;
TestErr = [];

%% Evaluate Model Accuracy
% Make predictions.
[XInp,YInp] = meshgrid(XTest,YTest);
XInpScaled = XInp/scaleX;
YInpScaled = YInp/scaleY;
for j=1:size(XInp,1)
    dlXInp = dlarray(XInpScaled(j,:),'CB');
    dlYInp = dlarray(YInpScaled(j,:),'CB');
    dlUPred = model_CPB_LBFGS(parameters,dlXInp,dlYInp);
    ZPred(j,:) = alpha*scaleU*dlUPred;
end
ZPredDat = extractdata(ZPred);

% Plot predictions.
figure()
contourf(XInp,YInp,ZPredDat)
view(2)
colorbar
colormap(jet)
if m==1 && n==1
    clim([0 0.018]);
    legend('Prediction, m = 1, n = 1')
elseif m==2 && n==2
    clim([-1.2e-3 1.2e-3]);
    legend('Prediction, m = 2, n = 2')
elseif m==3 && n==3
    clim([-2.1e-4 2.1e-4]);
    legend('Prediction, m = 3, n = 3')
end

[X,Y] = meshgrid(XTest,YTest);
Z = q0/(pi^4*D)*(((m/a)^2 + (n/b)^2)^(-2))*(sin(m*pi*X/a).*sin(n*pi*Y/b));
figure()
err = abs(ZPredDat - Z);
contourf(X,Y,err)
view(2)
colorbar
colormap(jet)
if m==1 && n==1
    clim([0 0.018]);
    legend('Point-wise AbsErr, m = 1, n = 1')
elseif m==2 && n==2
    clim([-1.2e-3 1.2e-3]);
    legend('Point-wise AbsErr, m = 2, n = 2')
elseif m==3 && n==3
    clim([-2.1e-4 2.1e-4]);
    legend('Point-wise AbsErr, m = 3, n = 3')
end
end

% Plot true values.
figure()
contourf(X,Y,Z)
view(2)
colorbar
colormap(jet)
if m==1 && n==1
    clim([0 0.018]);
    legend('True, m = 1, n = 1')
elseif m==2 && n==2
    clim([-1.2e-3 1.2e-3]);
    legend('True, m = 2, n = 2')
elseif m==3 && n==3
    clim([-2.1e-4 2.1e-4]);
    legend('True, m = 3, n = 3')
end

%% Statistical results
TestErr_Final = [];
for i=1:numTest
    TestErr_Final = [TestErr_Final TestErr_LBFGS.("time"+i)(end)];
end
meanErr = mean(TestErr_Final);
stdErr = std(TestErr_Final);

%% Save results
save problemInfo.mat a b h q0 m n E muy D scaleX scaleY scaleU alpha numBCxPoints numBCyPoints numLayers numNeurons numPredictions
save resultsLBFGS_nPINN.mat parameters loss TestErr_LBFGS TestErr_Final meanErr stdErr

%% Main function for implementing hnPINN to solve supported-plate problem by Adam optimizer
%% Programmer: Thang Le-Duc
%  Emails: le.duc.thang0312@gmail.com

%% Begin main function
clear all, close all, clc
rng('default')
addpath('./utils')

%% Initially physical model
a = 10; b = 5; h = 0.05;
q0 = 1000; m = 1; n = 1;
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
u0ICx1 = zeros(1,numBCxPoints(1));
u0ICx2 = zeros(1,numBCxPoints(2));
u0ICx1 = u0ICx1/(scaleU*alpha);
u0ICx2 = u0ICx2/(scaleU*alpha);

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
u0ICy1 = zeros(1,numBCxPoints(1));
u0ICy2 = zeros(1,numBCxPoints(2));
u0ICy1 = u0ICy1/(scaleU*alpha);
u0ICy2 = u0ICy2/(scaleU*alpha);

% Group together the data for boundary conditions
X0BCx = [x0BCx1 x0BCx2];
Y0BCx = [y0BCx1 y0BCx2];
U0BCx = [u0BCx1 u0BCx2];
U0ICx = [u0ICx1 u0ICx2];

X0BCy = [x0BCy1 x0BCy2];
Y0BCy = [y0BCy1 y0BCy2];
U0BCy = [u0BCy1 u0BCy2];
U0ICy = [u0ICy1 u0ICy2];

% Select points to enforce the network output
numIntColPoints = 1000;
pointsX = a*rand(numIntColPoints,1);
pointsY = b*rand(numIntColPoints,1);
dataX = pointsX;
dataX = dataX/scaleX;
dataY = pointsY;
dataY = dataY/scaleY;
ds = arrayDatastore([dataX dataY]);

%% Specify Training Options
numEpochs = 20000;
miniBatchSize = 1000;
executionEnvironment = "auto"; % "auto" "cpu" "gpu"
lrSchedule = 'time-based';    % 'none' 'piecewise' 'time-based' 'exponential' 'step'
[learningRate, initialLearnRate, decayRate, lrTepoch] = DNN_LearningRate(numEpochs, lrSchedule);

%% Train Network
% Create a minibatchqueue object
mbq = minibatchqueue(ds, ...
    'MiniBatchSize',miniBatchSize, ...
    'MiniBatchFormat','BC', ...
    'OutputEnvironment',executionEnvironment);
% Convert boundary conditions to dlarray
dlX = dlarray(dataX','CB');
dlY = dlarray(dataY','CB');

dlX0BCx = dlarray(X0BCx,'CB');
dlY0BCx = dlarray(Y0BCx,'CB');
dlU0BCx = dlarray(U0BCx,'CB');
dlU0ICx = dlarray(U0ICx,'CB');

dlX0BCy = dlarray(X0BCy,'CB');
dlY0BCy = dlarray(Y0BCy,'CB');
dlU0BCy = dlarray(U0BCy,'CB');
dlU0ICy = dlarray(U0ICy,'CB');

% Convert boundary conditions to gpuArray for training using a GPU
if (executionEnvironment == "auto" && canUseGPU) || (executionEnvironment == "gpu")
    dlX0BCx = gpuArray(dlX0BCx);
    dlY0BCx = gpuArray(dlY0BCx);
    dlU0BCx = gpuArray(dlU0BCx);
    dlU0ICx = gpuArray(dlU0ICx);

    dlX0BCy = gpuArray(dlX0BCy);
    dlY0BCy = gpuArray(dlY0BCy);
    dlU0BCy = gpuArray(dlU0BCy);
    dlU0ICy = gpuArray(dlU0ICy);
end

% Initialize the parameters for the Adam solver
averageGrad = []; averageSqGrad = [];
% Define PINN gradients function
accfun = dlaccelerate(@modelGradients_CPB_nPINN_Adam);
% accfun = @modelGradients_CPB_nPINN_Adam;

% Initialize the training progress plot
figure
C = colororder;
lineLoss = animatedline('Color',C(2,:));
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

% Calculate true values.
numPredictions = 1001;
XTest = linspace(0,a,numPredictions);
YTest = linspace(0,b,numPredictions);
UTest = q0/(pi^4*D)*(((m/a)^2 + (n/b)^2)^(-2))*(sin(m*pi*XTest/a).*sin(n*pi*YTest/b));

%% Train PINN model
iteration = 0;
numTest = 10;
TestErr_Adam = zeros(numEpochs,numTest);
loss = zeros(numEpochs,numTest);
lossF = zeros(numEpochs,numTest);
lossUBCx1 = zeros(numEpochs,numTest);
lossUBCx2 = zeros(numEpochs,numTest);
lossUBCy1 = zeros(numEpochs,numTest);
lossUBCy2 = zeros(numEpochs,numTest);
for i = 1:numTest
% Define PINN Model
numLayers = 3;
numNeurons = 40;
parameters = struct;
sz = [numNeurons 2];
parameters.fc1.Weights = initializeGlorot(sz,numNeurons,1);
parameters.fc1.Bias = initializeZeros([numNeurons 1]);
for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;
    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parameters.(name).Weights = initializeGlorot(sz,numNeurons,numIn);
    parameters.(name).Bias = initializeZeros([numNeurons 1]);
end
sz = [1 numNeurons];
numIn = numNeurons;
parameters.("fc" + numLayers).Weights = initializeGlorot(sz,1,numIn);
parameters.("fc" + numLayers).Bias = initializeZeros([1 1]);
% Train the network
start = tic;
for epoch = 1:numEpochs
    reset(mbq);
    while hasdata(mbq)
        iteration = iteration + 1;
        dlXY = next(mbq);
        dlX = dlXY(1,:);
        dlY = dlXY(2,:);
        [gradients,loss(iteration,i),lossF(iteration,i),lossUBCx1(iteration,i),lossUBCx2(iteration,i),lossUBCy1(iteration,i),lossUBCy2(iteration,i)] = ...
            dlfeval(accfun,parameters,dlX,dlY,dlX0BCx,dlY0BCx,dlU0BCx,dlU0ICx,dlX0BCy,dlY0BCy,dlU0BCy,dlU0ICy,...
            a,b,muy,q0,m,n,D,scaleX,scaleY,scaleU,alpha);
        learningRate = LRSchedule(learningRate, initialLearnRate, decayRate, lrTepoch, lrSchedule, epoch, iteration);
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,learningRate);
    end
    % Plot training progress.
    lossiter = loss(iteration,i);
    addpoints(lineLoss,iteration, lossiter);
    Dtime = duration(0,0,toc(start),'Format','hh:mm:ss');
    title("Epoch: " + epoch + ", Elapsed: " + string(Dtime) + ", Loss: " + lossiter)
    drawnow
    % Calculate error.
    dlXTest = dlarray(XTest,'CB')/scaleX;
    dlYTest = dlarray(YTest,'CB')/scaleY;
    dlUPred = model_CPB(parameters,dlXTest,dlYTest);
    dlUPred = alpha*scaleU*dlUPred;
    err = norm(extractdata(dlUPred) - UTest) / norm(UTest)
    TestErr_Adam(iteration,i) = err;
end
iteration = 0;

%% Evaluate Model Accuracy
% Make predictions.
[XInp,YInp] = meshgrid(XTest,YTest);
XInpScaled = XInp/scaleX;
YInpScaled = YInp/scaleY;
for i=1:size(XInp,1)
    dlXInp = dlarray(XInpScaled(i,:),'CB');
    dlYInp = dlarray(YInpScaled(i,:),'CB');
    dlUPred = model_CPB(parameters,dlXInp,dlYInp);
    ZPred(i,:) = alpha*scaleU*dlUPred;
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

% Plot exact solutions.
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
meanErr = mean(TestErr_Adam(numEpochs,:));
stdErr = std(TestErr_Adam(numEpochs,:));

%% Save results
save problemInfo.mat a b h q0 m n E muy D scaleX scaleY scaleU alpha numBCxPoints numBCyPoints numLayers numNeurons numPredictions lrSchedule initialLearnRate
save resultsAdam_nPINN.mat parameters loss lossF lossUBCx1 lossUBCx2 lossUBCy1 lossUBCy2 TestErr_Adam Dtime meanErr stdErr

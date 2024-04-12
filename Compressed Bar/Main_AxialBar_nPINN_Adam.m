%% Main function for implementing hnPINN to solve compressed-bar problem by Adam optimizer
%% Programmer: Thang Le-Duc
%  Emails: le.duc.thang0312@gmail.com

%% Begin main function
clear all, close all, clc
rng('default')
addpath('./utils')

%% Initially physical model
E = 1e9; A = 1e-4;
L = 1;
g = 5;
P = 50;

%% Normalized physical model
scaleX = L;
scaleU = scaleX^2/(E*A);
% alpha = 1;
alpha = max(g*A, abs(-P*scaleX/(E*A*scaleU)));

%% Generate Training Data
% Select points to enforce boundary conditions
x0BC1_Init = L;
u0BC1_Init = 0;
x0BC2_Init = 0;
u0BC2_Init = P/(alpha*E*A*scaleU/scaleX);
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
numEpochs = 800;
miniBatchSize = numIntColPoints;
executionEnvironment = "auto"; % "auto" "cpu" "gpu"
learningRate = 1e-3;

%% Training Setup
% Create a minibatchqueue object
mbq = minibatchqueue(ds, ...
    'MiniBatchSize',miniBatchSize, ...
    'MiniBatchFormat','BC', ...
    'OutputEnvironment',executionEnvironment);
% Convert the initial and boundary conditions to dlarray
dlX0BC1 = dlarray(X0BC1,'CB');
dlU0BC1 = dlarray(U0BC1);
dlX0BC2 = dlarray(X0BC2,'CB');
dlU0BC2 = dlarray(U0BC2);
% Convert boundary conditions to gpuArray for training using a GPU
if (executionEnvironment == "auto" && canUseGPU) || (executionEnvironment == "gpu")
    dlX0BC1 = gpuArray(dlX0BC1);
    dlU0BC1 = gpuArray(dlU0BC1);
    dlX0BC2 = gpuArray(dlX0BC2);
    dlU0BC2 = gpuArray(dlU0BC2);
end
% Initialize the parameters for the Adam solver
averageGrad = []; averageSqGrad = [];
% Define PINN gradients function
accfun = dlaccelerate(@modelGradients_AxialBar_nPINN_Adam);
% accfun = @modelGradients_AxialBar_nPINN_Adam;

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
XTest = linspace(0,L,numPredictions);
UTest = 1/E*(g/2*(L^2 - XTest.^2) + P/A*(L - XTest));
dlUTest = dlarray(UTest);

%% Train PINN model
start = tic;
iteration = 0;
numTest = 10;
TestErr_Adam = zeros(numEpochs,numTest);
loss = zeros(numEpochs,numTest);
lossF = zeros(numEpochs,numTest);
lossUBC1 = zeros(numEpochs,numTest);
lossUBC2 = zeros(numEpochs,numTest);
for i = 1:numTest
% Define PINN Model
numLayers = 3;
numNeurons = 40;
parameters = struct;
sz = [numNeurons 1];
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
for epoch = 1:numEpochs
    reset(mbq);
    while hasdata(mbq)
        iteration = iteration + 1;
        dlX = next(mbq);
        [gradients,loss(iteration,i),lossF(iteration,i),lossUBC1(iteration,i),lossUBC2(iteration,i)] = ...
            dlfeval(accfun,parameters,dlX,dlX0BC1,dlU0BC1,dlX0BC2,dlU0BC2,E,A,L,g,scaleX,scaleU,alpha);
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,learningRate);
    end
    % Plot training progress.
    lossiter = loss(iteration,i);
    addpoints(lineLoss,iteration, lossiter);
    D = duration(0,0,toc(start),'Format','hh:mm:ss');
    title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + lossiter)
    drawnow
    % Calculate error.
    dlXTest = dlarray(XTest,'CB')/scaleX;
    dlUPred = model_AxialBar(parameters,dlXTest);
    dlUPred = alpha*scaleU*dlUPred;
    err = norm(extractdata(dlUPred) - UTest) / norm(UTest)
    TestErr_Adam(iteration,i) = err;
end
iteration = 0;

%% Evaluate Model Accuracy
dlXTest = dlarray(XTest,'CB')/scaleX;
dlUPred = model_AxialBar(parameters,dlXTest);
dlUPred = alpha*scaleU*dlUPred;
err = norm(extractdata(dlUPred) - UTest) / norm(UTest);
% Plot predictions vs true values.
figure
plot(XTest,extractdata(dlUPred),'-','LineWidth',2);
hold on
plot(XTest, UTest, '--','LineWidth',2)
hold off
legend('Predicted','True')
end

%% Statistical results
meanErr = mean(TestErr_Adam(end,:));
stdErr = std(TestErr_Adam(end,:));

%% Save results
save problemInfo.mat E A L g P numLayers numNeurons numIntColPoints numEpochs
save resultsAdam_nPINN.mat parameters loss lossF lossUBC1 lossUBC2 err TestErr_Adam D meanErr stdErr
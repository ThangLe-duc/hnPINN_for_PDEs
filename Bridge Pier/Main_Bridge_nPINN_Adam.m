%% Main function for implementing hnPINN to solve bridge-pier problem by Adam optimizer
%% Programmer: Thang Le-Duc
%  Emails: le.duc.thang0312@gmail.com

%% Begin main function
clear all, close all, clc
rng('default')
addpath('./utils')

%% Initially physical model
E = 28e6;
L = 2;
q0 = 6.25;
q0IC = -5;

%% Normalized physical model
scaleX = L;
scaleU = 4*q0*scaleX^2/E;
% alpha = 1;
alpha = 10;

%% Generate Training Data
% Select points to enforce boundary conditions
x0BC1_Init = L;
u0BC1_Init = 0;
x0BC2_Init = 0;
u0BC2_Init = q0IC/(alpha*scaleX*q0);
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
numEpochs = 1000;
miniBatchSize = numIntColPoints;
executionEnvironment = "auto";
lrSchedule = 'time-based';    % 'none' 'piecewise' 'time-based' 'exponential' 'step'
[learningRate, initialLearnRate, decayRate, lrTepoch] = DNN_LearningRate(numEpochs, lrSchedule);

%% Train Network
% Create a minibatchqueue object
mbq = minibatchqueue(ds, ...
    'MiniBatchSize',miniBatchSize, ...
    'MiniBatchFormat','BC', ...
    'OutputEnvironment',executionEnvironment);
% Convert boundary conditions to dlarray
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
accfun = dlaccelerate(@modelGradients_Bridge_nPINN_Adam);
% accfun = @modelGradients_Bridge_nPINN_Adam;

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
UTest = 1/E*(56.25 - 6.25*(1+XTest).^2 -7.5*log((1+XTest)/3));

%% Train PINN model
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
start = tic;
for epoch = 1:numEpochs
    reset(mbq);
    while hasdata(mbq)
        iteration = iteration + 1;
        dlX = next(mbq);
        [gradients,loss(iteration,i),lossF(iteration,i),lossUBC1(iteration,i),lossUBC2(iteration,i)] = ...
            dlfeval(accfun,parameters,dlX,dlX0BC1,dlU0BC1,dlX0BC2,dlU0BC2,E,L,q0,scaleX,scaleU,alpha);
        learningRate = LRSchedule(learningRate, initialLearnRate, decayRate, lrTepoch, lrSchedule, epoch, iteration);
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
    dlUPred = model_Bridge(parameters,dlXTest);
    dlUPred = alpha*scaleU*dlUPred;
    err = norm(extractdata(dlUPred) - UTest) / norm(UTest)
    TestErr_Adam(iteration,i) = err;
end
iteration = 0;

%% Evaluate Model Accuracy
dlXTest = dlarray(XTest,'CB')/scaleX;
dlUPred = model_Bridge(parameters,dlXTest);
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
meanErr = mean(TestErr_Adam(numEpochs,:));
stdErr = std(TestErr_Adam(numEpochs,:));

%% Save results
save problemInfo.mat E L q0 q0IC numIntColPoints numLayers numNeurons numPredictions numEpochs miniBatchSize lrSchedule initialLearnRate
save resultsAdam.mat parameters loss lossF lossUBC1 lossUBC2 TestErr_Adam D meanErr stdErr
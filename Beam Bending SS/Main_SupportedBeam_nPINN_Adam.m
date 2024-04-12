%% Main function for implementing hnPINN to solve simply-supported beam problem by Adam optimizer
%% Programmer: Thang Le-Duc
%  Emails: le.duc.thang0312@gmail.com

%% Begin main function
clear all, close all, clc
rng('default')
addpath('./utils')

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
numEpochs = 1500;
miniBatchSize = 1000;
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
accfun = dlaccelerate(@modelGradients_SupportedBeam_nPINN_Adam);
% accfun = @modelGradients_SupportedBeam_nPINN_Adam;

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
UTest = q0*L^4/(24*EI)*(XTest/L - 2*(XTest/L).^3 + (XTest/L).^4);

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
            dlfeval(accfun,parameters,dlX,dlX0BC1,dlU0BC1,dlX0BC2,dlU0BC2,EI,q0,L,scaleX,scaleU,alpha);
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
    dlUPred = model_SupportedBeam(parameters,dlXTest);
    dlUPred = alpha*scaleU*dlUPred;
    err = norm(extractdata(dlUPred) - UTest) / norm(UTest)
    TestErr_Adam(iteration,i) = err;
end
iteration = 0;

%% Evaluate Model Accuracy
dlXTest = dlarray(XTest,'CB')/scaleX;
dlUPred = model_SupportedBeam(parameters,dlXTest);
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
save problemInfo.mat E L EI q0 numIntColPoints numLayers numNeurons numEpochs miniBatchSize lrSchedule initialLearnRate
save resultsAdam.mat parameters loss lossF lossUBC1 lossUBC2 TestErr_Adam D meanErr stdErr
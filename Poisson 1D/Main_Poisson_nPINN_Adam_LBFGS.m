%% Main function for implementing hnPINN to solve 1D Poisson problem by Adam-LBFGS optimizer
%% Programmer: Thang Le-Duc
%  Emails: le.duc.thang0312@gmail.com

%% Begin main function
clear all, close all, clc
rng('default')
addpath('./utils')

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
x0BC1 = [0 L];
u0BC1 = [0 0];
% Normalize boundary conditions
X0BC1 = x0BC1/scaleX;
U0BC1 = u0BC1;
% Select points to enforce the network output
numIntColPoints = 1000;
dataX = L*rand(numIntColPoints,1);
dataX = dataX/scaleX;
ds = arrayDatastore(dataX);

%% Specify Adam Training Options
numEpochs = 2240;
miniBatchSize = 1000;
executionEnvironment = "cpu"; % "auto" "cpu" "gpu"
lrSchedule = 'time-based';    % 'none' 'piecewise' 'time-based' 'exponential' 'step'
[learningRate, initialLearnRate, decayRate, lrTepoch] = DNN_LearningRate(numEpochs, lrSchedule);

%% Specify LBFGS Training Options
MaxFuncEval = 560;
% Optimize using the fmincon optmizer with the LBFGS algorithm
options = optimoptions('fmincon', ...
    'HessianApproximation','lbfgs', ...
    'MaxIterations',2*MaxFuncEval, ...
    'MaxFunctionEvaluations',MaxFuncEval, ...
    'OptimalityTolerance',1e-16, ...
    'SpecifyObjectiveGradient',true);
options.Display = 'iter';

%% Train Network
% Create a minibatchqueue object
mbq = minibatchqueue(ds, ...
    'MiniBatchSize',miniBatchSize, ...
    'MiniBatchFormat','BC', ...
    'OutputEnvironment',executionEnvironment);
% Convert boundary conditions to dlarray
dlX0BC1 = dlarray(X0BC1,'CB');
dlU0BC1 = dlarray(U0BC1);
% Convert boundary conditions to gpuArray for training using a GPU
if (executionEnvironment == "auto" && canUseGPU) || (executionEnvironment == "gpu")
    dlX0BC1 = gpuArray(dlX0BC1);
    dlU0BC1 = gpuArray(dlU0BC1);
end
% Initialize the parameters for the Adam solver
averageGrad = []; averageSqGrad = [];
% Define PINN gradients function
accfun = dlaccelerate(@modelGradients_Poisson_nPINN_Adam);
% accfun = @modelGradients_Poisson_nPINN_Adam;

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
UTest = sin(k*pi*XTest);
dlUTest = dlarray(UTest);

%% Train PINN model
iteration = 0;
numTest = 10;
TestErr_Adam = zeros(numEpochs,numTest);
TestErr_LBFGS = struct;
loss = zeros(numEpochs,numTest);
lossF = zeros(numEpochs,numTest);
lossUBC = zeros(numEpochs,numTest);
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
        [gradients,loss(iteration,i),lossF(iteration,i),lossUBC(iteration,i)] = dlfeval(accfun,parameters,dlX,...
            dlX0BC1,dlU0BC1,k,scaleX,scaleU,alpha);
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
    dlXTest = dlarray(XTest,'CB');
    dlUPred = model_Poisson(parameters,dlXTest);
    dlUPred = alpha*scaleU*dlUPred;
    err = norm(extractdata(dlUPred) - UTest) / norm(UTest)
    TestErr_Adam(iteration,i) = err;
end
iteration = 0;

%% L-BFGS Algorithm
global TestErr
dlXTest = dlarray(XTest,'CB');
% Convert network struct to a vector for LBFGS algorithm
parametersLBFGS = parameterStructToStructLBFGS(parameters);
[parametersVLBFGS,parameterNames,parameterSizes] = parameterStructToVector(parametersLBFGS);
parametersVLBFGS = double(extractdata(parametersVLBFGS));
% Train Network
tstart = tic;
objFun = @(parameters) objFunc_Poisson_nPINN(parameters,dlX,dlX0BC1,dlU0BC1,dlXTest,dlUTest,...
    parameterNames,parameterSizes,k,scaleX,scaleU,alpha);
parametersVLBFGS = fmincon(objFun,parametersVLBFGS,[],[],[],[],[],[],[],options);
toc(tstart)
% Convert the vector of parameters to network structure
parametersLBFGS = parameterVectorToStruct(parametersVLBFGS,parameterNames,parameterSizes);
TestErr_LBFGS.("time"+i) = TestErr;
TestErr = [];

%% Evaluate Model Accuracy
dlXTest = dlXTest/scaleX;
dlUPred = model_Poisson_LBFGS(parametersLBFGS,dlXTest);
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
TestErr_Conv = struct;
for i=1:numTest
    TestErr_Conv.("time"+i) = [TestErr_Adam(:,i); TestErr_LBFGS.("time"+i)];
end
TestErr_Final = [];
for i=1:numTest
    TestErr_Final = [TestErr_Final TestErr_LBFGS.("time"+i)(end)];
end
meanErr = mean(TestErr_Final);
stdErr = std(TestErr_Final);

%% Save results
save problemInfo.mat k L scaleX scaleU alpha numIntColPoints numLayers numNeurons numEpochs miniBatchSize lrSchedule initialLearnRate
save resultsAdamLBFGS_nPINN.mat parameters loss lossF lossUBC TestErr_Adam TestErr_LBFGS TestErr_Conv TestErr_Final D meanErr stdErr
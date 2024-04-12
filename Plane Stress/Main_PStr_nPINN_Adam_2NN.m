%% Main function for implementing hnPINN to solve inplane-plate deformation problem by Adam optimizer
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
% alpha_u = 1; alpha_v = 1;        % ndPINN
alpha_u = 0.01; alpha_v = 0.01;    % hnPINN

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

%% Specify Training Options
numEpochs = 10000;
miniBatchSize = numIntColPoints;
executionEnvironment = "cpu";
lrSchedule = 'time-based';    % 'none' 'piecewise' 'time-based' 'exponential' 'step'
[learningRate, initialLearnRate, decayRate, lrTepoch] = DNN_LearningRate(numEpochs, lrSchedule);
% Create a minibatchqueue object
mbq = minibatchqueue(ds, ...
    'MiniBatchSize',miniBatchSize, ...
    'MiniBatchFormat','BC', ...
    'OutputEnvironment',executionEnvironment);
% Convert boundary conditions to dlarray
dlX = dlarray(dataX','CB');
dlY = dlarray(dataY','CB');

dlX0BCx1 = dlarray(X0BCx1,'CB');
dlY0BCx1 = dlarray(Y0BCx1,'CB');
dlU0BCx1 = dlarray(U0BCx1);

dlX0BCy1 = dlarray(X0BCy1,'CB');
dlY0BCy1 = dlarray(Y0BCy1,'CB');
dlU0BCy1 = dlarray(U0BCy1);

dlX0Nxx = dlarray(X0Nxx,'CB');
dlY0Nxx = dlarray(Y0Nxx,'CB');
dlU0Nxx = dlarray(U0Nxx);

dlX0Nyy = dlarray(X0Nyy,'CB');
dlY0Nyy = dlarray(Y0Nyy,'CB');
dlU0Nyy = dlarray(U0Nyy);

% Convert boundary conditions to gpuArray for training using a GPU
if (executionEnvironment == "auto" && canUseGPU) || (executionEnvironment == "gpu")
    X0BCx1 = gpuArray(X0BCx1);
    Y0BCx1 = gpuArray(Y0BCx1);
    U0BCx1 = gpuArray(U0BCx1);
    X0BCy1 = gpuArray(X0BCy1);
    dlY0BCy1 = gpuArray(Y0BCy1);
    dlU0BCy1 = gpuArray(U0BCy1);

    dlX0Nxx = gpuArray(X0Nxx);
    dlY0Nxx = gpuArray(Y0Nxx);
    dlU0Nxx = gpuArray(U0Nxx);

    dlX0Nyy = gpuArray(X0Nyy);
    dlY0Nyy = gpuArray(Y0Nyy);
    dlU0Nyy = gpuArray(U0Nyy);
end

% Initialize the parameters for the Adam solver
averageGradU = []; averageSqGradU = [];
averageGradV = []; averageSqGradV = [];
% Define PINN gradients function
accfun = dlaccelerate(@modelGradients_PStr_nPINN_Adam_2NN_Type1);
% accfun = @modelGradients_PStr_nPINN_Adam_2NN;

% Initialize the training progress plot
figure
C = colororder;
lineLoss = animatedline('Color',C(2,:));
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on
% Calculate true values.
load FEM_Results.mat
numelements = 1000;
indices = randperm(length(nodes),numelements);
XTest = nodes(1,indices);
YTest = nodes(2,indices);
UTest = XDisp(indices)';
VTest = YDisp(indices)';
clear nodes XDisp YDisp

%% Train PINN model
iteration = 0;
numTest = 10;
TestErr_Adam_U = zeros(numEpochs,numTest);
TestErr_Adam_V = zeros(numEpochs,numTest);
loss = zeros(numEpochs,numTest);
lossF = zeros(numEpochs,numTest);
lossUBC1 = zeros(numEpochs,numTest);
lossUBC2 = zeros(numEpochs,numTest);
TestErr_LBFGS = struct;
for i = 1:numTest
% Define PINN Model
numLayers = 3;
numNeurons = 40;
% Define network for predicting u
parametersU = struct;
sz = [numNeurons 2];
parametersU.fc1.Weights = initializeGlorot(sz,numNeurons,1);
parametersU.fc1.Bias = initializeZeros([numNeurons 1]);
for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;
    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parametersU.(name).Weights = initializeGlorot(sz,numNeurons,numIn);
    parametersU.(name).Bias = initializeZeros([numNeurons 1]);
end
sz = [1 numNeurons];
numIn = numNeurons;
parametersU.("fc" + numLayers).Weights = initializeGlorot(sz,1,numIn);
parametersU.("fc" + numLayers).Bias = initializeZeros([1 1]);

% Define network for predicting v
parametersV = struct;
sz = [numNeurons 2];
parametersV.fc1.Weights = initializeGlorot(sz,numNeurons,1);
parametersV.fc1.Bias = initializeZeros([numNeurons 1]);
for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;
    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parametersV.(name).Weights = initializeGlorot(sz,numNeurons,numIn);
    parametersV.(name).Bias = initializeZeros([numNeurons 1]);
end
sz = [1 numNeurons];
numIn = numNeurons;
parametersV.("fc" + numLayers).Weights = initializeGlorot(sz,1,numIn);
parametersV.("fc" + numLayers).Bias = initializeZeros([1 1]);
% Train the network
start = tic;
for epoch = 1:numEpochs
    reset(mbq);
    while hasdata(mbq)
        iteration = iteration + 1;
        dlXY = next(mbq);
        dlX = dlXY(1,:);
        dlY = dlXY(2,:);
        [gradientsU,gradientsV,loss(iteration,1),lossF(iteration,1),lossUBC1(iteration,1),lossUBC2(iteration,1)] = ...
            dlfeval(accfun,parametersU,parametersV,dlX,dlY,dlX0BCx1,dlY0BCx1,dlU0BCx1,dlX0BCy1,dlY0BCy1,dlU0BCy1,dlX0Nxx,...
            dlY0Nxx,dlU0Nxx,dlX0Nyy,dlY0Nyy,dlU0Nyy,a,b,muy,E,scaleX,scaleY,scaleU,scaleV,alpha_u,alpha_v);
        learningRate = LRSchedule(learningRate, initialLearnRate, decayRate, lrTepoch, lrSchedule, epoch, iteration);
        [parametersU,averageGradU,averageSqGradU] = adamupdate(parametersU,gradientsU,averageGradU, ...
            averageSqGradU,iteration,learningRate);
        [parametersV,averageGradV,averageSqGradV] = adamupdate(parametersV,gradientsV,averageGradV, ...
            averageSqGradV,iteration,learningRate);
    end
    % Plot training progress.
    lossiter = loss(iteration,1);
    addpoints(lineLoss,iteration, lossiter);
    D = duration(0,0,toc(start),'Format','hh:mm:ss');
    title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + lossiter)
    drawnow
    % Calculate error.
    dlXTest = dlarray(XTest,'CB')/scaleX;
    dlYTest = dlarray(YTest,'CB')/scaleY;
    dlUPred = model_PStr_2NN_U(parametersU,dlXTest,dlYTest);
    dlVPred = model_PStr_2NN_V(parametersV,dlXTest,dlYTest);
    dlUPred = alpha_u*scaleU*dlUPred;
    dlVPred = alpha_v*scaleV*dlVPred;
    errU = norm(extractdata(dlUPred) - UTest) / norm(UTest);
    errV = norm(extractdata(dlVPred) - VTest) / norm(VTest);
    err = errU + errV
    TestErr_Adam_U(iteration,i) = errU;
    TestErr_Adam_V(iteration,i) = errV;
end
iteration = 0;

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
    dlUPredDat(k,:) = model_PStr_2NN_U(parametersU,dlXInp,dlYInp);
    dlVPredDat(k,:) = model_PStr_2NN_V(parametersV,dlXInp,dlYInp);
    dlUPredDat(k,:) = scaleU*alpha_u*dlUPredDat(k,:);
    dlVPredDat(k,:) = scaleV*alpha_v*dlVPredDat(k,:);
end
UPredDat = extractdata(dlUPredDat);
VPredDat = extractdata(dlVPredDat);
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
end

%% Statistical results
meanErr_U = mean(TestErr_Adam_U);
stdErr_U = std(TestErr_Adam_U);
meanErr_V = mean(TestErr_Adam_V);
stdErr_V = std(TestErr_Adam_V);

%% Save results
save problemInfo.mat a b h E muy numBCPoints numLayers numNeurons numPredictions numEpochs miniBatchSize lrSchedule initialLearnRate
save resultsAdam_nPINN.mat parametersU parametersV loss D TestErr_Adam_U TestErr_Adam_V meanErr_U meanErr_V stdErr_U stdErr_V

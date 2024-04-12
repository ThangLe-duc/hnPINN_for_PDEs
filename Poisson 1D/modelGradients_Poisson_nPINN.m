% Model Gradients Function
function [gradients,loss,lossF,lossUBC] = modelGradients_Poisson_nPINN(parameters,dlX,dlX0BC,dlU0BC,k,scaleX,scaleU,alpha)

% Make predictions with the initial conditions.
U = model_Poisson_LBFGS(parameters,dlX);

% Calculate derivatives with respect to X.
Ux = dlgradient(sum(U,'all'),dlX,'EnableHigherDerivatives',true);
Uxx = dlgradient(sum(Ux,'all'),dlX,'EnableHigherDerivatives',true);

% Calculate loss functions.
f = Uxx + scaleX^2/scaleU*k^2*pi^2/alpha*sin(k*pi*scaleX*dlX);
zeroTarget = zeros(size(f), 'like', f);
lossF = mse(f, zeroTarget);

dlU0BCPred = model_Poisson_LBFGS(parameters,dlX0BC);
lossUBC = mse(dlU0BCPred,dlU0BC);

% Combine losses.
loss = lossF + lossUBC;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);

end
% Model Gradients Function
function [gradients,loss,lossF,lossUBC] = modelGradients_Poisson_nPINN_Adam(parameters,dlX,dlX0BC1,dlU0BC1,...
    k,scaleX,scaleU,scaleBC)

% Make predictions.
U = model_Poisson(parameters,dlX);

% Calculate derivatives with respect to X.
Ux = dlgradient(sum(U,'all'),dlX,'EnableHigherDerivatives',true);
Uxx = dlgradient(sum(Ux,'all'),dlX,'EnableHigherDerivatives',true);

% Calculate loss functions.
f = Uxx + scaleX^2/scaleU*k^2*pi^2/scaleBC*sin(k*pi*scaleX*dlX);
zeroTarget = zeros(size(f), 'like', f);
lossF = mse(f, zeroTarget);

dlU0BCPred = model_Poisson(parameters,dlX0BC1);
lossUBC = mse(dlU0BCPred,dlU0BC1);

% Combine losses.
loss = lossF + lossUBC;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);

end
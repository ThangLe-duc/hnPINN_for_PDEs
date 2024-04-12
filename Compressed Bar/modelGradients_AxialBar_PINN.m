% Model Gradients Function
function [gradients,loss,lossF,lossUBC1,lossUBC2] = modelGradients_AxialBar_PINN(parameters,dlX,dlX0BC1,dlU0BC1,dlX0BC2,dlU0BC2,...
    E,A,L,g)

% Make predictions.
U = model_AxialBar_LBFGS(parameters,dlX);

% Calculate derivatives with respect to X.
gradientsU = dlgradient(sum(U,'all'),dlX,'EnableHigherDerivatives',true);
Ux = E*A*gradientsU;
Uxx = dlgradient(sum(Ux,'all'),dlX,'EnableHigherDerivatives',true);

% Calculate loss function.
f = Uxx + g*A;
zeroTarget = zeros(size(f), 'like', f);
lossF = mse(f, zeroTarget);

dlU0BC1Pred = model_AxialBar_LBFGS(parameters,dlX0BC1);
lossUBC1 = mse(dlU0BC1Pred,dlU0BC1);

dlU0BC2Pred = model_AxialBar_LBFGS(parameters,dlX0BC2);
Ux0BC2 = dlgradient(dlU0BC2Pred,dlX0BC2,'EnableHigherDerivatives',true);
Ux0BC2 = E*A*Ux0BC2;
lossUBC2 = mse(Ux0BC2,-dlU0BC2);

% Combine losses.
loss = lossF + lossUBC1 + lossUBC2;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);

end
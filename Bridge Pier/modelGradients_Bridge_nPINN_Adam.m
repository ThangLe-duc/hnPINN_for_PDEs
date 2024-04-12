% Model Gradients Function
function [gradients,loss,lossF,lossUBC1,lossUBC2] = modelGradients_Bridge_nPINN_Adam(parameters,dlX,dlX0BC1,dlU0BC1,dlX0BC2,dlU0BC2,...
    E,L,q0,scaleX,scaleU,alpha)

% Make predictions.
U = model_Bridge(parameters,dlX);
% Calculate derivatives with respect to X.
gradientsU = dlgradient(sum(U,'all'),dlX,'EnableHigherDerivatives',true);
Ux = (1+scaleX*dlX).*gradientsU;
Uxx = dlgradient(sum(Ux,'all'),dlX,'EnableHigherDerivatives',true);
% Calculate loss function.
f = Uxx + (1+scaleX*dlX)/alpha;
zeroTarget = zeros(size(f), 'like', f);
lossF = mse(f, zeroTarget);

dlU0BC1Pred = model_Bridge(parameters,dlX0BC1);
lossUBC1 = mse(dlU0BC1Pred,dlU0BC1);

dlU0BC2Pred = model_Bridge(parameters,dlX0BC2);
Ux0 = dlgradient(dlU0BC2Pred,dlX0BC2,'EnableHigherDerivatives',true);
Ux0 = (1+scaleX*dlX0BC2).*Ux0;
lossUBC2 = mse(Ux0,dlU0BC2);

% Combine losses.
loss = lossF + lossUBC1 + lossUBC2;

% Calculate gradients with respect to network parameters.
gradients = dlgradient(loss,parameters);

end
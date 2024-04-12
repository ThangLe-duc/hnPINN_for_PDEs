% Model Gradients Function
function [gradients,loss,lossF,lossUBC1,lossUBC2] = modelGradients_SupportedBeam_nPINN_Adam(parameters,dlX,dlX0BC1,dlU0BC1,...
    dlX0BC2,dlU0BC2,EI,q0,L,scaleX,scaleU,alpha)

% Make predictions.
U = model_SupportedBeam(parameters,dlX);
% Calculate derivatives with respect to X.
Ux = dlgradient(sum(U,'all'),dlX,'EnableHigherDerivatives',true);
Uxx = dlgradient(sum(Ux,'all'),dlX,'EnableHigherDerivatives',true);
Uxxx = dlgradient(sum(Uxx,'all'),dlX,'EnableHigherDerivatives',true);
Uxxxx = dlgradient(sum(Uxxx,'all'),dlX,'EnableHigherDerivatives',true);

% Calculate loss function.
f = alpha*Uxxxx - 1;
zeroTarget = zeros(size(f), 'like', f);
lossF = mse(f, zeroTarget);

dlU0BC1Pred = model_SupportedBeam(parameters,dlX0BC1);
lossUBC1 = mse(dlU0BC1Pred,dlU0BC1);

dlU0BC2Pred = model_SupportedBeam(parameters,dlX0BC2);
Ux0BC2 = dlgradient(sum(dlU0BC2Pred,'all'),dlX0BC2,'EnableHigherDerivatives',true);
Uxx0BC2 = dlgradient(sum(Ux0BC2,'all'),dlX0BC2,'EnableHigherDerivatives',true);
lossUBC2 = mse(Uxx0BC2,dlU0BC2);

% Combine losses.
loss = lossF + lossUBC1 + lossUBC2;

% Calculate gradients with respect to network parameters.
gradients = dlgradient(loss,parameters);

end
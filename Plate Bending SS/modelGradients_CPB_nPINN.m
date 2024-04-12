% Model Gradients Function
function [gradients,loss,lossF,lossUBCx,lossUBCx2,lossUBCy,lossUBCy2] = modelGradients_CPB_nPINN(parameters,dlX,dlY,dlX0BCx,...
    dlY0BCx,dlU0BCx,dlU0BCx2,dlX0BCy,dlY0BCy,dlU0BCy,dlU0BCy2,a,b,muy,q0,m,n,D,scaleX,scaleY,scaleU,alpha)

% Make predictions with the initial conditions.
U = model_CPB_LBFGS(parameters,dlX,dlY);

% Calculate derivatives with respect to X and Y.
gradientsU = dlgradient(sum(U,'all'),{dlX,dlY},'EnableHigherDerivatives',true);
Ux = gradientsU{1};
Uy = gradientsU{2};
% Calculate 2nd derivatives with respect to X and Y.
Uxx = dlgradient(sum(Ux,'all'),dlX,'EnableHigherDerivatives',true);
Uyy = dlgradient(sum(Uy,'all'),dlY,'EnableHigherDerivatives',true);
% Calculate 3rd derivatives with respect to X and Y.
Uxxx = dlgradient(sum(Uxx,'all'),dlX,'EnableHigherDerivatives',true);
Uxxy = dlgradient(sum(Uxx,'all'),dlY,'EnableHigherDerivatives',true);
Uyyy = dlgradient(sum(Uyy,'all'),dlY,'EnableHigherDerivatives',true);
% Calculate 4th derivatives with respect to X and Y.
Uxxxx = dlgradient(sum(Uxxx,'all'),dlX,'EnableHigherDerivatives',true);
Uxxyy = dlgradient(sum(Uxxy,'all'),dlY,'EnableHigherDerivatives',true);
Uyyyy = dlgradient(sum(Uyyy,'all'),dlY,'EnableHigherDerivatives',true);

% Calculate loss functions.
f = scaleY^2/scaleX^2*alpha*Uxxxx + 2*alpha*Uxxyy + scaleX^2/scaleY^2*alpha*Uyyyy - ...
    (scaleX^2*scaleY^2*q0/(D*scaleU))*sin(m*pi*dlX*scaleX/a).*sin(n*pi*dlY*scaleY/b);
zeroTarget = zeros(size(f), 'like', f);
lossF = mse(f, zeroTarget);

dlU0BCxPred = model_CPB_LBFGS(parameters,dlX0BCx,dlY0BCx);
lossUBCx = mse(dlU0BCxPred,dlU0BCx);

dlU0BCyPred = model_CPB_LBFGS(parameters,dlX0BCy,dlY0BCy);
lossUBCy = mse(dlU0BCyPred,dlU0BCy);

dlU0BCx2Pred = model_CPB_LBFGS(parameters,dlX0BCx,dlY0BCx);
dlU0BCx2Predx = dlgradient(sum(dlU0BCx2Pred,'all'),dlX0BCx,'EnableHigherDerivatives',true);
dlU0BCx2Predxx = dlgradient(sum(dlU0BCx2Predx,'all'),dlX0BCx,'EnableHigherDerivatives',true);
dlU0BCx2Predy = dlgradient(sum(dlU0BCx2Pred,'all'),dlY0BCx,'EnableHigherDerivatives',true);
dlU0BCx2Predyy = dlgradient(sum(dlU0BCx2Predy,'all'),dlY0BCx,'EnableHigherDerivatives',true);
Mxx = (1/scaleX^2)*dlU0BCx2Predxx + (muy/scaleY^2)*dlU0BCx2Predyy;
lossUBCx2 = mse(Mxx,dlU0BCx2);

dlU0BCy2Pred = model_CPB_LBFGS(parameters,dlX0BCy,dlY0BCy);
dlU0BCy2Predx = dlgradient(sum(dlU0BCy2Pred,'all'),dlX0BCy,'EnableHigherDerivatives',true);
dlU0BCy2Predxx = dlgradient(sum(dlU0BCy2Predx,'all'),dlX0BCy,'EnableHigherDerivatives',true);
dlU0BCy2Predy = dlgradient(sum(dlU0BCy2Pred,'all'),dlY0BCy,'EnableHigherDerivatives',true);
dlU0BCy2Predyy = dlgradient(sum(dlU0BCy2Predy,'all'),dlY0BCy,'EnableHigherDerivatives',true);
Myy = (muy/scaleX^2)*dlU0BCy2Predxx + (1/scaleY^2)*dlU0BCy2Predyy;
lossUBCy2 = mse(Myy,dlU0BCy2);

% Combine losses.
loss = lossF + lossUBCx + lossUBCx2 + lossUBCy + lossUBCy2;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);

end
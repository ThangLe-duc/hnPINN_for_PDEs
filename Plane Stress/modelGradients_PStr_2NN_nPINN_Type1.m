% Model Gradients Function
function [gradientsU,gradientsV,loss,lossF,lossUNxx,lossUNyy] = modelGradients_PStr_2NN_nPINN_Type1(parametersU,parametersV,dlX,dlY,...
    dlX0BCx1,dlY0BCx1,dlU0BCx1,dlX0BCy1,dlY0BCy1,dlU0BCy1,dlX0Nxx,dlY0Nxx,dlU0Nxx,dlX0Nyy,dlY0Nyy,dlU0Nyy,a,b,muy,E,...
    scaleX,scaleY,scaleU,scaleV,alpha_u,alpha_v)

% Make predictions.
U = model_PStr_LBFGS_2NN_U(parametersU,dlX,dlY);
% Calculate derivatives of u with respect to X and Y.
gradientsU = dlgradient(sum(U,'all'),{dlX,dlY},'EnableHigherDerivatives',true);
Ux = gradientsU{1};
Uy = gradientsU{2};
Uxx = dlgradient(sum(Ux,'all'),dlX,'EnableHigherDerivatives',true);
Uyy = dlgradient(sum(Uy,'all'),dlY,'EnableHigherDerivatives',true);
Uxy = dlgradient(sum(Ux,'all'),dlY,'EnableHigherDerivatives',true);

% Calculate derivatives of v with respect to X and Y.
V = model_PStr_LBFGS_2NN_V(parametersV,dlX,dlY);
gradientsV = dlgradient(sum(V,'all'),{dlX,dlY},'EnableHigherDerivatives',true);
Vx = gradientsV{1};
Vy = gradientsV{2};
Vxx = dlgradient(sum(Vx,'all'),dlX,'EnableHigherDerivatives',true);
Vyy = dlgradient(sum(Vy,'all'),dlY,'EnableHigherDerivatives',true);
Vxy = dlgradient(sum(Vx,'all'),dlY,'EnableHigherDerivatives',true);

% Calculate loss functions.
fx = E/(1-muy^2)*(scaleU/scaleX^2*Uxx + (1-muy)/2*scaleU/scaleY^2*Uyy + (1+muy)/2*(alpha_v/alpha_u)*scaleV/(scaleX*scaleY)*Vxy);
fy = E/(1-muy^2)*((alpha_v/alpha_u)*scaleV/scaleY^2*Vyy + (1-muy)/2*(alpha_v/alpha_u)*scaleV/scaleX^2*Vxx + (1+muy)/2*scaleU/(scaleX*scaleY)*Uxy);
zeroTargetx = zeros(size(fx), 'like', fx);
zeroTargety = zeros(size(fy), 'like', fy);
lossF = mse(fx, zeroTargetx) + mse(fy, zeroTargety);

dlU0ICxPred = model_PStr_LBFGS_2NN_U(parametersU,dlX0Nxx,dlY0Nxx);
dlV0ICxPred = model_PStr_LBFGS_2NN_V(parametersV,dlX0Nxx,dlY0Nxx);
dlU0ICxPredx = dlgradient(sum(dlU0ICxPred,'all'),dlX0Nxx,'EnableHigherDerivatives',true);
dlV0ICxPredy = dlgradient(sum(dlV0ICxPred,'all'),dlY0Nxx,'EnableHigherDerivatives',true);
dlU0ICxPredy = dlgradient(sum(dlU0ICxPred,'all'),dlY0Nxx,'EnableHigherDerivatives',true);
dlV0ICxPredx = dlgradient(sum(dlV0ICxPred,'all'),dlX0Nxx,'EnableHigherDerivatives',true);
Nxx = E/(1-muy^2)*(scaleU/scaleX*dlU0ICxPredx + muy*(alpha_v/alpha_u)*scaleV/scaleY*dlV0ICxPredy);
Nxy1 = 0.5*E/(1+muy)*(scaleU/scaleY*dlU0ICxPredy + (alpha_v/alpha_u)*scaleV/scaleX*dlV0ICxPredx);
zeroNxy1 = zeros(size(Nxy1), 'like', Nxy1);
lossUNxx = mse(Nxx,dlU0Nxx) + mse(Nxy1,zeroNxy1);

dlU0ICyPred = model_PStr_LBFGS_2NN_U(parametersU,dlX0Nyy,dlY0Nyy);
dlV0ICyPred = model_PStr_LBFGS_2NN_V(parametersV,dlX0Nyy,dlY0Nyy);
dlU0ICyPredx = dlgradient(sum(dlU0ICyPred,'all'),dlX0Nyy,'EnableHigherDerivatives',true);
dlV0ICyPredy = dlgradient(sum(dlV0ICyPred,'all'),dlY0Nyy,'EnableHigherDerivatives',true);
dlU0ICyPredy = dlgradient(sum(dlU0ICyPred,'all'),dlY0Nyy,'EnableHigherDerivatives',true);
dlV0ICyPredx = dlgradient(sum(dlV0ICyPred,'all'),dlX0Nyy,'EnableHigherDerivatives',true);
Nyy = E/(1-muy^2)*(muy*scaleU/scaleX*dlU0ICyPredx + (alpha_v/alpha_u)*scaleV/scaleY*dlV0ICyPredy);
Nxy2 = 0.5*E/(1+muy)*(scaleU/scaleY*dlU0ICyPredy + (alpha_v/alpha_u)*scaleV/scaleX*dlV0ICyPredx);
zeroNxy2 = zeros(size(Nxy2), 'like', Nxy2);
lossUNyy = mse(Nyy,dlU0Nyy) + mse(Nxy2,zeroNxy2);

% Combine losses.
loss = lossF + lossUNxx + lossUNyy;

% Calculate gradients with respect to the learnable parameters.
gradientsU = dlgradient(loss,parametersU);
gradientsV = dlgradient(loss,parametersV);

end
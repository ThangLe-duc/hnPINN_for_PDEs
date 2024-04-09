# hnPINN_for_PDEs
Physics-informed neural networks (PINNs) usually confront significant difficulties to accurately solve partial differential equations (PDEs) due to many pathologies caused by gradient failures during training process. Particularly, the regular PINN approaches can obtain good predictions for trivial problems but may fail to learn solutions when their PDE coefficients and/or physical domain sizes vary. To overcome the abovementioned restrictions, a hierarchically normalized PINN (hnPINN) is devised. The key idea of the proposed hnPINN is, on the one hand, to transform the original PDE system into one of two proposed dimensionless forms to alleviate the negative effects of PDE coefficients and domain size; on the other hand, to use secondary output scalers to flexibly calibrate the gradient flow for training effectively and improving the solution preciseness. The determination of the secondary output scaler is formulated by a heuristic framework inspired from theoretical analyses on the hnPINN gradient flow. The obtained results from some typical PDEs and common problems in solid mechanics strongly confirm the high effectiveness of the hnPINN in practice.

Source codes of each example in our manuscript are provided in each folder. To implement the traditional ndPINN and the proposed hnPINN, the user just needs to change the parameter "alpha" at the head of main file. In particular, alpha = 1 is for the ndPINN and alpha = specific value is for the hnPINN, in that how to determine the specific value was described in the manuscript.

# Programmer
Thang Le-Duc, email: le.duc.thang0312@gmail.com

# Reference
...

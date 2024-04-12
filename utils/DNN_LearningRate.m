function [lrValue, lrInit, lrDropFrac, lrTepoch] = DNN_LearningRate(MaxEpoch, lrSchedule)
% Determine learning rate
switch lrSchedule
    case 'piecewise'
        lrValue = 1e-2;
        lrInit = lrValue;
        lrDropFrac = 0.5;
        lrTepoch = 2000;
    case 'time-based'
        lrValue = 0.1e-2;
        lrInit = lrValue;
        lrDropFrac = 0.001;
        lrTepoch = 1000;
    case 'exponential'
        lrValue = 1e-2;
        lrInit = lrValue;
        lrDropFrac = lrValue/MaxEpoch;
        lrTepoch = 2000;
    case 'step'
        lrValue = 1e-2;
        lrInit = lrValue;
        lrDropFrac = 0.5;
        lrTepoch = 1000;
    case 'none'
        lrValue = 1e-3;
        lrInit = lrValue;
        lrDropFrac = 0.5;
        lrTepoch = 200;
end
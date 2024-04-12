function lrValue = LRSchedule(lrValue, lrInit, lrDropFrac, lrTepoch, lrSchedule, igen, iter)
switch lrSchedule
    case 'none'
        
    case 'piecewise'
        if mod(igen-1,lrTepoch) == 0 && (igen-1 > 0)
            lrValue = lrValue * lrDropFrac;
        end
        
    case 'time-based'
        lrValue = lrInit/(1 + lrDropFrac*igen);
        
    case 'exponential'
        lrValue = lrInit*exp(-lrDropFrac*igen);
        
    case 'step'
        lrValue = lrInit*exp(-lrDropFrac*floor(igen/lrTepoch));
end
end


function [resp,ps] = mapminmax (mode,y,a=-1,b=1,ps=0,psoffset=0)

  if (strcmp(mode,'apply')==1)
    [resp,ps]=normalize(y,a,b,ps);
  elseif (strcmp(mode,'reverse')==1)
    resp=denormalize(y,ps,a,b,psoffset)
  
  endif
endfunction



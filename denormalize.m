

function [resp]=denormalize(y,ps,a=-1,b=1,psoffset)
%ps to include xmin, xmax arrays per row
%psoffset is a way to denormalize based on an offset from the original ps
%e.g. in cases when we need only the output column to be denormalized

%y is assumed to be rows: number of inputs,  columns: concrete data points e.g. 1x105
      
      %DENORMALIZATION from (a,b) to (A,B) included in ps
      
            
      
      sizey=size(y)
      row_index=1;
      
      for row_index=1:sizey(1)
        
        A=ps.xmin(row_index+psoffset);
        B=ps.xmax(row_index+psoffset);
        norm_index2=1;
        for norm_index2=1:sizey(2)
                resp(row_index,norm_index2)=(A-B)*y(row_index,norm_index2)/(a-b)+(a*B-b*A)/(a-b);
        end
      end
      
      
 endfunction %no need to return the values explicitely
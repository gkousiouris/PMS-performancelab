

function [resp,ps]=normalize(y,a=-1,b=1,ps)

%y is assumed to be rows: number of inputs, columns: concrete data points e.g. 1x105
%if ps has the default value of 0, the min max are extracted from the dataset y 
      if (isstruct(ps))
      
      else
        ps=struct('xmin',[],'xmax', []);
        sizey=size(y)
        for i=1:sizey(1)
          ps.xmin(i)=min(y(i,1:end));
          ps.xmax(i)=max(y(i,1:end));
                  
        end
      endif
      %NORMALIZATION FROM XMIN, XMAX to (a,b)
     
      ps
      sizey=size(y);
      for row_index=1:sizey(1)
        A=ps.xmin(row_index)
        B=ps.xmax(row_index)
        for norm_index2=1:sizey(2)
                resp(row_index,norm_index2)=a+ (y(row_index,norm_index2)-A)*(b-a)/(B-A);   
        end
      end
      save ps.mat ps
      
 endfunction 
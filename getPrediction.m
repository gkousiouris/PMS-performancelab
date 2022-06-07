## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

function [retval] = getPrediction (modelID, timestamp, numPoints)

workDir=['/models/',modelID];
loadcommand=['load ',workDir,'/bestmodel.mat ']
eval(loadcommand)
loadcommand=['load ',workDir,'/config.mat'];
eval(loadcommand)
loadcommand=['inputsampleWithExtraLine=csvread("',workDir,'/estimation_',timestamp,'.csv")'];
eval(loadcommand)

whos
inputsample=inputsampleWithExtraLine(2:end,1:end);
%inputsample=inputsampleWithExtraLine; %remove after test and replace with above

inputsample=inputsample';
[inputsample,ps]=normalize(inputsample,-1,1,ps); %use ps argument in order to use preexisting ps from training set

inputsize=size(inputsample);



for final_sample_index=1:str2num(numPoints) %numPoints here means number of input points
    final_sample_index
  eval(['yfin(',num2str(final_sample_index),',1)=sim(net,inputsample(1:end,',num2str(final_sample_index),'));' ]); %CHANGE?TO 1:4
  final_sample_index=final_sample_index+1;
end
yfin=yfin';
yfin=denormalize(yfin,ps,-1,1,str2num(numInputs));
yfin=yfin';
       

eval(['csvwrite("',workDir,'/out_',timestamp,'.csv",yfin)'])

endfunction

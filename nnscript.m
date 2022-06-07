function [perf]=nnscript(l)
  warning off;
%The number of variables passed from the GA depends only on the max number of layers, not the other parameters (neurons/layer or tf/layer)  
l;
silent_functions
%rough fix of ga bounds bug
t=1;
for t=1:19;
	if l(t)>1
		l(t)=0.5;
	endif
	if l(t)<0
		l(t)=0.5;
	endif
	t=t+1;
end 

l;
%end of bugfix

NLmax=5;
NperLmax=30;
maxtransfer=3;

%MHPWS 8ELEI KAI GIA TA ALLA EDW?FAINETAI MONO GIA TA PRWTA LAYERS NA TA KANEI  
net.layers{1}.initFcn = 'initwb';
net.inputWeights{1,1}.initFcn = 'rands';
net.biases{1,1}.initFcn = 'rands';
net.biases{2,1}.initFcn = 'rands';


%TO GA TA STELNEI ME ENA ORIO MONO
%8ELEI NORMALIZE APO TO DIASTHMA TO GENIKO POU 8A EXOUME SE AYTO POU 8ELOUME
%TYPOS NORMALIZATION APO TO (a,b) sto (A,B)
%y=(A-B)*x/(a-b)+(a*B-b*A)/(a-b)
%(a,b)=(1,3) GIA TO TF MEROS (l(2)->(l(11))
%VAZOUME TA ORIA TOU TF NA ERXONTAI KATEY8EIAN OPOTE 8ELOUME 2 ANTI GIA TREIS NORMALIZATIONS
a=0;
b=1;

%NORMALIZATION GIA TO l(1)-NUMBER OF LAYERS
A=3;
B=NLmax;

l(1)=(A-B)*l(1)/(a-b)+(a*B-b*A)/(a-b);

%NORMALIZATION FOR l(2)-l(11)- tf per layer
A=1;
B=maxtransfer;
k=2;
for k=2:11
	l(k)=(A-B)*l(k)/(a-b)+(a*B-b*A)/(a-b);
	k=k+1;
end


%NORMALIZATION GIA TO l(12)-l(19)- NEURONS PER LAYER
A=1;
B=NperLmax;
k=12;
for k=12:19
  l(k)=(A-B)*l(k)/(a-b)+(a*B-b*A)/(a-b);
  k=k+1;
end 


%ROUND GIA OLA GIA NA GINOUN AKERAIOI
k=1;
for k=1:19
  l(k)=round(l(k));
  k=k+1;
end
l  
%CREATE DYNAMIC COMMAND FOR NEWFF


translist={'''logsig''','''tansig''','''purelin'''};


%Dynamic number of input layers INSIDE PARAMS.M
%load params.m;

load config.mat;

workDir=['/models/',modelID];
loadcommand=['load ',workDir,'/config.mat'];
eval(loadcommand)


%num_inputs=num2str(num_inputs);%REMOVE THIS IF STRING
%num_outputs=num2str(num_outputs);

num_inputs=numInputs;
num_outputs=numOutputs;
%SAVED LIKE NUMBERS



layerbegin=['[',num_inputs];
layerend=[',',num_outputs,']'];  %eksodou,se alles periptwseis mporei na einai parapanw (p.x. cycles,storage,network)

z=l(2); %sthles tou x apo 2 ews +max periptwseis tf,min 1 max osa exoume sto translist
%neurons_per_layer=round(x(i,2+NLmax))% sthles tou x apo 2+maxtfn ews 61, ari8mos neyrwnwn gia kathe layer min 1 max 30


layer_number=l(1);
flag2='VALUE OF Z';
z;
tfrepeat=char(translist(z)); 

%DYNAMICALLY ADD -1 1 FIELDS ACCORDING TO NUM_INPUTS
num_inputs=str2num(num_inputs);
input_index=1;
input_range='-1 1';
for input_index=2:num_inputs
		input_range=[input_range,';-1 1'];
		input_index=input_index+1;
end

%tfbegin=['net = newff([-1 1;-1 1;-1 1;-1 1], layerbegin , {',char(translist(z))];  %input layer  CHANGE FOR INPUTS
tfbegin=['net = newff([',input_range,'], layerbegin , {',char(translist(z))];  %input layer  CHANGE FOR INPUTS

space=' ';

k=1;
for k=1:(layer_number-2), % build vector of layers- regards internal layers only-no input output
        
    neurons_per_layer=num2str(l(1+NLmax+k));% sthles tou x apo 2+layernumber ews 61, ari8mos neyrwnwn gia kathe layer min 1 max 28 (-2 layers,input,ouput)
        %ok
    z=round(l(3+k)); %sthles tou x apo 2 ews +max periptwseis layer_number
    
    layerbegin=strcat(layerbegin,',',neurons_per_layer);
    tfbegin=[tfbegin, space,char(translist(z))];%h mhpws 8elei char(translist(z));  
end

z=l(3);%output, to pairnoume stadar sth trith 8esh gia meta de kseroume poses tha vgoun

tfbegin=[tfbegin, space, char(translist(z)),'});']; %default synh8ws einai to purelin

layerbegin=strcat(layerbegin,layerend);
layerbegin=str2num(layerbegin);


tfbegin;

eval(tfbegin);

warning off all


%net = init(net);
net.trainParam.epochs =150;	%(Max no. of epochs to train) [100]
net.trainParam.goal =0.0000001;		%(stop training if the error goal hit) [0]
net.trainParam.lr ='trainlm';%x(i);		%(learning rate, not default trainlm) [0.01]
%net.trainParam.show =NaN;		%(no. epochs between showing error) [25]
net.trainParam.time =500;	%(Max time to train in sec) [inf]

l;


net.trainFcn='trainlm';

%load elearnNEW.mat %UNFINISHED---normally it will take the dataset from inside the param file and divide it into p,t,etc.

load setup


%load elearnver2_time.mat;
loadcommand=['load ',workDir,'/elearnver2_time.mat']
eval(loadcommand)


loadcommand=['load ',workDir,'/indicator']
eval(loadcommand)

net = train(net, p, t);
Ytrain = sim(net,p);
etrain=Ytrain-t;

%manual mse
etrainsqrandsum=etrain*etrain'; %equal me ^2 ka8e etrain(i) kai a8roisma olwn twn synistwswn
sizeetrain=size(etrain)
thismse=etrainsqrandsum/sizeetrain(2);


Yest=sim(net,medvalyin);


supplement_size=size(medvalyout);


%%%deal with no mapminmax
%TYPOS NORMALIZATION APO TO (a,b) sto (A,B)
A=ps.xmin(length(ps.xmin));
B=ps.xmax(length(ps.xmin));
a=-1;
b=1;

for denorm_index1=1:supplement_size(2)
	medvalyout(denorm_index1)=(A-B)*medvalyout(denorm_index1)/(a-b)+(a*B-b*A)/(a-b);
end

for denorm_index2=1:supplement_size(2)
	Yest(denorm_index2)=(A-B)*Yest(denorm_index2)/(a-b)+(a*B-b*A)/(a-b);
end

medvalyout;
Yest;

%%%%end of mapminmax


e=Yest-medvalyout;

size_e=size(e)
%to e prepei na diaire8ei me to abs(medvalyout) gia na einai %
med_sample_index=1;
for med_sample_index=1:size_e(2)      
     if (medvalyout(med_sample_index)==0)
       medvalyout(med_sample_index)=0.001;            %rough fix for Inf values because of value=0
     else         
     e(med_sample_index)=(e(med_sample_index)/abs(medvalyout(med_sample_index)));
     med_sample_index=med_sample_index+1;
     end
     %medvalyout
end
e;
medium_validation_metric=mean(abs(e))
    %this will be in an if clause, condition for good validation in order
    %to save the nn
    
    
    
    
%if mse(etrain)<1*10^(-28)
    if medium_validation_metric<0.20 % 20% error
        disp(".............................")
        disp(".........SAVING MODEL........")
        disp(".............................")
        eval(['net',num2str(indicator),'=','net;']);
        %eval(['save -binary ',path,'\net',num2str(indicator),' net',num2str(indicator)]);
        eval(['save -binary ',workDir,'/bestnets/net',num2str(indicator),' net',num2str(indicator)]);
        indicator=indicator+1;
        %savespergen(generation)=savespergen(generation)+1;
        %save indicator indicator savespergen
        %savecommand=['save ',path,'indicator indicator savespergen']
        savecommand=['save ',workDir,'/indicator indicator savespergen'];
        eval(savecommand);
    end
%end

%perf=medium_validation_metric
perf=thismse;%to be changed

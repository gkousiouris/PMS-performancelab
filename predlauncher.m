args = argv();
numInputs=args{3};
modelID=args{1};
timestamp=args{2};

getPrediction(modelID, timestamp, numInputs)
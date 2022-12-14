## Test 1
kFolds = 5
numEpochs = 200
batchSize = 128
learningRate = 1e-3
weightDecay = 1e-4
lossFunction = nn.MSELoss()
layerDimension=[32, 16, 8]
dropoutPercents=[0.025, 0.05, 0.1]
ExponentialLR(optimizer, gamma=0.1)
![](20221206123915.png)  
![](20221206123929.png)  
## Test 2: Larger Learning rate

kFolds = 5
numEpochs = 200
batchSize = 128
learningRate = 4e-3
weightDecay = 1e-4
lossFunction = nn.MSELoss()
layerDimension=[32, 16, 8]
dropoutPercents=[0.025, 0.05, 0.1]
ExponentialLR(optimizer, gamma=0.5)
![](20221206132324.png)  
![](20221206132335.png)  

## Test 3: consine annealing: Not fitting for simple network
kFolds = 5
numEpochs = 200
batchSize = 128
learningRate = 1e-2
weightDecay = 1e-3
lossFunction = nn.MSELoss()

network = Net2(catDims=[], cntNum=len(col)-1, out_sz=1, layerDimension=[32, 16, 8],
            dropoutPercents=[0.025, 0.05, 0.1], y_range=[0.0, 1.0]).to(device)

lr_cosine = CosineAnnealingLR(optimizer, T_max=5000, eta_min=1e-5)  # T_max: width of cosine/

- ![](20221206141633.png)  
- ![](20221206141649.png) 

## Test 4:

## Test 5: cosine annealing with only 1 curve
kFolds = 5
numEpochs = 200
batchSize = 128
learningRate = 1e-3
momentum = 0.9
weightDecay = 1e-3
lossFunction = nn.MSELoss()
layerDimension=[32, 16, 8]
dropoutPercents=[0.025, 0.05, 0.1]
CosineAnnealingLR(optimizer, T_max=90000, eta_min=1e-5)
![](20221206231008.png)  
![](20221206231017.png)  

## Test 7: 
kFolds = 5
numEpochs = 400
batchSize = 128
learningRate = 7e-3
momentum = 0.9
weightDecay = 1e-3
lossFunction = nn.MSELoss()
network = Net2(catDims=[], cntNum=len(col)-1, out_sz=1, layerDimension=[32, 16, 8],
                dropoutPercents=[0.025, 0.05, 0.1], y_range=[0.0, 1.0]).to(device)
scheduler = CosineAnnealingLR(optimizer, T_max=180000, eta_min=1e-5)  # T_max: width of cosine/
![](20221207210114.png)
![](20221207210126.png)  
## Test 8: consine annealing
kFolds = 5
numEpochs = 100
batchSize = 128
learningRate = 7e-3
momentum = 0.9
weightDecay = 1e-3
lossFunction = nn.MSELoss()

network = Net2(catDims=[], cntNum=len(col)-1, out_sz=1, layerDimension=[32, 16, 8],
                dropoutPercents=[0.025, 0.05, 0.1], y_range=[0.0, 1.0]).to(device)
scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-5)  # T_max: width of cosine/

![](20221207225005.png)  
![](20221207225017.png)  

## Reduce learning Rate
kFolds = 5
numEpochs = 100
batchSize = 128
learningRate = 7e-3
momentum = 0.9
weightDecay = 1e-3
lossFunction = nn.MSELoss()

network = Net2(catDims=[], cntNum=len(col)-1, out_sz=1, layerDimension=[32, 16, 8],
                dropoutPercents=[0.025, 0.05, 0.1], y_range=[0.0, 1.0]).to(device)
scheduler = CosineAnnealingLR(optimizer, T_max=4e4, eta_min=1e-5)


![](20221207233332.png)  
![](20221207233342.png)

## Learning rate too high

kFolds = 5
numEpochs = 200
batchSize = 128
learningRate = 1e-2
momentum = 0.9
weightDecay = 1e-3
lossFunction = nn.MSELoss()

network = Net2(catDims=[], cntNum=len(col)-1, out_sz=1, layerDimension=[32, 16, 8],
                dropoutPercents=[0.025, 0.05, 0.1], y_range=[0.0, 1.0]).to(device)
scheduler = CosineAnnealingLR(optimizer, T_max=9e4, eta_min=1e-5)


![](20221208003408.png)  
![](20221208003416.png)  


kFolds = 5
numEpochs = 200
batchSize = 128
learningRate = 1e-3
momentum = 0.9
weightDecay = 1e-3
lossFunction = nn.MSELoss()
layerDimension=[64, 32, 16, 8]
dropoutPercents=[0.025, 0.05, 0.1, 0.2]
torch.optim.lr_scheduler.MultiStepLR(optimizer, 
    milestones=[20000,60000,120000], gamma=0.1)
    
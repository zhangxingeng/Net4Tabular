### API
```python
# Psudo code
def Train(trainFile, modelName):
    trainData = load(trainFile)
    model = train(trainData)
    save(model, "./model/" + modelName)

def Predict(predictFile, modelName):
    dataNoApi = load(predictFile)
    model = load("./model/" + modelName)
    result = predict(dataNoApi)
    return dataNoApi + result
```
1. File Content
   - DataColumns: `["latitude", "longitude", "value", "so2", "no2", "o3", "value2"]`
   - PredictionColumns = `["aqi"]`
   - trainFile contains: DataColumns + PredictColumns
   - predictFile contains: DataColumns
   - res contains: DataColumns (from PredictFile) + PredictColumns
2. Model File (`"./model/modelName.pth"`)
   - After training, the model can be output into a file
   - **Loading model $\rightarrow$ predict $===$ training $\rightarrow$ predict**
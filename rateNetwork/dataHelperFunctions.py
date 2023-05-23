import dill as pickle
def loadData(fileName):
    return pickle.load(open(f'./simulationData/{fileName}.pkl','rb'))
def saveData(data,fileName):
    pickle.dump(data,open(f'./simulationData/{fileName}.pkl','wb'))
import numpy as np
def getAllReservoirData(goNoGoNetworkRun):
    networkHistoryArr = np.array(goNoGoNetworkRun['testingNetworkStateHistory'])
    dataShape = networkHistoryArr.shape
    collapsedData = networkHistoryArr.reshape(dataShape[2],dataShape[0]*dataShape[1])
    return collapsedData

def getReservoirDataByTrials(goNoGoNetworkRun):
    networkHistoryArr = np.array(goNoGoNetworkRun['testingNetworkStateHistory'])
    testingTrialChoice = goNoGoNetworkRun['testingTrialChoice']
    goTrials = [networkHistoryArr[i] for i in range(len(testingTrialChoice)) if testingTrialChoice[i] == 0]
    noGoTrials = [networkHistoryArr[i] for i in range(len(testingTrialChoice)) if testingTrialChoice[i] == 1]
    emptyTrials = [networkHistoryArr[i] for i in range(len(testingTrialChoice)) if testingTrialChoice[i] == 2]
    goTrialData = np.array(goTrials)
    noGoTrialData = np.array(noGoTrials)
    emptyTrialData = np.array(emptyTrials)
    return goTrialData,noGoTrialData,emptyTrialData

def singleNeuronData(goNoGoNetworkRun,neuron):
    goTrials,noGoTrials,emptyTrials = getReservoirDataByTrials(goNoGoNetworkRun)
    return (goTrials[:,:,neuron],noGoTrials[:,:,neuron],emptyTrials[:,:,neuron])
    
def dims(data,cutoff=.90):
    dataCentered = data - data.mean(axis=1).reshape((data.shape[0],1))*np.ones((1,data.shape[1]))
    covarianceMatrix = np.matmul(dataCentered,dataCentered.T)/data.shape[1]
    eigvalue,eigvectors = np.linalg.eig(covarianceMatrix)
    totalEigenvalues = sum(eigvalue)
    for i in range(len(eigvectors)):
        if sum(eigvalue[:i])/totalEigenvalues > cutoff:
            break
    return {'fractional':sum(eigvalue)**2/sum(np.square(eigvalue)),'cutoff':i}
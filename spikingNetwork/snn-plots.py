import numpy as np
import matplotlib.pyplot as plt
import dill as pickle
def buildRasterPlot(simOutputDict,inhibitory=True):
    excitatorySpikes = simOutputDict['simExcitatorySpikes']
    inhibitorySpikes = simOutputDict['simInhibitorySpikes']
    print(np.sort(excitatorySpikes.sum(axis=1)))
    print(np.sort(inhibitorySpikes.sum(axis=1)))
def pca(data,pcs):
    dataCentered = data - data.mean(axis=1).reshape((data.shape[0],1))*np.ones((1,data.shape[1]))
    covarianceMatrix = np.matmul(dataCentered,dataCentered.T)/data.shape[1]
    eigvalue,eigvectors = np.linalg.eig(covarianceMatrix)
    firstEigvectors = eigvectors[:,:pcs]
    projectedData = np.matmul(firstEigvectors.T,data)
    return eigvalue[:pcs],eigvectors[:,:pcs],projectedData
def dims(data,cutoff=.99):
    dataCentered = data - data.mean(axis=1).reshape((data.shape[0],1))*np.ones((1,data.shape[1]))
    covarianceMatrix = np.matmul(dataCentered,dataCentered.T)/data.shape[1]
    eigvalue,eigvectors = np.linalg.eig(covarianceMatrix)
    totalEigenvalues = sum(eigvalue)
    for i in range(len(eigvectors)):
        if sum(eigvalue[:i])/totalEigenvalues > cutoff:
            break
    return {'fractional':sum(eigvalue)**2/sum(np.square(eigvalue)),'cutoff':i}
def buildStackedTimeGraph(timePoints,data,title):
    ''' time along columns, data along rows
    '''
    dataRows = data.shape[0]
    maxPlotFit = min(dataRows,5)
    fig,axs = plt.subplots(maxPlotFit)
    for i in range(maxPlotFit):
        axs[i].plot(timePoints,data[i,:])
    plt.suptitle(title)
    plt.show()
def plotBranchingFactor(branchingNumbers):
    plt.plot(branchingNumbers)
    plt.show()
def getBranchingFactor(simData,delayTime,windowSize,branchingFactorType='simpleRatio'):
    ''' branching factorType should be one of: {"simpleRatio","causalRatio"}
        the difference being that the causal ratio will double count some spikes
        depending on the causal structure
    '''
    eulerStepSize = simData['model'].eulerTimeDelta
    eulerTimeSteps = simData['model'].eulerTimeSteps
    delayTimeSteps = round(delayTime/eulerStepSize)
    windowTimeSteps = round(windowSize/eulerStepSize)
    EEWeightMatrix = simData['model'].excitatoryToExcitatoryWeightMatrix
    EIWeightMatrix = simData['model'].excitatoryToInhibitoryWeightMatrix
    IIWeightMatrix = simData['model'].inhibitoryToInhibitoryWeightMatrix
    IEWeightMatrix = simData['model'].inhibitoryToExcitatoryWeightMatrix
    inhibitorySpikeMatrix = simData['trainDict']['simExcitatorySpikes']
    excitatorySpikeMatrix = simData['trainDict']['simInhibitorySpikes']
    totalWeightMatrix = np.block([[EEWeightMatrix,IEWeightMatrix]
              ,[EIWeightMatrix,IIWeightMatrix]])
    totalSpikeMatrix = np.block([[excitatorySpikeMatrix]
                                 ,[inhibitorySpikeMatrix]])
    causalMatrix = totalWeightMatrix != 0
    #empty array for iterating over
    branchingFactorTimeVector = np.zeros(eulerTimeSteps-2*delayTimeSteps-2*windowTimeSteps-1)
    for timeStep in range(delayTimeSteps+windowTimeSteps+1,eulerTimeSteps-delayTimeSteps-windowTimeSteps):
        upstreamActivity = totalSpikeMatrix[:,timeStep-delayTimeSteps-windowTimeSteps:timeStep-delayTimeSteps]
        downstreamActivity = totalSpikeMatrix[:,timeStep+delayTimeSteps:timeStep+delayTimeSteps+windowTimeSteps]
        upstreamUnion = np.any(upstreamActivity,axis=1)
        downstreamUnion = np.any(downstreamActivity,axis=1)
        currentSpikeVector = totalSpikeMatrix[:,timeStep][:,np.newaxis]
        #divide by 0 issues
        if branchingFactorType == 'simpleRatio':
            branchingFactorTimeVector[timeStep - (delayTimeSteps+windowTimeSteps+1)] = (sum(upstreamUnion)+1)/(sum(downstreamUnion)+1)
            #branchingFactorTimeVector[timeStep]=(sum(upstreamUnion)+1)/(sum(downstreamUnion)+1)
        else:
            upstreamUnionVector = upstreamUnion.reshape(len(upstreamUnion),1)
            downstreamUnionVector = downstreamUnion.reshape(len(downstreamUnion),1)
            presynapticSpikesCountVector = np.matmul(causalMatrix,upstreamUnionVector)
            postsynapticSpikesCountVector = np.matmul(causalMatrix.T,downstreamUnionVector)
            #individualNeuronBranchingFactor = presynapticSpikesCountVector/postsynapticSpikesCountVector
            populationBranchingFactor = (np.matmul(presynapticSpikesCountVector.T,currentSpikeVector)+1)/(np.matmul(postsynapticSpikesCountVector.T,currentSpikeVector)+1)
            branchingFactorTimeVector[timeStep]=populationBranchingFactor
    return branchingFactorTimeVector
simData = pickle.load(open("./rnnTestbed/reservoirData.pkl",'rb'))
trainDict = simData['trainDict']
model = simData['model']
#branchingFactor = getBranchingFactor(simData,10,20,'simpleRatio')
#print(np.average(branchingFactor))
#plotBranchingFactor(branchingFactor)
#buildRasterPlot(trainDict)
print(dims(trainDict['simExcitatoryOutputStates']))
#eVarianceExplained, epcs, eprojectedData = pca(trainDict['simExcitatoryOutputStates'],5)
#iVarianceExplained, ipcs, iprojectedData = pca(trainDict['simInhibitoryOutputStates'],5)
#buildStackedTimeGraph(trainDict['time'],eprojectedData,'ePC output')
#buildStackedTimeGraph(trainDict['time'],iprojectedData,'iPC output')
#buildStackedTimeGraph(trainDict['time'],trainDict['simExcitatoryOutputStates'],'Excitatory Output')
#buildStackedTimeGraph(trainDict['time'],trainDict['simInhibitoryOutputStates'],'Inhibitory Output')
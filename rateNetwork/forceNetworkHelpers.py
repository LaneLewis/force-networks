import numpy as np
from forceNetwork import ForceNetwork,ReservoirNetwork
class RecursiveLeastSquares():
    def __init__(self,stateLength,regularization=.1):
        self.P = np.identity(stateLength)*regularization
    def updateWs(self,X_i,Y_i,lastWeights):
        X_i_t = np.transpose(X_i)
        pXi = np.matmul(self.P,X_i)
        pInverseApx = 1+np.matmul(X_i_t,pXi)
        #np.matmul(np.matmul(pXi,np.matmul(x_i_t,self.P))
        #pNumerator = np.matmul(self.P,np.matmul(X_i,np.matmul(X_i_t,self.P)))
        pNumerator = np.matmul(pXi,np.matmul(X_i_t,self.P))
        P_next = self.P - pNumerator/pInverseApx
        K = np.matmul(P_next,X_i)
        error = np.matmul(lastWeights.T,X_i) - Y_i
        #w_next = lastWeights+np.matmul(K,error)
        w_next = lastWeights  - np.matmul(K,error.T)
        self.P = P_next
        return w_next
    
def weightedFeedbackFunction(alpha=.1):
    def feedback(newNeuralState,newNetworkOutput,lossArr):
        return newNetworkOutput*alpha
    return feedback

def gammaArctanActivation(gamma):
    def arctanActivation(externalInput,currentNeuralState,internalWeights,inputWeights,feedback):
        inputActivation = np.matmul(inputWeights,externalInput)
        selfActivation = np.matmul(internalWeights,currentNeuralState)
        newNetworkRate = (1-gamma)*currentNeuralState+gamma*np.arctan(selfActivation+inputActivation+feedback)
        return newNetworkRate
    return arctanActivation

def sparseUniformConnectionMatrix(connectionProbability,matrixShape,mean=0,width=1,rescale=True):
    rnnWeights = np.random.uniform(mean-width,mean+width,matrixShape)
    weightExists = np.random.binomial(1, connectionProbability,matrixShape)
    sparseMatrix = rnnWeights*weightExists
    if rescale == True:
        sparseMatrix = sparseMatrix/largestSingularValue(sparseMatrix)
    return sparseMatrix

def sparseNormalConnectionMatrix(connectionProbability,matrixShape,mean=0,std=1,rescale=True):
    rnnWeights = np.random.normal(mean,std,matrixShape)
    weightExists = np.random.binomial(1,connectionProbability,matrixShape)
    sparseMatrix = rnnWeights*weightExists
    if rescale == True:
        sparseMatrix = sparseMatrix/largestSingularValue(sparseMatrix)
    return sparseMatrix

def randomNormalInitialRates(networkSize,trials,mean=0,std=1):
    return [np.random.normal(mean,1,(networkSize,1)) for _ in range(trials)]

def squareLossFunction(predictedMatrix,correctMatrix):
    return sum(np.square(predictedMatrix-correctMatrix))

def largestSingularValue(matrix):
    return np.linalg.svd(matrix)[1][0]

def sparseArctanForceNetwork(networkSize,inputDims,outputDims,inputConnectionProbability,
                             feedbackConnectionProbability,recurrentConnectionProbability,
                             recurrentStandardDeviation,inputWidth,feedbackWidth,inputWeightScaling,
                             feedbackWeightScaling,recurrentWeightScaling,leastSquaresRegularization,
                             gamma):
    inputConnectionMatrix = inputWeightScaling*sparseUniformConnectionMatrix(inputConnectionProbability,(networkSize,inputDims),mean=0,width=inputWidth,rescale=True)
    feedbackConnectionMatrix = feedbackWeightScaling*sparseUniformConnectionMatrix(feedbackConnectionProbability,(networkSize,outputDims),mean=0,width=feedbackWidth,rescale=True)
    recurrentConnectionMatrix = recurrentWeightScaling*sparseNormalConnectionMatrix(recurrentConnectionProbability,(networkSize,networkSize),mean=0,std=recurrentStandardDeviation,rescale=True)
    recursiveLeastSquares = RecursiveLeastSquares(networkSize,leastSquaresRegularization)
    feedbackFunction = weightedFeedbackFunction(1)
    initialOutputWeights = np.zeros((networkSize,outputDims))
    forceNet = ForceNetwork(networkSize,gammaArctanActivation(gamma),recurrentConnectionMatrix,
                 inputConnectionMatrix,feedbackConnectionMatrix,recursiveLeastSquares.updateWs,
                 initialOutputWeights,feedbackFunction)
    return forceNet,inputConnectionMatrix,feedbackConnectionMatrix,recurrentConnectionMatrix

def sparseArctanReservoirNetwork(networkSize,inputDims,outputDims,inputConnectionProbability,recurrentConnectionProbability,
                             recurrentStandardDeviation,inputWidth,inputWeightScaling,
                             recurrentWeightScaling,leastSquaresRegularization,gamma):
    inputConnectionMatrix = inputWeightScaling*sparseUniformConnectionMatrix(inputConnectionProbability,0,inputWidth,(networkSize,inputDims),rescale=True)
    recurrentConnectionMatrix = recurrentWeightScaling*sparseNormalConnectionMatrix(recurrentConnectionProbability,0,recurrentStandardDeviation,(networkSize,networkSize),rescale=True)
    recursiveLeastSquares = RecursiveLeastSquares(networkSize,leastSquaresRegularization)
    initialOutputWeights = np.zeros((networkSize,outputDims))
    reservoirNetwork = ReservoirNetwork(networkSize,gammaArctanActivation(gamma),recurrentConnectionMatrix,
                                        inputConnectionMatrix,recursiveLeastSquares.updateWs,initialOutputWeights)
    return reservoirNetwork
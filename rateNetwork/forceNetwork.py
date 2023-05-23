import numpy as np
from tqdm import tqdm
class ForceNetwork():
    def __init__(self,neuronNumber,activationIterativeMap,
                internalWeightConnectionMatrix,inputWeightConnectionMatrix,
                feedbackWeightConnectionMatrix, outputWeightUpdateFunction,
                initialOutputWeights,feedbackFunction):
        ''' inputDimension = ID
            outputDimension = OD
        neuron number: number of neurons,n
        activationIterativeMap: function (externalInput,neuralState,internalWeights,inputWeights,feedback) -> newNeuralState
        internalWeightConnectionMatrix: n by n
        inputWeightConnectionMatrix: n by ID
        feedbackWeightConnectionMatrix: n by OD
        outputWeightUpdateFunction: function(correctOutput,newNeuralState,currentOutputWeights) -> newOutputWeights
        initialOutputWeights: n by OD
        feedbackFunction: function (newNeuralState,correctOutput,currentOutputWeights) -> newFeedback
        '''
        self.neuronNumber = neuronNumber
        self.activationIterativeMap = activationIterativeMap
        self.internalWeightConnectionMatrix = internalWeightConnectionMatrix
        self.inputWeightConnectionMatrix = inputWeightConnectionMatrix
        self.initialOutputWeights = initialOutputWeights
        self.feedbackWeightConnectionMatrix = feedbackWeightConnectionMatrix
        self.outputWeightUpdateFunction = outputWeightUpdateFunction
        self.feedbackFunction = feedbackFunction

    def trainUpdateStep(self,externalInput,currentNeuralState,
                   internalWeights,inputWeights,feedback,
                   currentOutputWeights,correctOutput,
                   feedbackWeightConnectionMatrix,lossArr):
        #updates the neural state
        newNeuralState = self.activationIterativeMap(externalInput,currentNeuralState,
                                    internalWeights,inputWeights,feedback)
        #gives the model prediction
        newNetworkOutput = np.matmul(currentOutputWeights.T,newNeuralState)
        #updates the output weights
        newOutputWeights = self.outputWeightUpdateFunction(newNeuralState,correctOutput,currentOutputWeights)
        #updates the feedback
        newFeedback = self.feedbackFunction(newNeuralState,newNetworkOutput,lossArr)
        #gives the feedback back into the network for the next time
        newForceFeedback = np.matmul(feedbackWeightConnectionMatrix,newFeedback)
        return newNeuralState,newOutputWeights,newNetworkOutput,newForceFeedback
    
    def testUpdateStep(self,externalInput,currentNeuralState,
                   internalWeights,inputWeights,feedback,
                   currentOutputWeights,
                   feedbackWeightConnectionMatrix,lossArr):
        #updates the neural state
        newNeuralState = self.activationIterativeMap(externalInput,currentNeuralState,
                                    internalWeights,inputWeights,feedback)
        #gives the model prediction
        newNetworkOutput = np.matmul(currentOutputWeights.T,newNeuralState)
        #updates the feedback
        newFeedback = self.feedbackFunction(newNeuralState,newNetworkOutput,lossArr)
        #gives the feedback back into the network for the next time
        newForceFeedback = np.matmul(feedbackWeightConnectionMatrix,newFeedback)
        return newNeuralState,newNetworkOutput,newForceFeedback
            
    def train(self,inputMatrixArr,correctOutputMatrixArr,initialNeuralStateArr,lossFunction,progressBar=True):
        #instantiates return variables
        #array of [network states of across trial time]
        networkStateMatrixArr = []
        #array of [total feedback into neural state across trial time]
        feedbackMatrixArr = []
        #array of [output from neural network across trial time]
        outputMatrixArr = []
        #array of [last weight matrix to outputs from trial across time]
        outputWeightsMatrixArr = []
        #array of loss from each trial
        lossArr = []
        #initializes the output weights
        outputWeights = self.initialOutputWeights
        if progressBar:
            trainIterator = tqdm(range(len(inputMatrixArr)),desc='Train',colour='blue')
        else:
            trainIterator = range(len(inputMatrixArr))
        for trialIndex in trainIterator:
            #grabs the number of rows
            trialLength = inputMatrixArr[trialIndex].shape[0]
            #initializes the output variables: the neural state, 
            #the feedback into the network at the next time, and the output of the network
            trialNeuralStateMatrix = np.zeros(shape=(trialLength,self.neuronNumber))
            trialFeedbackMatrix = np.zeros(shape=(trialLength,self.neuronNumber))
            trialOutputMatrix = np.zeros(shape=(trialLength,correctOutputMatrixArr[trialIndex].shape[1]))
            #initializes the specific variables for the trial
            forceFeedback = np.zeros(shape=(self.neuronNumber,1))
            trialInputs = inputMatrixArr[trialIndex]
            trialCorrectOutputs = correctOutputMatrixArr[trialIndex]
            currentNeuralState = initialNeuralStateArr[trialIndex]
            for trialTime in range(trialLength):
                inputAtTrialTime = np.expand_dims(trialInputs[trialTime,:],axis=1)
                correctOutputAtTrialTime = np.expand_dims(trialCorrectOutputs[trialTime,:],axis=1)
                newNeuralState,newOutputWeights,newNetworkOutput,newForceFeedback = self.trainUpdateStep(inputAtTrialTime,
                                    currentNeuralState,self.internalWeightConnectionMatrix,
                                    self.inputWeightConnectionMatrix,forceFeedback,outputWeights,
                                    correctOutputAtTrialTime,self.feedbackWeightConnectionMatrix,lossArr)
                #adds to the output variables
                trialNeuralStateMatrix[trialTime,:] = np.squeeze(newNeuralState,axis=1)
                trialOutputMatrix[trialTime,:] = np.squeeze(newNetworkOutput,axis=1)
                trialFeedbackMatrix[trialTime,:] = np.squeeze(newForceFeedback,axis=1)
                #updates in the loop
                currentNeuralState = newNeuralState
                outputWeights = newOutputWeights
                forceFeedback = newForceFeedback
            #calculates the trial loss
            trialLoss = lossFunction(trialOutputMatrix,trialCorrectOutputs)
            #gets the new weight vector to add to return
            endOfTrialOutputWeights = outputWeights
            #adds to overall outputs
            lossArr.append(trialLoss)
            outputWeightsMatrixArr.append(endOfTrialOutputWeights)
            outputWeightsMatrixArr.append(endOfTrialOutputWeights)
            networkStateMatrixArr.append(trialNeuralStateMatrix)
            feedbackMatrixArr.append(trialFeedbackMatrix)
            outputMatrixArr.append(trialOutputMatrix)

        def prediction(currentNeuralState,forceFeedback,externalInput):
            ''''allows an easy prediction function'''
            newNeuralState,newNetworkOutput,newForceFeedback = self.testUpdateStep(externalInput,currentNeuralState,
                                self.internalWeightConnectionMatrix,self.inputWeightConnectionMatrix,
                                forceFeedback,outputWeights,self.feedbackWeightConnectionMatrix,
                                lossArr)
            return newNeuralState,newNetworkOutput,newForceFeedback
        
        return {'predict': prediction,
                'networkStateHistory':networkStateMatrixArr,
                'feedbackHistory':feedbackMatrixArr,
                'outputHistory':outputMatrixArr,
                'outputWeightsHistory':outputWeightsMatrixArr,
                'lossHistory':lossArr,
                'trainedOutputWeights':outputWeights}
    
    def test(self,inputMatrixArr,correctOutputMatrixArr,initialNeuralStateArr,lossFunction,outputWeightMatrix,progressBar=True):
                    #instantiates return variables
        #array of [network states of across trial time]
        networkStateMatrixArr = []
        #array of [total feedback into neural state across trial time]
        feedbackMatrixArr = []
        #array of [output from neural network across trial time]
        outputMatrixArr = []
        #array of loss from each trial
        lossArr = []
        #initializes the output weights
        if progressBar:
            testIterator = tqdm(range(len(inputMatrixArr)),desc='Test',colour='green')
        else:
            testIterator = range(len(inputMatrixArr))
        for trialIndex in testIterator:
            #grabs the number of rows
            trialLength = inputMatrixArr[trialIndex].shape[0]
            #initializes the output variables: the neural state, 
            #the feedback into the network at the next time, and the output of the network
            trialNeuralStateMatrix = np.zeros(shape=(trialLength,self.neuronNumber))
            trialFeedbackMatrix = np.zeros(shape=(trialLength,self.neuronNumber))
            trialOutputMatrix = np.zeros(shape=(trialLength,correctOutputMatrixArr[trialIndex].shape[1]))
            #initializes the specific variables for the trial
            forceFeedback = np.zeros(shape=(self.neuronNumber,1))
            trialInputs = inputMatrixArr[trialIndex]
            trialCorrectOutputs = correctOutputMatrixArr[trialIndex]
            currentNeuralState = initialNeuralStateArr[trialIndex]
            for trialTime in range(trialLength):
                inputAtTrialTime = np.expand_dims(trialInputs[trialTime,:],axis=1)
                correctOutputAtTrialTime = np.expand_dims(trialCorrectOutputs[trialTime,:],axis=1)
                newNeuralState,newNetworkOutput,newForceFeedback = self.testUpdateStep(inputAtTrialTime,
                                    currentNeuralState,self.internalWeightConnectionMatrix,
                                    self.inputWeightConnectionMatrix,forceFeedback,outputWeightMatrix,self.feedbackWeightConnectionMatrix,lossArr)
                #adds to the output variables
                trialNeuralStateMatrix[trialTime,:] = np.squeeze(newNeuralState,axis=1)
                trialOutputMatrix[trialTime,:] = np.squeeze(newNetworkOutput,axis=1)
                trialFeedbackMatrix[trialTime,:] = np.squeeze(newForceFeedback,axis=1)
                #updates in the loop
                currentNeuralState = newNeuralState
                forceFeedback = newForceFeedback
            #calculates the trial loss
            trialLoss = lossFunction(trialOutputMatrix,trialCorrectOutputs)
            #gets the new weight vector to add to return
            #adds to overall outputs
            lossArr.append(trialLoss)
            networkStateMatrixArr.append(trialNeuralStateMatrix)
            feedbackMatrixArr.append(trialFeedbackMatrix)
            outputMatrixArr.append(trialOutputMatrix)
        return {
            'networkStateHistory':networkStateMatrixArr,
            'feedbackHistory':feedbackMatrixArr,
            'outputHistory':outputMatrixArr,
            'lossHistory':lossArr,
            'averageLoss':np.average(lossArr)
        }
    
def ReservoirNetwork(neuronNumber,activationIterativeMap,
                internalWeightConnectionMatrix,inputWeightConnectionMatrix,
                outputWeightUpdateFunction,initialOutputWeights):
    def identityFeedbackFunction(newNeuralState,newNetworkOutput,lossArr):
            return newNetworkOutput
    zeroOutputWeights = np.zeros((neuronNumber,len(initialOutputWeights)))
    return ForceNetwork(neuronNumber,activationIterativeMap,internalWeightConnectionMatrix,
                    inputWeightConnectionMatrix,zeroOutputWeights,outputWeightUpdateFunction,
                    initialOutputWeights,identityFeedbackFunction)

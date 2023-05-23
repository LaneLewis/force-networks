import numpy as np
from dataHelperFunctions import loadData, saveData
from forceNetworkHelpers import sparseArctanForceNetwork,sparseArctanReservoirNetwork,squareLossFunction,randomNormalInitialRates
from lyapunovExponentHelpers import calculateForceLargestLyapunovExponent,calculateForceLyapunovDimension
from goNoGoTask import goNoGo
import itertools
from tqdm import tqdm

def goNoGoForceLearning(networkSize=500,trainTrials=100,testTrials=100,inputConnectionProbability=.2,feedbackConnectionProbability=.3,
                         recurrentConnectionProbability=.05,recurrentWeightScaling=2.4,goTone=2,noGoTone=6,toneLength=10,gapLength=20,
                         afterLength=10,correctOutputLength=15,goCorrectOutput=10.0,leastSquaresRegularization=100,
                         inputWeightScaling=50,feedbackWeightScaling=1,gamma=1,saveDataPickle=True,calculateLyapunovStability=True,
                         progressBar=True,calculateDimension=True):
    '''

    optional:
        --- overall parameters ---
        networkSize - number of neurons to include in the force network
        trainTrials - number of training trials to run
        testTrials - number of testing trials to run

        --- network parameters ---
        inputConnectionProbability - probability of the 1-d input connecting to a neuron
        feedbackConnectionProbability - probability of the 1-d output connecting to a neuron
        recurrentConnectionProbability - probability of one neuron in the reservoir connecting to another
            neuron in the reservoir. 
        recurrentWeightScaling - the recurrent weight matrix is normalized by its largest singular value.
            recurrentWeightScaling scales the weight matrix after normalization.
        feedbackWeightScaling - the feedback weight matrix is normalized by its largest singular value.
            feedbackWeightScaling scales the weight matrix after normalization. This controls the strength
            of the force feedback.
        gamma - the arctan network hyperparameter

        ---task parameters ---
        goTone - number to be input into the network during the 'input' period during a go trial
        noGoTone - number to be input into the network during the 'input' period during a noGo trial
        toneLength - the number of time steps during the 'input' period of the trial
        gapLength - the number of time steps between the 'input' period of the trial and the 'output'
            period of the trial
        correctOutputLength - the number of time steps in the 'output' period of the trial.
        afterLength - the number of time steps after the 'output' period of the trial.
        goCorrectOutput - the correct response that the network should do on a go trial

        ---data parameters ---
        saveDataPickle - if the data dictionary should be saved into a pickle file under the name 'forceData' in 
            the simulationData folder
        calculateLyapunovStability - if the largest lyapunov exponent and coordinate stabilities should
            be calculated and added to the simulation data
        progressBar - if a progress bar should be displayed for the train, test, and exponent calculations.
            Additionally, a print statement of the major computations will be displayed at the end of the run
        calculateDimension - if the lyapunov dimension should be calculated and added to the simulation data
    
    outputs: dataDict
        dataDict - a dictionary with keys corresponding to the important data about the run. The keys are:
            let trialLength = toneLength+gapLength+correctOutputLength+afterLength
            --- always in dict ---
            'inputMatrix':np.ndarray[1,networkSize] - the connectivity matrix that sends the trial input to the network
            'feedbackMatrix':np.ndarray[1,networkSize]- the connectivity matrix that sends the force feedback into the network
            'recurrentMatrix':np.ndarray[networkSize,networkSize] - the connectivity matrix between neurons in the reservoir of the force network
            'trainingCorrectResponses':list[np.ndarray[1,trialLength]] - the target response for each training run
            'trainingTrialInputs':
            'trainedPredictionFunction':
            'trainingNetworkStateHistory':
            'trainingOutputHistory':
            'trainingOutputWeightsHistory':
            'trainingLossHistory':
            'trainedOutputWeights':
            'trainingTrialChoice':
            'testingCorrectResponses':
            'testingTrialInputs':
            'testingNetworkStateHistory':
            'testingOutputHistory':
            'testingLossHistory':
            'testingAverageLoss':
            'testingTrialChoice':
            'parameters':

    '''
    #saves all the parameters of the network to store in the saved pickle file
    inputParameters = locals()
    #constructs the force network
    forceNetwork,inputMatrix,feedbackMatrix,recurrentMatrix = sparseArctanForceNetwork(networkSize=networkSize,inputDims=1,outputDims=1,inputConnectionProbability=inputConnectionProbability,
                             feedbackConnectionProbability=feedbackConnectionProbability,recurrentConnectionProbability=recurrentConnectionProbability,
                             recurrentStandardDeviation=1,inputWidth=1,feedbackWidth=1,inputWeightScaling=inputWeightScaling,
                             feedbackWeightScaling=feedbackWeightScaling,recurrentWeightScaling=recurrentWeightScaling,
                             leastSquaresRegularization=leastSquaresRegularization,gamma=gamma)
    #sets the initial state on each training trial to a random draw from a normal distribution.
    initialNetworkTrainRates = randomNormalInitialRates(networkSize,trainTrials)
    #constructs the inputs and correct response for the goNoGo task to train on
    trainInputs,trainCorrect,trainTrialChoice = goNoGo(trainTrials,goTone,noGoTone,toneLength,gapLength,correctOutputLength,afterLength,goCorrectOutput)
    trainingModel = forceNetwork.train(trainInputs,trainCorrect,initialNetworkTrainRates,squareLossFunction,progressBar=progressBar)
    #sets the initial state on each testing trial to a random draw from a normal distribution.
    initialNetworkTestRates = randomNormalInitialRates(networkSize,testTrials)
    #constructs the inputs and correct response for the goNoGo task to test on
    testInputs,testCorrect,testTrialChoice = goNoGo(testTrials,goTone,noGoTone,toneLength,gapLength,correctOutputLength,afterLength,goCorrectOutput)
    testModel = forceNetwork.test(testInputs,testCorrect,initialNetworkTestRates,squareLossFunction,trainingModel['trainedOutputWeights'],progressBar=progressBar)
    #sends all the output into a data dict to be stored in a clear way.
    dataDict = outputDict(trainingModel,trainInputs,trainCorrect,trainTrialChoice,
                          testModel,testInputs,testCorrect,testTrialChoice,inputParameters,
                          inputMatrix,feedbackMatrix,recurrentMatrix)
    #dictionary of what to print out at the end of the run. If progressBar is true, this will be printed.
    printDict = {}
    #calculates the largest lyapunov exponent of the force network under zero external input.
    #the initial state is chosen at 0. In addition, the coordinate stability is calculated.
    #the results are added into the dataDict
    if calculateLyapunovStability:
        zeroInput = np.zeros((1,1))
        initialState = np.zeros((networkSize,1))
        largestExponent,coordinateExponents = calculateForceLargestLyapunovExponent(trainingModel['predict'],networkSize,zeroInput,initialState,
                                                                                    withForceOn=True,iterations=3000,progressBar=False)
        dataDict['largestLyapunovExponent'] = largestExponent
        dataDict['coordinateExponent'] = coordinateExponents
        printDict['LLE'] = np.round(largestExponent,4)
    #calculates the lyapunov dimension of the force network under zero external input.
    #the initial state for calculating this is chosen at 0.
    if calculateDimension:
        initialState = np.zeros((networkSize,1))
        lyapunovDimension = calculateForceLyapunovDimension(trainingModel['predict'],networkSize,zeroInput,initialState,withForceOn=True,iterations=1000)
        dataDict['lyapunovDimension'] = lyapunovDimension
        printDict['D_KY'] = np.round(lyapunovDimension,4)
    if progressBar:
        printDict['Average Loss'] = {np.round(testModel['averageLoss'],4)}
        print(printDict)
    #saves the model
    if saveDataPickle:
        saveData(dataDict,'forceData')
    return dataDict

def outputDict(trainingModel,trainInputs,trainCorrect,trainTrialChoice,
               testModel,testInputs,testCorrect,testTrialChoice,inputParameters,
               inputMatrix,feedbackMatrix,recurrentMatrix):
        dataDict ={
        'inputMatrix':inputMatrix,
        'feedbackMatrix':feedbackMatrix,
        'recurrentMatrix':recurrentMatrix,

        'trainingCorrectResponses':trainCorrect,
        'trainingTrialInputs':trainInputs,
        'trainedPredictionFunction':trainingModel['predict'],
        'trainingNetworkStateHistory':trainingModel['networkStateHistory'],
        'trainingOutputHistory':trainingModel['outputHistory'],
        'trainingOutputWeightsHistory':trainingModel['outputWeightsHistory'],
        'trainingLossHistory':trainingModel['lossHistory'],
        'trainedOutputWeights':trainingModel['trainedOutputWeights'],
        'trainingTrialChoice':trainTrialChoice,

        'testingCorrectResponses':testCorrect,
        'testingTrialInputs':testInputs,
        'testingNetworkStateHistory':testModel['networkStateHistory'],
        'testingOutputHistory':testModel['outputHistory'],
        'testingLossHistory':testModel['lossHistory'],
        'testingAverageLoss':np.average(testModel['lossHistory']),
        'testingTrialChoice':testTrialChoice,

        'parameters':inputParameters
        }
        return dataDict

def multirunGoNoGoForceLearning(networkSizeArr=[500],trainTrialsArr=[300],testTrialsArr=[300],inputConnectionProbabilityArr=[.2],feedbackConnectionProbabilityArr=[1],
                         recurrentConnectionProbabilityArr=[.05],recurrentStandardDeviationArr=[.4],recurrentWeightScalingArr=[2.4],
                         goToneArr=[2],noGoToneArr=[6],toneLengthArr=[10],gapLengthArr=[20],afterLengthArr=[10],correctOutputLengthArr=[15],goCorrectOutputArr=[10],
                         feedbackStrengthArr=[.05],leastSquaresRegularizationArr=[1],inputWeightScalingArr=[50],feedbackWeightScalingArr=[50],gammaArr=[1]):
    allParameters = locals()
    completeProduct = list(itertools.product(*list(allParameters.values())))
    allParametersKeys = list(allParameters.keys())
    parameterList = ['networkSize', 'trainTrials', 'testTrials', 'inputConnectionProbability',
                      'feedbackConnectionProbability', 'recurrentConnectionProbability',
                      'recurrentStandardDeviation', 'recurrentWeightScaling', 'goTone',
                      'noGoTone', 'toneLength', 'gapLength', 'afterLength', 'correctOutputLength',
                      'goCorrectOutput', 'feedbackStrength', 'leastSquaresRegularization', 'inputWeightScaling',
                      'feedbackWeightScaling', 'gamma']
    multirunForce = []
    for counter,parameterRun in tqdm(enumerate(completeProduct),total=len(completeProduct)):
        parameterSet = {parameterList[i]:parameterRun[i] for i in range(len(allParametersKeys))}
        runData = goNoGoForceLearning(**parameterSet,saveDataPickle=False,progressBar=True)
        multirunForce.append(runData)
        #print(f"Simulation {round(counter/len(completeProduct)*100)}%\n currentScaling:{parameterSet['recurrentWeightScaling']}")
    saveData(multirunForce,'multirunReservoirData2')

multirunGoNoGoForceLearning(networkSizeArr=[500],feedbackWeightScalingArr=[0],recurrentWeightScalingArr=np.linspace(1.5,4,40),feedbackStrengthArr=[0])
#goNoGoForceLearning(recurrentWeightScaling=2.4)
import dill as pickle
import numpy as np
import matplotlib.pyplot as plt
from rateNetwork.dataHelperFunctions import loadData
from rateNetwork.dimensionGraphs import *
def getTrials(outputHistory,trialChoice):
    goTrials = [outputHistory[i] for i in range(len(trialChoice)) if trialChoice[i] == 0]
    noGoTrials = [outputHistory[i] for i in range(len(trialChoice)) if trialChoice[i] == 1]
    emptyTrials = [outputHistory[i] for i in range(len(trialChoice)) if trialChoice[i] == 2]
    goTrialData = np.array(goTrials).squeeze(axis=2)
    noGoTrialData = np.array(noGoTrials).squeeze(axis=2)
    emptyTrialData = np.array(emptyTrials).squeeze(axis=2)
    return goTrialData,noGoTrialData,emptyTrialData

def getAverageAndStd(data):
    dataAverage = np.average(data,axis=0)
    dataStd = np.std(data,axis=0)
    return dataAverage,dataStd

def plotAverage(xAxis,data,color='red',label=''):
    average,std = getAverageAndStd(data)
    plt.errorbar(xAxis,average,yerr=std,color=color,label=label)

def plotGoNoGoEmpty(goData,noGoData,emptyData,title,saveLoc=None):
    '''plots all on the same axis'''
    xAxis = range(goData.shape[1])
    plt.title(title)
    plotAverage(xAxis,goData,label='go trial',color='red')
    plotAverage(xAxis,noGoData,label='no-go trial',color='blue')
    plotAverage(xAxis,emptyData,label='empty trial',color='green')
    plt.xlabel('Time')
    plt.ylabel('Output Activation')
    plt.legend()
    if saveLoc == None:
        plt.show()
    else:
        plt.savefig(saveLoc)
    plt.cla()

def findFirstIndex(arr,value):
    for i in range(len(arr)):
        if arr[i] == value:
            return i

def getCorrectResponse(networkInputs,correctResponse,stimuliChoice):
    goTrialIndex = findFirstIndex(stimuliChoice,0)
    noGoTrialIndex = findFirstIndex(stimuliChoice,1)
    emptyTrialIndex = findFirstIndex(stimuliChoice,2)

    goTrialInput = networkInputs[goTrialIndex]
    noGoTrialInput = networkInputs[noGoTrialIndex]
    emptyTrialInput = networkInputs[emptyTrialIndex]

    goTrialCorrectResponse = correctResponse[goTrialIndex]
    noGoTrialCorrectResponse = correctResponse[noGoTrialIndex]
    emptyTrialCorrectResponse = correctResponse[emptyTrialIndex]
    return (goTrialInput,goTrialCorrectResponse),(noGoTrialInput,noGoTrialCorrectResponse),(emptyTrialInput,emptyTrialCorrectResponse)

def responseComparisonPlot(trialInput,trialCorrectOutput,data,title,
                           saveLoc=None,inputColor='orange',correctResponseColor='black',
                           responseColor='red',maxY=11,minY=-1):
    dataAv,dataStd = getAverageAndStd(data)
    plt.title(title)
    plt.plot(range(len(trialInput)),trialInput, color=inputColor, label='tone input')
    plt.plot(range(len(trialInput)), trialCorrectOutput,color=correctResponseColor,label='target response')
    plt.errorbar(range(len(trialInput)),dataAv,yerr=dataStd,color=responseColor,label='network response')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Output Activation')
    plt.ylim(minY,maxY)
    if saveLoc == None:
        plt.show()
    else:
        plt.savefig(saveLoc)
    plt.cla()

def plotSingleRun(goNoGoNetworkRun,saveFolder):
    testingOutputHistory = goNoGoNetworkRun['testingOutputHistory']
    testingTrialChoice = goNoGoNetworkRun['testingTrialChoice']
    testingTrialInputs = goNoGoNetworkRun['testingTrialInputs']
    testingCorrectResponses = goNoGoNetworkRun['testingCorrectResponses']
    goTestTrials,noGoTestTrials,emptyTestTrials = getTrials(testingOutputHistory,testingTrialChoice)
    goTrialIdeal,noGoTrialIdeal,emptyTrialIdeal = getCorrectResponse(testingTrialInputs,testingCorrectResponses,testingTrialChoice)
    responseComparisonPlot(*goTrialIdeal,goTestTrials,'Go Trial Average Output Response',saveLoc=f'./figs/{saveFolder}/go-output.png')
    responseComparisonPlot(*noGoTrialIdeal,noGoTestTrials,'No-go Trial Average Output Response',saveLoc=f'./figs/{saveFolder}/noGo-output.png')
    responseComparisonPlot(*emptyTrialIdeal,emptyTestTrials,'Empty Trial Average Output Response',saveLoc=f'./figs/{saveFolder}/empty-output.png')
    plotGoNoGoEmpty(goTestTrials,noGoTestTrials,emptyTestTrials,'Combined Average Response Across Stimuli',saveLoc=f'./figs/{saveFolder}/all-response.png')

def plotStimuliCorrect(goNoGoNetworkRun,saveFolder):
    testingTrialChoice = goNoGoNetworkRun['testingTrialChoice']
    testingTrialInputs = goNoGoNetworkRun['testingTrialInputs']
    testingCorrectResponses = goNoGoNetworkRun['testingCorrectResponses']
    goTrialIdeal,noGoTrialIdeal,emptyTrialIdeal = getCorrectResponse(testingTrialInputs,testingCorrectResponses,testingTrialChoice)
    plotCorrect(goTrialIdeal[0],goTrialIdeal[1],'Correct Stimulus Response For Go Trial','Go',saveLoc=f'./figs/{saveFolder}/go-correct.png')
    plotCorrect(noGoTrialIdeal[0],noGoTrialIdeal[1],'Correct Stimulus Response For No-Go Trial','No-Go',saveLoc=f'./figs/{saveFolder}/no-go-correct.png')
    plotCorrect(emptyTrialIdeal[0],emptyTrialIdeal[1],'Correct Stimulus Response For Empty Trial','Empty',saveLoc=f'./figs/{saveFolder}/empty-correct.png')

def plotCorrect(stimulus,idealResponse,title,trialType,saveLoc=None):
    time = range(len(stimulus))
    plt.plot(time,stimulus,label='Input Tone',color='orange')
    plt.plot(time, idealResponse,label=f'Correct {trialType} Response',color='black')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Response Activation')
    plt.legend()
    plt.ylim(-1,11)
    if saveLoc == None:
        plt.show()
    else:
        plt.savefig(saveLoc)
    plt.cla()

def knnAverage(xData,yData,window=8):
    sortedData = list(zip(*list(sorted(zip(xData,yData),key=lambda x:x[0]))))
    sortedX = sortedData[0]
    sortedY = sortedData[1]
    averagedList = []
    dataLength = len(sortedX)
    for i in range(dataLength):
        if i < window:
            averagedList.append(np.average(sortedY[:window]))
        elif i + window >= dataLength:
            averagedList.append(np.average(sortedY[dataLength-window:]))
        else:
            averagedList.append(np.average(sortedY[i-window:i+window+1]))
    return sortedX,sortedY,averagedList

def plotMultiRun(fileName,saveFolder):
    '''plots largest lyapunov exponent, max weight singular value and test loss against eachother'''
    multiRun = loadData(fileName)
    exponents = [run['largestLyapunovExponent'] for run in multiRun]
    losses = [run['testingAverageLoss'] for run in multiRun]
    scaling = [run['parameters']['recurrentWeightScaling'] for run in multiRun]
    extrinsicDimensions = [dims(getAllReservoirData(run))['fractional'] for run in multiRun]
    intrinsicDimensions = [dims(getAllReservoirData(run))['cutoff'] for run in multiRun]
    #plots the lyapunov exponent test error
    sortedExp,sortedLoss,averaged = knnAverage(exponents,losses,3)
    plt.title('Effect of Largest Lyapunov Exponent On Test Error')
    plt.scatter(sortedExp,sortedLoss,color='purple')
    plt.xlabel('Largest Lyapunov Exponent')
    plt.ylabel('Test Loss')
    plt.savefig(f'./figs/{saveFolder}/lle-vs-test.png')
    plt.cla()
    #plots the lyapunov exponent as a function of the weight scaling
    sortedScaling,sortedExponents,averaged = knnAverage(scaling,exponents,3)
    plt.title('Largest Lyapunov Exponent vs\n Max Recurrent Weight Singular Value')
    plt.scatter(sortedScaling,sortedExponents,color='purple')
    plt.xlabel('Max Recurrent Weight Singular Value')
    plt.ylabel('Largest Lyapunov Exponent')
    plt.savefig(f'./figs/{saveFolder}/singular-vs-lle.png')
    plt.cla()
    #plots the test error as a function of the weight scaling
    sortedScaling,sortedLosses,averaged = knnAverage(scaling,losses,3)
    plt.title('Effect of Weight Scaling on Test Loss')
    plt.scatter(sortedScaling,sortedLosses,color='purple')
    plt.xlabel('Max Recurrent Weight Singular Value')
    plt.ylabel('Test Loss')
    plt.savefig(f'./figs/{saveFolder}/singular-vs-loss.png')
    plt.cla()
    #plots the best run
    smallestLossIndex = np.argmin(np.array(losses))
    plotSingleRun(multiRun[smallestLossIndex],'forceSingleRun')
    #plots the correct response
    plotStimuliCorrect(multiRun[smallestLossIndex],'forceCorrect')
    #plots the extrinsic dimension as a function of the lyapunov exponent
    plt.title('Effect of Lyapunov Exponent on Effective Dimension')
    plt.scatter(exponents,extrinsicDimensions,color='purple')
    plt.xlabel('Largest Lyapunov Exponent')
    plt.ylabel('Effective Dimension')
    plt.savefig(f'./figs/{saveFolder}/extdim-vs-lle.png')
    plt.cla()
    #plots the intrinsic dimension as a function of the lyapunov exponent
    plt.title('Effect of Lyapunov Exponent on Intrinsic Dimension')
    plt.scatter(exponents,intrinsicDimensions,color='purple')
    plt.xlabel('Largest Lyapunov Exponent')
    plt.ylabel('Intrinsic Dimension')
    plt.savefig(f'./figs/{saveFolder}/intdim-vs-lle.png')
    plt.cla()

def plotCoordinateExp(coordinateExponents,largestExponent):
    plt.hist(coordinateExponents,density=True,bins=100)
    plt.vlines(largestExponent,0,1,color='red')
    plt.show()

def plotMeanSd(dataFrame):
    mean = dataFrame.mean()
    plt.plot(dataFrame.T,alpha=.1,color='red')
    plt.plot(mean,alpha=1,color='black')
    plt.show()

def plotCoordinateStability():
    multiRun = loadData('multirunReservoirData2')
    losses = [run['testingAverageLoss'] for run in multiRun]
    smallestLossIndex = np.argmin(np.array(losses))
    coordinateExponents = multiRun[smallestLossIndex]['coordinateExponent']
    stable = np.argmin(coordinateExponents)
    unstable = np.argmax(coordinateExponents)
    forceFeedback = multiRun[smallestLossIndex]['feedbackMatrix']
    absForceFeedback = np.abs(forceFeedback)
    cutoff = sorted(absForceFeedback)[int(len(forceFeedback)*.1)]
    filteredFeedback = [i for i in range(len(absForceFeedback)) if absForceFeedback[i] > cutoff]
    remainingFeedback = [i for i in range(len(absForceFeedback)) if absForceFeedback[i] <= cutoff]
    filteredCoordinateExp = [coordinateExponents[i] for i in filteredFeedback]
    remainingCoordinateExp = [coordinateExponents[i] for i in remainingFeedback]
    plotCoordinateExp(coordinateExponents,multiRun[smallestLossIndex]['largestLyapunovExponent'])
    print(len(filteredCoordinateExp))
    plt.scatter(filteredCoordinateExp,np.random.normal(0,1,size=len(filteredCoordinateExp)),color='red',alpha=1)
    plt.scatter(remainingCoordinateExp,np.random.normal(0,1,size=len(remainingCoordinateExp)),color='yellow',alpha=1)
    plt.show()
    goData, noGoData,emtpyData = singleNeuronData(multiRun[smallestLossIndex],filteredFeedback[0])
    plotMeanSd(goData)
    goData, noGoData,emtpyData = singleNeuronData(multiRun[smallestLossIndex],filteredFeedback[1])
    print(multiRun[smallestLossIndex]['largestLyapunovExponent'])
    plotMeanSd(goData)

#plotMeanSd(pd.DataFrame(singleEmpty))
#multiRun = loadData('multirunReservoirData2')[0]
#coordinateExponents = multiRun['coordinateExponent']
#largestExponent = multiRun['largestLyapunovExponent']

#plotSingleRun(loadData('forceData'),'forceSingleRun')
#plotMultiRun('multirunReservoirData2','forceMultiRun')
def plotLyapunovDimension():
    data = loadData('multirunReservoirData2')
    lyapunovDims = [data[i]['lyapunovDimension'] for i in range(len(data))]
    lyapunovExp = [data[i]['largestLyapunovExponentOther'] for i in range(len(data))]
    plt.title('Increasing the Largest Lyapunov Exponent\n Increases the Lyapunov Dimension')
    plt.scatter(lyapunovExp,lyapunovDims)
    plt.xlabel('Largest Lyapunov Exponent')
    plt.ylabel('Lyapunov Dimension')
    plt.show()

#getReservoirData(loadData('forceData'))
plotCoordinateStability()
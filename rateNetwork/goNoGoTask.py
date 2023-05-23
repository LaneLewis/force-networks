import numpy as np
def goNoGo(trials:int,goToneAmount:float,noGoToneAmount:float,
           toneLength:int,gapLength:int,correctOutputLength:int, afterLength:int,
           goCorrectOutputAmount:float,goProbability=.3,noGoProbability=.3,
           emptyTrialProbability=.4):
    '''Constructs the go-nogo paradigm. When a go trial is given, the 
    correct response is to wait some gap of time after hearing the tone 
    and then produce a constant response for some amount of time. On a
    no-go trial the correct response is to output 0 for the entire trial
    output'''
    #Arrays of matricies specifying the input signals into the network 
    #(matricies to handle a potentially multidimensional input signal
    # in the general rnn implementation)
    trialInputMatrixArr = []
    trialCorrectResponseMatrixArr = []
    #specifies whether trial is a go trial or not
    trialChoice = np.random.choice([0,1,2],p=[goProbability,noGoProbability,emptyTrialProbability],size=trials)
    totalTrialLength = toneLength+gapLength+correctOutputLength+afterLength
    #constructs the go trial
    goTrialInput = np.array([*[goToneAmount]*toneLength,*[0]*(gapLength+correctOutputLength+afterLength)]).reshape(totalTrialLength,1)
    goTrialCorrectResponse = np.array([*[0]*(toneLength+gapLength),*[goCorrectOutputAmount]*correctOutputLength,*[0]*afterLength]).reshape(totalTrialLength,1)
    #constructs the no-go trial 
    noGoTrialInput = np.array([*[noGoToneAmount]*toneLength,*[0]*(gapLength+correctOutputLength+afterLength)]).reshape(totalTrialLength,1)
    noGoTrialCorrectResponse = np.array([*[0]*(toneLength+gapLength),*[0]*(correctOutputLength+afterLength)]).reshape(totalTrialLength,1)
    #constructs the empty trial
    emptyTrialInput = np.zeros((totalTrialLength,1))
    emptyTrialCorrectResponse = np.zeros((totalTrialLength,1))
    for trial in trialChoice:
        if trial == 0:
            #trial is go trial
            trialInputMatrixArr.append(goTrialInput)
            trialCorrectResponseMatrixArr.append(goTrialCorrectResponse)
        if trial == 1:
            #trial is a no-go
            trialInputMatrixArr.append(noGoTrialInput)
            trialCorrectResponseMatrixArr.append(noGoTrialCorrectResponse)
        if trial == 2:
            #trial is empty
            trialInputMatrixArr.append(emptyTrialInput)
            trialCorrectResponseMatrixArr.append(emptyTrialCorrectResponse)
    return trialInputMatrixArr,trialCorrectResponseMatrixArr,trialChoice

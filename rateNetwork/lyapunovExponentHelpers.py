# This module sits as a small wrapper on top of the lyapunovExponents module
# and allows for the lyapunov exponents, coordinate stability, and lyapunov dimension
# to be easily calculated for a force network
import numpy as np
import lyapunovExponents
class forceIterativeMap():
    '''Used to build a simplified version of the force prediction function where the input
    is kept constant over time and the force feedback is calculated internally. This allows
    for '''
    '''Constructs a mapping that only relies on the previous point. The force feedback is
    tracked internally and can be kept from updating by setting updateMap to false'''
    def __init__(self,predictionFunction,networkSize,externalInput,withForceOn=True):
        self.forceFeedback = np.zeros((networkSize,1))
        self.externalInput = externalInput
        self.predictionFunction = predictionFunction
        self.withForceOn = withForceOn
    def iterativeMap(self,point,updateMap=True):
        '''if hold map is true, force feedback doesnt change'''
        if self.withForceOn:
            nextPoint,_,newForceFeedback = self.predictionFunction(point,self.forceFeedback,self.externalInput)
            if updateMap:
                self.forceFeedback = newForceFeedback
        else:
            nextPoint,_,_=self.predictionFunction(point,self.forceFeedback,self.externalInput)
        return nextPoint

def forceMap(predictionFunction, networkSize,constantExternalInput,withForceOn=True):
    forceFeedback = np.zeros((networkSize,1))
    def forceIterativeMap(point,updateMap=True):
        if withForceOn:
            nextPoint,_,newForceFeedback = predictionFunction(point,forceFeedback,constantExternalInput)
            if updateMap:
                forceFeedback = newForceFeedback
        else:
            nextPoint,_,_ = predictionFunction(point,forceFeedback,constantExternalInput)
        return nextPoint
    return forceIterativeMap

def calculateForceLargestLyapunovExponent(predictionFunction,networkSize,constantExternalInput,initialState,
                                       withForceOn=True,iterations=2000,progressBar=True):
    forceIterMap = forceMap(predictionFunction,networkSize,constantExternalInput,withForceOn=withForceOn)
    stability,coordinateStability = lyapunovExponents.calculateLargestLyapunovExponent(forceIterMap,initialState,.0000001*np.ones((networkSize,1)),iterations,progressBar=progressBar)
    return stability,coordinateStability

def calculateForceLyapunovDimension(predictionFunction,networkSize,constantExternalInput,initialState,
                                                withForceOn=True,iterations=2000):
    forceIterMap = forceMap(predictionFunction,networkSize,constantExternalInput,withForceOn=withForceOn)
    exponents = lyapunovExponents.calculateAllLyapunovExponents(forceIterMap,initialState,iterations,loadBarDesc="forceLLE")
    lyapunovDimension = lyapunovExponents.calculateLyapunovDimension(exponents)
    return lyapunovDimension

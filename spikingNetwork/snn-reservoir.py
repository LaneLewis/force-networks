import numpy as np
import dill as pickle

class LIFSpikingReservoir:
    def __init__(self,excitatoryToExcitatoryWeightMatrix:np.ndarray,excitatoryToInhibitoryWeightMatrix:np.ndarray,
                 inhibitoryToInhibitoryWeightMatrix:np.ndarray,inhibitoryToExcitatoryWeightMatrix:np.ndarray,
                 initialExcitatoryVoltage:np.ndarray,initialInhibitoryVoltage:np.ndarray,
                 excitatoryBiasCurrent:np.ndarray,inhibitoryBiasCurrent:np.ndarray,
                 excitatoryOutputMask:np.ndarray,inhibitoryOutputMask:np.ndarray,
                 inhibitorySynapticTimeConstant=20,excitatorySynapticTimeConstant:float=20,membraneTimeConstant:float=20,
                 outputTimeConstant=20,restVoltage:float =-65,spikeVoltageThreshold:float=-55,
                 eulerTimeDelta:float=.05,eulerTimeSteps:float=50000,
                 )->None:
        '''uses an exponential filter'''
        #reservoir network parameters
        self.excitatoryToExcitatoryWeightMatrix = excitatoryToExcitatoryWeightMatrix
        self.excitatoryToInhibitoryWeightMatrix = excitatoryToInhibitoryWeightMatrix
        self.inhibitoryToInhibitoryWeightMatrix = inhibitoryToInhibitoryWeightMatrix
        self.inhibitoryToExcitatoryWeightMatrix = inhibitoryToExcitatoryWeightMatrix
        self.initialExcitatoryVoltage = initialExcitatoryVoltage
        self.initialInhibitoryVoltage = initialInhibitoryVoltage
        self.excitatoryBiasCurrent = excitatoryBiasCurrent
        self.inhibitoryBiasCurrent = inhibitoryBiasCurrent
        self.inhibitorySynapticTimeConstant = inhibitorySynapticTimeConstant
        self.excitatorySynapticTimeConstant = excitatorySynapticTimeConstant
        self.spikeVoltageThreshold = spikeVoltageThreshold
        self.membraneTimeConstant = membraneTimeConstant
        self.excitatoryNeuronNumber = len(initialExcitatoryVoltage)
        self.inhibitoryNeuronNumber = len(initialInhibitoryVoltage)
        self.inhibitoryRestVoltage = restVoltage*np.ones((self.inhibitoryNeuronNumber,1))
        self.excitatoryRestVoltage = restVoltage*np.ones((self.excitatoryNeuronNumber,1))
        self.excitatoryOutputMask = excitatoryOutputMask
        self.inhibitoryOutputMask = inhibitoryOutputMask
        self.outputTimeConstant = outputTimeConstant
        self.restVoltage = restVoltage
        #euler method parameters
        self.time=0
        self.eulerTimeDelta = eulerTimeDelta
        self.eulerTimeSteps = eulerTimeSteps

    def nextTimeStep(self,excitatoryInputCurrent:np.ndarray,inhibitoryInputCurrent:np.ndarray,excitatoryVoltageState:np.ndarray,inhibitoryVoltageState:np.ndarray,
                     excitatoryExcitatorySynapticState:np.ndarray,excitatoryInhibitorySynapticState:np.ndarray,
                     inhibitoryInhibitorySynapticState:np.ndarray, inhibitoryExcitatorySynapticState:np.ndarray,
                     excitatoryOutputState:np.ndarray,inhibitoryOutputState:np.ndarray):
        '''General Description:
            Performs the euler voltage update step on the network to approximately solve the LIF network.
            differential equation. The synaptic coupled current state for both inhibitory and excitatory currents
            is tracked for the populations of excitatory and inhibitory neurons over the time steps as well to 
            reduce the computational and memory load of the network. This is possible because of the summation carried 
            out by the exponential filter.

        '''
        excitatorySpikeVector,newExcitatoryVoltageState = updateVoltageEuler(excitatoryVoltageState,
                                                                             self.excitatoryRestVoltage,
                                                                             self.spikeVoltageThreshold,
                                                                             excitatoryExcitatorySynapticState,
                                                                             inhibitoryExcitatorySynapticState,
                                                                            self.excitatoryBiasCurrent,
                                                                            excitatoryInputCurrent,
                                                                            self.eulerTimeDelta,
                                                                            self.membraneTimeConstant,
                                                                            )
        inhibitorySpikeVector,newInhibitoryVoltageState = updateVoltageEuler(inhibitoryVoltageState,
                                                                             self.inhibitoryRestVoltage,
                                                                             self.spikeVoltageThreshold,
                                                                             excitatoryInhibitorySynapticState,
                                                                             inhibitoryInhibitorySynapticState,
                                                                             self.inhibitoryBiasCurrent,
                                                                             inhibitoryInputCurrent,
                                                                             self.eulerTimeDelta,
                                                                             self.membraneTimeConstant,
                                                                             )
        newInhibitoryInhibitorySynapticState,newInhibitoryExcitatorySynapticState = updateInhibitorySynapticStateEuler(
                                                                                inhibitoryInhibitorySynapticState,
                                                                                inhibitoryExcitatorySynapticState,
                                                                                self.inhibitoryToInhibitoryWeightMatrix,
                                                                                self.inhibitoryToExcitatoryWeightMatrix,
                                                                                inhibitorySpikeVector,
                                                                                self.inhibitorySynapticTimeConstant,
                                                                                self.eulerTimeDelta)
        newExcitatoryExcitatorySynapticState,newExcitatoryInhibitorySynapticState = updateExcitatorySynapticStateEuler(
                                                                                excitatoryExcitatorySynapticState,
                                                                                excitatoryInhibitorySynapticState,
                                                                                self.excitatoryToExcitatoryWeightMatrix,
                                                                                self.excitatoryToInhibitoryWeightMatrix,
                                                                                excitatorySpikeVector,
                                                                                self.excitatorySynapticTimeConstant,
                                                                                self.eulerTimeDelta)
        newExcitatoryOutputState,newInhibitoryOutputState = updateOutputStateEuler(excitatoryOutputState,
                                                                inhibitoryOutputState,
                                                                excitatorySpikeVector,inhibitorySpikeVector,
                                                                self.excitatoryOutputMask,self.inhibitoryOutputMask,
                                                                self.outputTimeConstant,self.eulerTimeDelta)
        return (newExcitatoryVoltageState,newInhibitoryVoltageState,
                newInhibitoryInhibitorySynapticState,newInhibitoryExcitatorySynapticState,
                newExcitatoryExcitatorySynapticState,newExcitatoryInhibitorySynapticState,
                excitatorySpikeVector,inhibitorySpikeVector,
                newExcitatoryOutputState,newInhibitoryOutputState)
    
    def train(self):
        inhibitoryVoltageState = self.initialInhibitoryVoltage
        excitatoryVoltageState = self.initialExcitatoryVoltage
        #initializes all synapse variables to 0
        inhibitoryInhibitorySynapticState = np.zeros((self.inhibitoryNeuronNumber,1))
        excitatoryInhibitorySynapticState = np.zeros((self.inhibitoryNeuronNumber,1))
        excitatoryExcitatorySynapticState = np.zeros((self.excitatoryNeuronNumber,1))
        inhibitoryExcitatorySynapticState = np.zeros((self.excitatoryNeuronNumber,1))
        #output
        excitatoryOutputState = np.zeros((self.excitatoryNeuronNumber,1))
        inhibitoryOutputState = np.zeros((self.inhibitoryNeuronNumber,1))
        #set to no input current for testing
        excitatoryInputCurrent = np.zeros((self.excitatoryNeuronNumber,1))#np.random.binomial(1,.5,(self.excitatoryNeuronNumber,1))
        inhibitoryInputCurrent = np.zeros((self.inhibitoryNeuronNumber,1))
        #set loop variables
        simExcitatoryVoltageStates = np.zeros((self.excitatoryNeuronNumber,self.eulerTimeSteps))
        simInhibitoryVoltageStates = np.zeros((self.inhibitoryNeuronNumber,self.eulerTimeSteps))
        simExcitatorySpikes = np.zeros((self.excitatoryNeuronNumber,self.eulerTimeSteps))
        simInhibitorySpikes = np.zeros((self.inhibitoryNeuronNumber,self.eulerTimeSteps))
        #output state inititation
        excitatoryOutputMaskSize = int(self.excitatoryOutputMask.sum())
        inhibitoryOutputMaskSize = int(self.inhibitoryOutputMask.sum())
        simExcitatoryOutputStates = np.zeros((excitatoryOutputMaskSize,self.eulerTimeSteps))
        simInhibitoryOutputStates = np.zeros((inhibitoryOutputMaskSize,self.eulerTimeSteps))
        for timeIndex in range(self.eulerTimeSteps):
            eulerUpdate = self.nextTimeStep(excitatoryInputCurrent,inhibitoryInputCurrent,
                              excitatoryVoltageState,inhibitoryVoltageState,excitatoryExcitatorySynapticState,
                              excitatoryInhibitorySynapticState,inhibitoryInhibitorySynapticState,
                              inhibitoryExcitatorySynapticState,excitatoryOutputState,inhibitoryOutputState)
            #set for next loop
            excitatoryVoltageState,inhibitoryVoltageState = eulerUpdate[0],eulerUpdate[1]
            inhibitoryInhibitorySynapticState,inhibitoryExcitatorySynapticState = eulerUpdate[2],eulerUpdate[3]
            excitatoryExcitatorySynapticState, excitatoryInhibitorySynapticState = eulerUpdate[4],eulerUpdate[5]
            excitatoryOutputState,inhibitoryOutputState = eulerUpdate[8],eulerUpdate[9]
            #add to simulation variables
            simExcitatorySpikes[:,timeIndex] = eulerUpdate[6].reshape(self.excitatoryNeuronNumber)
            simInhibitorySpikes[:,timeIndex] = eulerUpdate[7].reshape(self.inhibitoryNeuronNumber)
            simExcitatoryOutputStates[:,timeIndex] = excitatoryOutputState.reshape(len(excitatoryOutputState))
            simInhibitoryOutputStates[:,timeIndex] = inhibitoryOutputState.reshape(len(inhibitoryOutputState))
            simExcitatoryVoltageStates[:,timeIndex] = excitatoryVoltageState.reshape(self.excitatoryNeuronNumber)
            simInhibitoryVoltageStates[:,timeIndex] = inhibitoryVoltageState.reshape(self.inhibitoryNeuronNumber)
        return {'simExcitatoryVoltageStates':simExcitatoryVoltageStates,'simInhibitoryVoltageStates':simInhibitoryVoltageStates,
                'simExcitatorySpikes':simExcitatorySpikes,'simInhibitorySpikes':simInhibitorySpikes,
                'time':self.eulerTimeDelta*np.arange(self.eulerTimeSteps),'simExcitatoryOutputStates':simExcitatoryOutputStates,
                'simInhibitoryOutputStates':simInhibitoryOutputStates}

        
            
def updateVoltageEuler(voltageState,restVoltage, spikeThreshold,
                  excitatorySynapticState,inhibitorySynapticState,
                  restCurrent,inputCurrent,eulerTimeDelta,membraneTimeConstant)->tuple[np.array,np.array]:
    ''' approximates the LIF differential equation using eulers method with step size eulerTimeStep: 
    dV_i/dt = (voltageState - restVoltage + excitatorySynapticState - inhibitorySynapticState + restCurrent + inputCurrent)/membraneTimeConstant
    V_i > spikeThreshold = 0
    Where V_i > spikeThreshold gives a spike event

    returns an updated voltage as well as a vector of the neurons that spiked. The spiked vector
    has ones at the indexes where the neuron spiked and zeros elsewhere'''

    LIFDiffEq = (-voltageState + restVoltage + excitatorySynapticState - inhibitorySynapticState + restCurrent + inputCurrent)/membraneTimeConstant
    #overshoots the spike
    nextVoltageState = voltageState + LIFDiffEq*eulerTimeDelta
    neuronDidSpikeBool = (nextVoltageState > spikeThreshold).astype(int)
    #corrects the overshoot
    nextVoltageState = nextVoltageState*(1-neuronDidSpikeBool) + neuronDidSpikeBool*restVoltage
    spikedSelectionVector = neuronDidSpikeBool
    return spikedSelectionVector, nextVoltageState

def updateInhibitorySynapticStateEuler(inhibitoryInhibitorySynapticState:np.ndarray,inhibitoryExcitatorySynapticState:np.ndarray,
                                       inhibitoryToInhibitoryWeightMatrix:np.ndarray,inhibitoryToExcitatoryWeightMatrix:np.ndarray,
                                       inhibitorySpikeVector:np.ndarray,inhibitorySynapticTimeConstant:float,timeDelta:float) -> tuple[np.ndarray,np.ndarray]:
    '''
    General Description:
    Updates the inhibitory current stemming from inhibitory neurons synapsing onto excitatory and inhibitory neurons.
    The inhibitory current is a weighted exponential filter of spikes from the inhibitory population of neurons.

    Mathematical Motivation:
    Given a spikeVector evalutated at t_s (zeros in all spots except the indexes where a spike is emmited) = S(t_s)
    Where the inhibitory weight watrix at each spike event,t_s, is given by W_I(t_s) where W_I(t_s)
    is the block matrix ->  | inhibitoryToInhibitoryWeightMatrix |
                            | inhibitoryToExcitatoryWeightMatrix |
    Then the inhibitorySynapticState, I_I(t) = {\Sum_{t_s<t} W_I(t_s)S(t_s)e^{(t_s-t)/\tau}
    If t_s occurs at the same time as the time step t_{s-1}+timeDelta (as in the euler approximation):
    I_I(t_s) = e^{-timeDelta/\tau_I}*I_E(t_{s-1}) + W_I(t_s)S(t_s)
    Then I_I(t_s) is the block vector -> | newInhibitoryInhibitorySynapticState |
                                         | newInhibitoryExcitatorySynapticState |
    let N be the number of neurons in the network, 
    N_I be the number of inhibitory neurons,
    N_E be the number of excitatory neurons:

    Parameters:
    inhibitoryInhibitorySynapticState - \mathbb{R}^{N_I \times 1} ~ inhibitory current inside of inhibitory neurons
    inhibitoryExcitatorySynapticState - \mathbb{R}^{N_E \times 1} ~ inhibitory current inside of excitatory neurons
    inhibitoryToInhibitoryWeightMatrix - \mathbb{R}^{N_I \times N_I} ~ synapse weight matrix from inhibitory neurons to inhibitory neurons
    inhibitoryToExcitatoryWeightMatrix - \mathbb{R}^{N_E \times N_I} ~ synapse weight matrix from inhibitory neurons to excitatory neurons
    inhibitorySpikeVector - {0,1}^{N_I \times 1} ~ spike vector with ones indicating inhibitory neurons that spiked in the last time step and zeros for non-spiked
    timeDelta - \mathbb{R} ~ euler time step
    Outputs:
    newInhibitoryInhibitorySynapticState - \mathbb{R}^{N_I \times 1} ~ new inhibitory current inside of inhibitory neurons
    newInhibitoryExcitatorySynapticState - \mathbb{R}^{N_E \times 1} ~ new inhibitory current inside of excitatory neurons
    '''
    newInhibitoryInhibitorySynapticState= np.exp(-1*timeDelta/inhibitorySynapticTimeConstant)*inhibitoryInhibitorySynapticState+ np.matmul(inhibitoryToInhibitoryWeightMatrix,inhibitorySpikeVector)
    newInhibitoryExcitatorySynapticState = np.exp(-1*timeDelta/inhibitorySynapticTimeConstant)*inhibitoryExcitatorySynapticState+ np.matmul(inhibitoryToExcitatoryWeightMatrix,inhibitorySpikeVector)
    return newInhibitoryInhibitorySynapticState,newInhibitoryExcitatorySynapticState

def updateExcitatorySynapticStateEuler(excitatoryExcitatorySynapticState:np.ndarray,excitatoryInhibitorySynapticState:np.ndarray,
                                  excitatoryToExcitatoryWeightMatrix:np.ndarray,excitatoryToInhibitoryWeightMatrix:np.ndarray,
                                  excitatorySpikeVector:np.ndarray,inhibitorySynapticTimeConstant:float,timeDelta:float)->np.ndarray:
    '''General Description:
    Updates the excitatory current stemming from excitatory neurons synapsing onto excitatory and inhibitory neurons.
    The excitatory current is a weighted exponential filter of spikes from the excitatory population of neurons.

    Mathematical Motivation:
    Given an excitatorySpikeVector evalutated at t_s (zeros in all spots except the indexes where a spike is emmited) = S_E(t_s)
    Where the excitatory weight watrix at each spike event,t_s, is given by W_E(t_s) where W_E(t_s)
    is the block matrix ->  | excitatoryToExcitatoryWeightMatrix |
                            | excitatoryToInhibitoryWeightMatrix |
    Then the inhibitorySynapticState, I_E(t) = {\Sum_{t_s<t} W_E(t_s)S_E(t_s)e^{t_s-t}
    If t_s occurs at the same time as the time step t_{s-1}+timeDelta (as in the euler approximation):
    I_E(t_s) = e^{-timeDelta/tau_E}*I_E(t_{s-1}) + W_I(t_s)S(t_s)
    Then I_E(t_s) is the block vector -> | newExcitatoryExcitatorySynapticState |
                                         | newExcitatoryInhibitorySynapticState |

    let N be the number of neurons in the network, 
    N_I be the number of inhibitory neurons,
    N_E be the number of excitatory neurons

    Parameters:
    excitatoryExcitatorySynapticState - \mathbb{R}^{N_E \times 1} ~ excitatory current inside of excitatory neurons
    excitatoryInhibitorySynapticState - \mathbb{R}^{N_I \times 1} ~ excitatory current inside of inhibitory neurons
    excitatoryToExcitatoryWeightMatrix - \mathbb{R}^{N_E \times N_E} ~ synapse weight matrix from excitatory neurons to excitatory neurons
    excitatoryToInhibitoryWeightMatrix - \mathbb{R}^{N_I \times N_E} ~ synapse weight matrix from excitatory neurons to inhibitory neurons
    excitatorySpikeVector - {0,1}^{N_E \times 1} ~ spike vector with ones indicating inhibitory neurons that spiked in the last time step and zeros for non-spiked
    timeDelta - \mathbb{R} ~ euler time step
    Outputs:
    newExcitatoryExcitatorySynapticState - \mathbb{R}^{N_E \times 1} ~ new excitatory current inside of excitatory neurons
    newExcitatoryInhibitorySynapticState - \mathbb{R}^{N_I \times 1} ~ new excitatory current inside of inhibitory neurons
    '''
    newExcitatoryExcitatorySynapticState= np.exp(-1*timeDelta/inhibitorySynapticTimeConstant)*excitatoryExcitatorySynapticState + np.matmul(excitatoryToExcitatoryWeightMatrix,excitatorySpikeVector)
    newExcitatoryInhibitorySynapticState = np.exp(-1*timeDelta/inhibitorySynapticTimeConstant)*excitatoryInhibitorySynapticState + np.matmul(excitatoryToInhibitoryWeightMatrix,excitatorySpikeVector)
    return newExcitatoryExcitatorySynapticState,newExcitatoryInhibitorySynapticState
def updateOutputStateEuler(excitatoryOutputState:np.ndarray,inhibitoryOutputState:np.ndarray,
                excitatorySpikedVector:np.ndarray,inhibitorySpikedVector:np.ndarray,
                excitatoryToOutputMask:np.ndarray,inhibitoryToOutputMask:np.ndarray,
                outputTimeConstant:np.ndarray,timeDelta:float):
    '''implements an exponential filter over the spiking activity of neurons'''
    newExcitatoryOuputState = np.exp(-1*timeDelta/outputTimeConstant)*excitatoryOutputState + np.matmul(excitatoryToOutputMask,excitatorySpikedVector)
    newInhibitoryOutputState = np.exp(-1*timeDelta/outputTimeConstant)*inhibitoryOutputState + np.matmul(inhibitoryToOutputMask,inhibitorySpikedVector)
    return newExcitatoryOuputState,newInhibitoryOutputState

def buildUniformConnectionMatrix(connectionProbability,inputNeuronNumber,outputNeuronNumber,minWeight=0,maxWeight=1):
    '''inefficient'''
    fullConnectionMatrix = np.random.uniform(minWeight,maxWeight,(outputNeuronNumber,inputNeuronNumber))
    pruningMatrix = np.random.binomial(1,connectionProbability,(outputNeuronNumber,inputNeuronNumber))
    return fullConnectionMatrix*pruningMatrix

def runExperiment(EEConnectionProbability:float,EIConnectionProbability:float,IIConnectionProbability:float,IEConnectionProbability:float,
                  INeuronFraction:int,totalNeurons=1000,EEMax=11.9,EIMax=21,IEMax=30,IIMax=4,eulerStep=.05,eulerTimeSteps=50000):
    INeuronNumber = round(totalNeurons*INeuronFraction)
    ENeuronNumber = totalNeurons - INeuronNumber
    EEWeightMatrix = buildUniformConnectionMatrix(EEConnectionProbability,ENeuronNumber,ENeuronNumber,maxWeight=EEMax)
    EIWeightMatrix = buildUniformConnectionMatrix(EIConnectionProbability,ENeuronNumber,INeuronNumber,maxWeight=EIMax)
    IIWeightMatrix= buildUniformConnectionMatrix(IIConnectionProbability,INeuronNumber,INeuronNumber,maxWeight=IIMax)
    IEWeightMatrix = buildUniformConnectionMatrix(IEConnectionProbability,INeuronNumber,ENeuronNumber,maxWeight=IEMax)
    EStartingVoltage = np.random.uniform(-65,-55,(ENeuronNumber,1))
    IStartingVoltage = np.random.uniform(-65,-55,(INeuronNumber,1))
    IBiasCurrent = np.zeros((INeuronNumber,1))
    EBiasCurrent = np.zeros((ENeuronNumber,1))
    EBiasCurrent[0] = 12
    EBiasCurrent[1] = 15
    excitatoryOutputMask = np.identity(ENeuronNumber)
    inhibitoryOutputMask = np.identity(INeuronNumber)
    reservoir = LIFSpikingReservoir(EEWeightMatrix,EIWeightMatrix,
                                    IIWeightMatrix,IEWeightMatrix,
                                    EStartingVoltage,IStartingVoltage,
                                    EBiasCurrent,IBiasCurrent,
                                    excitatoryOutputMask,inhibitoryOutputMask,
                                    eulerTimeSteps=eulerTimeSteps,
                                    eulerTimeDelta=eulerStep)
    trainDict = reservoir.train()
    pickle.dump({'model':reservoir,'trainDict':trainDict},open("./rnnTestbed/reservoirData.pkl",'wb'))
    return trainDict

trainDict = runExperiment(.05,.05,.05,.05,.2)

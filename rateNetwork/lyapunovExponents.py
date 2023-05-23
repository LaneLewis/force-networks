# This module contains functions for calculating the largest lyapunov exponents and
# the lyapunov dimension.
# exports: calculateApproximateJacobian, calculateLargestLyapunovExponent,calculateLyapunovDimension
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from typing import Callable
def calculateApproximateJacobian(iterativeMap:Callable[[np.ndarray],np.ndarray],point:np.ndarray,
                            approxWidth=.00001)->np.ndarray:
    '''
    parameters:
        iterativeMap- function from R^n -> R^n. This map must have the optional parameter updateMap
            specifying whether to update the internal states of the map when passing in a value.
        point - point in R^n. The point to calculate the jacobian around
    optional:
        approxWidth - the width to use when approximating the right partial derivative using
            the perturbation secant.
    output: jacobianMatrix
        jacobianMatrix - the approximate R^n x R^n jacobian matrix at the evaluation point. 
    description: Approximates the jacobian of a map at a point. Uses the right
        secant vector under a small pertubation. So, more accurately, this
        function approximates the right partial dertivatives.
    '''
    nextPoint = iterativeMap(point,updateMap=False)
    standardBasisVectors = np.identity(len(point))
    imageDimension = len(nextPoint)
    jacobian = np.zeros(shape=(imageDimension,imageDimension))
    for i in range(imageDimension):
        basis_i = np.expand_dims(standardBasisVectors[:,i],axis=1)
        jacobianVector = (iterativeMap(point+basis_i*approxWidth,updateMap=False) - nextPoint)/approxWidth
        jacobian[i,:] = np.squeeze(jacobianVector)
    return jacobian

def calculateLargestLyapunovExponent(iterativeMap:Callable[[np.ndarray],np.ndarray],initialPoint:np.ndarray,
                    pertubation:np.ndarray,iterations:int,progressBarDesc='LLE+CLLE',progressBar=True)->tuple[float,np.ndarray]:
    '''
    parameters:
        iterativeMap - function from R^n -> R^n. This map must have the optional parameter updateMap
            specifying whether to update the internal states of the map when passing in a value
        initialPoint - point in R^n. Specifies the initial condition to calculate the lyapunov exponents around
        pertubation - vector in R^n. Specifies the initial direction of the pertubation to apply to the system.
            The magnitude of the pertubation should be very small i.e. 1 x 10^(-12)
        iterations - larger than zero int. Specifies the number of updates to perform when calculating the lyapunov
                    exponents around. Somewhere around two thousand time steps should be sufficient.
    optional:
        progressBarDesc - description to give the progress bar
        progressBar - whether a progress bar should be displayed

    output: tuple(largestLyapunovExponent,coordinateStability)
        largestLyapunovExponent - approximate largest lyapunov exponent of the iterative map
        coodinateStability - array of the stability of the individual coordinates

    description:
        Calculates the largest lyapunov exponent of the iterative map around the initial point. Much faster
         than calcluateAllLyapunovExponents so us this is only the largest exponent is needed. Uses
        the fact that in a linearizeable system with initial conditions:
        \delta_0e^{\lambda*t} \approx \delta_t
        where \lambda is the lyapunov exponent. This algorithm calculates the largest lyapunov exponent using
        the same algorithm as in (Boedecker, Joschka, et al.) which uses a rescaling and everaging step
        to avoid numerical issues.
        The coordinate stability is calculated using the formula C_i = ln(|delta_ti|/|delta_0i|)/t. The
        use of coordinate stability is documented in this project.
        Note: I havent verified that the recaling done in this algorithm preserves the C_i, so use the 
        coordinate stability here with caution. However, I believe it should hold. 
        Boedecker, Joschka, et al. "Information processing in echo state networks at the edge of chaos."
        Theory in Biosciences 131 (2012): 205-213.

    '''
    #instantiates an identical map to iterate the pertubation over. This handles the 
    #case where an iterative map has internal states that are modified when passing 
    #in a point which we don't want shared between the perturbed and unperturbed map.
    # By deep copying the map, we ensure that both maps have seperated internal states
    perturbedIterativeMap = deepcopy(iterativeMap)
    #initializes the unperturbed trajectory
    nextNonPerturbedPoint = iterativeMap(initialPoint,updateMap=True)
    #initializes the perturbed trajectory
    nextPerturbedPoint = perturbedIterativeMap(initialPoint+pertubation,updateMap=True)
    #list of the magnitude of the difference vector between the perturbed and unperturbed
    #trajectory across time
    pertubationList = []
    #list of all difference vectors between the perturbed trajectory and the unperturbed
    #trajectory across time
    pertubationVectorList = []
    #produces the progress bar
    if progressBar:
        indexIterator = tqdm(range(iterations - 1),desc=progressBarDesc)
    else:
        indexIterator = range(iterations-1)
    #Uses a rescaling technique to compute the lle. Takes the normed difference between the perturbed and
    #unperturbed trajectory and divides the original pertubation norm by this difference. This is then 
    #used to rescale the next location of the perturbed point to make sure it doesn't move away
    #exponentially from the unperturbed trajectory over time.
    for _ in indexIterator:
        pertubationVectorList.append(nextPerturbedPoint-nextNonPerturbedPoint)
        pertubationList.append(np.linalg.norm(nextPerturbedPoint-nextNonPerturbedPoint))
        errorRatio = np.linalg.norm(pertubation)/np.linalg.norm(nextPerturbedPoint-nextNonPerturbedPoint)
        nextPerturbedPoint = nextNonPerturbedPoint + errorRatio*(nextPerturbedPoint - nextNonPerturbedPoint)
        nextNonPerturbedPoint = iterativeMap(nextNonPerturbedPoint,updateMap=True)
        nextPerturbedPoint = perturbedIterativeMap(nextPerturbedPoint,updateMap=True)
    
    lle = np.average(np.log(np.array(pertubationList)/np.linalg.norm(pertubation)))
    coordinateWise = np.average([np.log(np.sqrt((np.square(x)/np.square(pertubation)))) for x in pertubationVectorList],axis=0)
    return lle,coordinateWise

def calculateLyapunovDimension(lyapunovSpectrum:np.ndarray) -> float:
    '''
    parameters:
        lyapunovSpectrum - all n lyapunov exponents of a system
    outputs: lyapunovDimension
        lyapunovDimension - the lyapunov dimension
    description:
        Calculates the lyapunov dimension (also known as Kaplan-Yorke dimension) of the 
        lyapunov spectrum. The spectrum is given by:

        Let k = max i s.t. \sum(\lambda_0,...\lambda_i)>0
        D_KY = k + [\sum^k_i(\lambda_i)]/\lambda_{k+1}

        Frederickson, Paul, et al. "The Liapunov dimension of strange attractors."
          Journal of differential equations 49.2 (1983): 185-207.
    '''
    #ensures that the exponents are sorted
    sortedExps = sorted(lyapunovSpectrum,reverse=True)
    #used to find k
    def indexWhereSumGreaterThanZero(array):
        '''returns the last index where the sum of all indexes
        less than or equal to that index is positive'''
        positiveCount = 0
        for indexPlus1 in range(1,len(array)):
            # sum of all indexes less than i
            lessThanIndexSum = sum(array[:indexPlus1])
            if lessThanIndexSum > 0:
                positiveCount+=1
        return positiveCount
    
    k = indexWhereSumGreaterThanZero(sortedExps)
    if k==0:
        D_ky = 0
    elif k == len(sortedExps)-1:
        D_ky = len(sortedExps)
    else:
        D_ky = k + sum(sortedExps[:k+1])/sortedExps[k+1]
    return D_ky

def calculateAllLyapunovExponents(iterativeMap:Callable[[np.ndarray],np.ndarray],initialPoint:np.ndarray,
                                        iterations:int,progressBarDesc='LEs',progressBar=True) -> np.ndarray:
    '''
    parameters:
    iterativeMap - function from R^n -> R^n. This function must have the optional parameter updateMap
                    specifying whether the function should update hidden internal states. This is necessary so
                    that internal state updates don't take place when calculating the jacobian.
    initialPoint - point in R^n. Specifies the initial condition to calculate the lyapunov exponents around
    iterations - larger than zero int. Specifies the number of updates to perform when calculating the lyapunov
                exponents around. Somewhere around a thousand time steps should be sufficient.
    optional:
    loadBarDesc - desciption of load bar to be displayed when performing the calculation
    displayLoadBar - whether the load bar should be displayed

    outputs: lyapunovExponentArray
    lyapunovExponentArray - an array of the approximate n lyapunov exponents of the iterative map 
                around the initial point.

    Description: Calculates all the lyapunov exponents of an n-dimensional iterative map. Uses the 
    QR factorization algorithm as noted in (paper name) with a small but important difference.
    Since the R matricies are upper triangular, the
      log(Diagonal(R_t*R_{t-1}*...*R_0) = sum(log(Diagonal(R_i)))
    So, to avoid the numerical issues of the product of the R_i growing exponentially, the 
    log is distributed as in the right hand side of the equation as the R_is are iteratively
    generated. An additional difference is that the jacobian is approximated instead of being
    known.'''
    #initializes the algorithm from the jacobian at the initial point 
    currentPoint = initialPoint
    currentJacobian = calculateApproximateJacobian(iterativeMap,currentPoint)
    currentQ,currentR = positiveDiagQR(currentJacobian)
    runningSum = np.log(np.diag(currentR))
    #handles the load bar display
    if progressBar:
        iterator = tqdm(range(iterations),desc=progressBarDesc)
    else:
        iterator = range(iterations)
    # recursively calculates Jstar_t = QR(J(f^t)*Q_{t-1}) = Q_t,R_t and stores the log(diag(R_t))
    # in the logarithm sum.
    for _ in iterator:
        currentPoint = iterativeMap(currentPoint,updateMap=True)
        currentJacobian = calculateApproximateJacobian(iterativeMap,currentPoint)
        jStar = np.matmul(currentJacobian,currentQ)
        currentQ,currentR = positiveDiagQR(jStar)
        runningSum = runningSum + np.log(np.diag(currentR))
    # divides by the time steps
    return runningSum/iterations

def positiveDiagQR(matrix:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    '''Calculates the QR decomposition of the matrix
    while ensuring that the diagonal elements of 
    the matrix R are strictly positive.

    parameters:
    matrix - R^n x R^n

    output: tuple(Q,R)
    Q - R^n x R^m
    R - R^m x R^n upper triangular matrix with positive diagonals
    '''
    Q,R = np.linalg.qr(matrix,mode='full')
    signArr = np.sign(np.diag(R))
    signMatrix = np.diag(signArr)
    return np.matmul(Q,signMatrix),np.matmul(signMatrix,R)
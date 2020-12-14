#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 17:19:00 2018

@author: yannis

Homework 4 PHZ5155
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def gaussxw(N):
    """
    returns integration points x and integration
    weights w such that sum_i w[i]*f(x[i]) is the Nth-order
    Gaussian approximation to the integral int_{-1}^1 f(x) dx
    
    Written by Mark Newman <mejn@umich.edu>, June 4, 2011
    You may use, share, or modify this file freely
    """

    # Initial approximation to roots of the Legendre polynomial
    a = np.linspace(3,4*N-1,N)/(4*N+2)
    x = np.cos(np.pi*a+1/(8*N*N*np.tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = np.ones(N,float)
        p1 = np.copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

def gaussxwab(N,a,b):
    """
    returns integration points and weights
    mapped to the interval [a,b], so that sum_i w[i]*f(x[i])
    is the Nth-order Gaussian approximation to the integral
    int_a^b f(x) dx
    
    Written by Mark Newman <mejn@umich.edu>, June 4, 2011
    You may use, share, or modify this file freely    
    """
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w


def trapezoidIntegratorFunction(lambdaFunction,myInterval):
    """
    Takes as input an interval (myInterval) and a lambda expression (lambdaFunction)
    and returns the integral function of lambdaFunction evaluated at the points of myInterval
    using the trapezoid rule.
    """
    deltaX = (myInterval[1]-myInterval[0])
    
    integralFunction = np.zeros (len(myInterval))
    for i in range(1,len(myInterval)):
        
        increment = (lambdaFunction(myInterval[i])+lambdaFunction(myInterval[i-1])) * deltaX /2
        
        integralFunction[i] = integralFunction[i-1] + increment
    
    return integralFunction
        
        
    

def simpson13IntegratorFunction(lambdaFunction,myInterval):
    """
    Takes as input an interval (myInterval) and a lambda expression (lambdaFunction)
    and returns the integral function of lambdaFunction evaluated at the even-indexed points of myInterval
    using Simpson's 1/3 rule.
    """
    deltaX = (myInterval[1]-myInterval[0])
    integralFunction = np.zeros(len(myInterval)) 
    mask = np.ones(len(myInterval), dtype=bool)
    
    for i in range(1,len(myInterval)):
        if i%2 == 1:
            mask[i] = False
        if i%2 == 0:
            evenPast = lambdaFunction(myInterval[i-2]) * deltaX/3
            oddPast = lambdaFunction(myInterval[i-1]) * deltaX/3
            evenNow = lambdaFunction(myInterval[i]) * deltaX/3
            integralFunction[i] = integralFunction[i-2] + evenPast + 4 * oddPast + evenNow
    return integralFunction[mask]


def gaussianQuad(lambdaFunction,a,b,N):
    """
    This function evaluates the integral of lambdaFunction from a to b
    using the method of Gaussian Quadrature with N nodes
    """
    nodesList, weightsList = gaussxwab(N,a,b)
    return np.sum( weightsList * lambdaFunction(nodesList) )


def gaussQuadFunc(lambdaFunction,myInterval,N):
    """
    This function evaluates the integral function of lambdaFunction at all points of myInterval
    using the method of Gaussian Quadrature with N nodes at each step
    """
    integralFunction = np.zeros (len(myInterval))
    nodesListUnit, weightsListUnit = gaussxw(N)
    for i in range(1,len(myInterval)):
        aI = myInterval[i-1]
        bI = myInterval[i]
        nodesList , weightsList = 0.5*(bI-aI)*nodesListUnit+0.5*(bI+aI) , 0.5*(bI-aI)*weightsListUnit
        increment = np.sum( weightsList * lambdaFunction(nodesList) )
        
        integralFunction[i] = integralFunction[i-1] + increment
    
    return integralFunction
    


def numpyTrapzIntegratorFunction(lambdaFunction,myInterval):
    """
    Takes as input an interval (myInterval) and a lambda expression (lambdaFunction)
    and returns the integral function of lambdaFunction evaluated at the points of myInterval
    using numpy's trapz method.
    """
    
    integralFunction = np.zeros (len(myInterval))
    for i in range(1,len(myInterval)):

        increment = np.trapz( lambdaFunction(myInterval[i-1:i+1]), myInterval[i-1:i+1] ) 
        integralFunction[i] = integralFunction[i-1] + increment
    
    return integralFunction

def firstDir(lambdaFunction, x0 , h):
    """
    Returns the first derivative of lambdaFunction evaluated at the point x0
    """
    return (lambdaFunction(x0+h) - lambdaFunction(x0-h)) / (2*h)

## the following function could have been used to evaluate the error in Simpson's 1/3 rule
#def thirdDir(lambdaFunction , x0 , h):
#    """
#    Returns the third derivative of lambdaFunction evaluated at the point x0
#    """
#    lF = lambdaFunction
#    return ( lF(x0+2*h)-lF(x0-2*h) - 2*(lF(x0+h)-lF(x0-h)) ) / (2 * h**3)


if __name__ == "__main__":
    
    ## since H0 is known with only known up to two significant digits
    ## we round the speed of light also to two significant digits
    
    H0 = 70; #hubble constant (now) in km per sec pec Mpc
    c = 30E4; #speed of light in km per sec 
    OmegaM = 0.3 #dimensionless matter density
    OmegaL = 0.7 #dimensionless dark energy density
    cOverH0 = c/H0;
#    print(cOverH0)
    
    funcToIntegrate = lambda z: 1 / np.sqrt( OmegaM * (1+z)**3 + OmegaL )
    
    zMax = 10
    
    exactIntegral =   integrate.quad(funcToIntegrate, 0, zMax)[0]
    exactValue = 1 / (zMax+1) * cOverH0 * exactIntegral        

    relErrorGoal = 1E-3
    
    ## Estimation of grid points needed to achieve relative accuracy of 
    ## relErrorGoal with the trapezoid rule
    firstDirPointA = firstDir(funcToIntegrate,0,1E-3)
    firstDirPointB = firstDir(funcToIntegrate,zMax,1E-3)
    hGoalTrap = np.sqrt( 12 * relErrorGoal * exactIntegral / np.abs(firstDirPointA-firstDirPointB) )
    nTrap = np.int(np.ceil( zMax / hGoalTrap))
    print("Number of grid points needed to achieve relative accuracy of {} at z = {} with the trapezoid rule :\t{}".format(relErrorGoal,zMax,nTrap))
    
    nPoints = nTrap
    ## nPoints has to be odd so that the number of bins is even (required for Simpson's 1/3)
    if nPoints % 2 == 0:
        nPoints += 1
    
    ## evaluation of the integrals
    intervalZ = np.linspace(0,zMax,nPoints)
    trapIntegral = trapezoidIntegratorFunction(funcToIntegrate,intervalZ)
    simps13Integral = simpson13IntegratorFunction(funcToIntegrate,intervalZ)
    nNodes = 2
    gaussQIntegralFunction = gaussQuadFunc(funcToIntegrate,intervalZ,nNodes)
    numpyTrapIntegral = numpyTrapzIntegratorFunction(funcToIntegrate,intervalZ)
    
    ## evaluation of D_A
    dAtrap = 1 / (intervalZ+1) * cOverH0 * trapIntegral
    dAsimps = 1 / (intervalZ[::2]+1) * cOverH0 * simps13Integral
    dAgaussQ = 1 / (intervalZ + 1) * cOverH0 * gaussQIntegralFunction
    dAnumpyTrap = 1 / (intervalZ+1) * cOverH0 * numpyTrapIntegral
    
    ## Estimation of nodes needed to achieve relative accuracy of 
    ## relErrorGoal with the method of Gaussian Quadrature
    nGauss = 0
    relError = 1
    while relError > relErrorGoal:
        nGauss += 1
        gaussQIntegralzMax = gaussianQuad(funcToIntegrate,0,zMax,nGauss)
        relError = np.abs( (exactIntegral - gaussQIntegralzMax) / exactIntegral )
    gaussQValuezMax = 1 / (zMax + 1) * cOverH0 * gaussQIntegralzMax
    
    print("Number of nodes needed to achieve relative accuracy of {} at z = {} with the Gaussian Quadrature :\t{}".format(relErrorGoal,zMax,nGauss))
    
    fig , ax = plt.subplots(num='Angular Diameter Distance',figsize=(12,8))
    
    plt.plot(intervalZ,dAtrap,'r',linewidth=7, label="Trapezoid")
    plt.plot(intervalZ,dAnumpyTrap,"w-.",linewidth=2, label="np.trapz")
    plt.plot(intervalZ[::2],dAsimps,"bo",markersize=12, label="Simpson's 1/3")
    plt.plot(intervalZ,dAgaussQ,"ko",markersize=6, label="Gaussian Quad")
    
    plt.xlim([0,zMax])
    plt.ylim([750,1800])
    plt.xlabel("Redshift z",fontsize=22)
    plt.ylabel("Angular Diameter Distance (Mpc)",fontsize=22)
    plt.title('The Angular Diamater Distance in the Standard Model of Cosmology',fontsize=22)
    legend = plt.legend(frameon = 1,fontsize=15)
    frame = legend.get_frame()
    frame.set_facecolor('grey')
    frame.set_edgecolor('k')
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    
    print("\nExact Value (scipy.integrate) of the Angular Diameter Distance at z={}:\t\t{:.5f} Mpc".format(zMax,exactValue))
    print("\nResult in Mpc at Ngrid = {} of my implementation of the trapezoid rule:\t\t{:.5f} Mpc".format(nPoints,dAtrap[-1]) )
    print("Result in Mpc at Ngrid = {} of numpy.trapz:\t\t\t\t\t{:.5f} Mpc".format(nPoints,dAnumpyTrap[-1]) )
    print("Result in Mpc at Ngrid = {} of my implementation of the Simpson's 1/3 rule:\t{:.5f} Mpc".format(nPoints,dAsimps[-1]) )
    print("Result in Mpc of my implementation of Gaussian Quadature with {} nodes:\t\t{:.5f} Mpc\n".format(nGauss,gaussQValuezMax))
    
    relD_A_gauss = np.abs( (exactValue - dAgaussQ[-1]) / exactValue )
    print("Note also that for the purposes of plotting the Gaussian Quadrature results we have evaluated D_A in {} evenly-spaced points from z=0 to z={} using {} unevenly-spaced nodes at each step.".format(nPoints,zMax,nNodes) ) 
    print("This obviously uses more that enough points as far as just the value at z={0} is concerned. Indeed in this case we find D_A(z={0}) = {1:.5f} Mpc which corresponds to a relative error of {2:.2e}.".format(zMax,dAgaussQ[-1],relD_A_gauss))  
    
    
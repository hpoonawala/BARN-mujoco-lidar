import numpy as np
from math import sin, pi, tanh, exp, atan2

def basic(sensordata,weights):
    inv_sensor_reading = np.zeros(360)
    for i in range(360):
        if sensordata[i] < 0:
            inv_sensor_reading[i] = 0.0
        else:
            inv_sensor_reading[i] = 1/(sensordata[i])

    w = tanh(sum(inv_sensor_reading*weights)) 
    v = 1*exp(-w*w)
    return v,w

def barn1(sensordata,weights):
    inv_sensor_reading = np.zeros(360)
    for i in range(360):
        if sensordata[i] < 0:
            inv_sensor_reading[i] = 0.0
        else:
            inv_sensor_reading[i] = 1/(sensordata[i]+0.001-0.4)

    ## Get the raw 'score' 
    w_raw = 0.01*sum(inv_sensor_reading*weights)
    ## Saturate to get the CA angular velocity
    w = 2*tanh(w_raw)
    ## Compute heading from quaternion
    theta = 2*atan2(sensordata[366],sensordata[363])
    ## Add a P control for heading to pi/2, based on BARN layouts
    w+=1.0*(1.57-theta)*exp(-w*w)
    ## Use the raw score to slow the robot 
    v = 1*exp(-w_raw*w_raw)
    return v,w


def barn1(sensordata,weights):
    inv_sensor_reading = np.zeros(360)
    for i in range(360):
        if sensordata[i] < 0:
            inv_sensor_reading[i] = 0.0
        else:
            inv_sensor_reading[i] = 1/(sensordata[i]+0.001-0.4)

    ## Get the raw 'score' 
    w_raw = 0.01*sum(inv_sensor_reading*weights)
    ## Saturate to get the CA angular velocity
    w = 2*tanh(w_raw)
    ## Compute heading from quaternion
    theta = 2*atan2(sensordata[366],sensordata[363])
    ## Add a P control for heading to pi/2, based on BARN layouts
    w+=1.0*(1.57-theta)*exp(-w*w)
    ## Use the raw score to slow the robot 
    v = 1*exp(-w_raw*w_raw)
    return v,w


def barn2(sensordata,weights):
    inv_sensor_reading = np.zeros(360)
    ## We can redefine r_max as  : min(r_0/sin(|phi|+eps) , r_max) , which lowers r_max using two walls on either side of the robot
    ## given an r_min, r_max, we can define the breach as max(0, (rmax - r)/(r - rmin) )
    ## Then, control is $\int - \sin phi b(phi)$
    for i in range(360):
        if sensordata[i] < 0:
            inv_sensor_reading[i] = 0.0
        else:
            rmax = np.min([ 0.6 / sin( abs( (i-180)*3.14/180) +0.001),1.0]) 
            inv_sensor_reading[i] = np.max([0, (rmax - sensordata[i])/(sensordata[i]-0.4)] ) 

    ## Get the raw 'score' 
    w_raw = 0.01*sum(inv_sensor_reading*weights)
    ## Saturate to get the CA angular velocity
    w = 2*tanh(w_raw)
    ## Compute heading from quaternion
    theta = 2*atan2(sensordata[366],sensordata[363])
    ## Add a P control for heading to pi/2, based on BARN layouts
    w+=1.0*(1.57-theta)*exp(-w*w)
    ## Use the raw score to slow the robot 
    v = 1*exp(-w_raw*w_raw)
    return v,w

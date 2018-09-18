import numpy as np
import matplotlib.pyplot as plt

class Agent(object):
    def __init__(self, dim_action):
        self.dim_action = dim_action

    def act(self, ob):

        # Get an Observation from the environment.
        # Each observation vectors are numpy array.
        # focus, opponents, track sensors are scaled into [0, 1]. When the agent
        # is out of the road, sensor variables return -1/200.
        # rpm, wheelSpinVel are raw values and then needed to be preprocessed.
        # vision is given as a tensor with size of (64*64, 3) = (4096, 3) <-- rgb
        # and values are in [0, 255]
        '''This is only an example. It will get around the track but the
        correct thing to do is write your own `drive()` function.'''
        S = ob
        target_speed=300
        steer = ob['steer']
        accel = ob['accel']

        # Steer To Corner
        steer = S['angle']*10 / np.pi
        # Steer To Center
        steer-= S['trackPos']*.10

        # Throttle Control
        if S['speedX'] < target_speed - (steer*50):
            accel+= .01
        else:
            accel-= .01
        if S['speedX']<10:
            accel+= 1/(S['speedX']+.1)

        # Traction Control System
        if ((S['wheelSpinVel'][2]+S['wheelSpinVel'][3]) -
            (S['wheelSpinVel'][0]+S['wheelSpinVel'][1]) > 5):
            accel-= .2
        return np.array([np.clip(steer, -1,1), np.clip(accel, 0,1), 0])
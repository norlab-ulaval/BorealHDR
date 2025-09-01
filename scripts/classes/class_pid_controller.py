import numpy as np

class SmoothingController():

    def __init__(self, time_to_target=1, time_constant_tau=1, delay=0):

        self.time_to_target = time_to_target
        self.tau = time_constant_tau
        self.delay = delay

    def first_order_model(self, error):
        delta_next_value = error*(1-np.exp(-(self.time_to_target-self.delay)/self.tau))
        return delta_next_value
    
    @staticmethod
    def ema(current_value, target_value, alpha=1.0):
        return alpha * target_value + (1 - alpha) * current_value
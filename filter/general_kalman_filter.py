import numpy as np

class GeneralKalmanFilter():
    def __init__(self, num_vars, state_covar, process_covar, measurement_covar):
    	"""
    	    Initialize the varables of a general Kalman filter

    	    num_vars: number of values in the state vector
    	    state_covar: vector
    	    process_covar: vector
    	    measurement_covar: vector
    	"""

        self.x = np.zeros(num_vars) # state (initialized at zero) 
        self.P = np.diag(state_covar)  # state covariance
        self.Q = np.diag(process_covar) # process covariance
        self.H = np.zeros(num_vars) # measurement (initialized at zero)
        self.R = np.diag(process_covar) # measurement covariance



    def predict(self):
    	""" Use process model to predict the posterior """
        pass
    

    def update(self):
        """ Calculate the residual and Kalman scaling factor """
    	pass
import numpy as np
import math
import matplotlib.pyplot as plt

class GeneralKalmanFilter():
    """
    Implements the multivariate Kalman Filter algorithm as described in
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/06-Multivariate-Kalman-Filters.ipynb.
    """
    def __init__(self, num_vars, initial_state, state_covar, process_covar, process_transition_function, measurement_function, measurement_covar, control_matrix):
        """
            Initialize the varables of a general Kalman filter

            num_vars: number of values in the state vector
            initial_state: a vector representing the initial guess of the state
            state_covar: vector
            process_covar: vector
            process_transition_function: square matrix
            measurement_covar: vector
            control_matrix: square matrix
        """

        self.x = initial_state # state
        assert initial_state.shape[0] == num_vars, 'state must have the correct number of variables.'
        self.P = np.diag(state_covar)  # state covariance
        assert self.P.shape[0] == self.P.shape[1], 'state covariance must be a square matrix'
        self.F = process_transition_function # transition previous posterior to prior
        assert self.F.shape[0] == self.F.shape[1] and self.F.shape[0] == self.x.shape[0], 'state and state transition function must be compatible sizes.'
        assert self.F.shape == self.P.shape, 'state covariance and state transition function must be the same shape.'
        self.Q = process_covar # process covariance
        assert self.Q.shape == self.F.shape, 'state transition function and process covariance function must be the same shape.'
        self.H = measurement_function # measurement (initialized at zero)
        assert self.H.shape[0] == self.x.shape[0], 'multiplication of the measurement function and the state must result in a 1x1 matrix.'
        self.R = np.diag(measurement_covar) # measurement covariance
        assert self.R.shape == self.P.shape, 'noise covariance must be the same shape as the process covariance.'
        self.B = control_matrix # transform control input

    def predict(self, u):
        """
            Use process model to predict the posterior
            u: control input vector.
        """
        # Calculate predicted state.
        assert self.B.shape[1] == u.shape[0], 'state and control input must be the same shape'
        assert self.x.shape[0] == self.B.shape[0], 'product of u and control matrix must be the same size as the state.'
        self.x = np.matmul(self.F, self.x) + np.matmul(self.B, u)

        # Calculate belief in predicted state.
        self.P = np.matmul(np.matmul(self.F, self.P), np.transpose(self.F)) + self.Q
    
    def update(self, z):
        """ 
            Calculate the residual and Kalman scaling factor 
            z: measurement
        """
        S = np.matmul(np.matmul(self.H, self.P), (self.H.T)) + self.R
        K = np.matmul(np.matmul(self.P, self.H.T), (np.linalg.inv(S)))
        y = z - np.matmul(self.H, self.x)
        self.x += np.matmul(K, y)
        self.P = self.P - np.matmul(np.matmul(K, self.H), (self.P))


if __name__ == "__main__":
    pass

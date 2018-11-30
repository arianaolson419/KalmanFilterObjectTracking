import numpy as np

class GeneralKalmanFilter():
    def __init__(self, num_vars, state_covar, process_covar, process_transition_function, measurement_covar, control_matrix):
        """
            Initialize the varables of a general Kalman filter

            num_vars: number of values in the state vector
            state_covar: vector
            process_covar: vector
            process_transition_function: square matrix
            measurement_covar: vector
            control_matrix: square matrix
        """

        self.x = np.zeros(num_vars) # state (initialized at zero) 
        self.P = np.diag(state_covar)  # state covariance
        assert P.shape[0] == P.shape[1], 'state covariance must be a square matrix'
        self.F = process_transition_function # transition previous posterior to prior
        assert F.shape[0] == F.shape[1] and F.shape[0] == x.shape[0]
        self.Q = np.diag(process_covar) # process covariance
        self.H = np.zeros(num_vars) # measurement (initialized at zero)
        self.R = np.diag(process_covar) # measurement covariance
        self.B = control_matrix # transform control input

    def predict(self, u):
        """ Use process model to predict the posterior """
        # Calculate predicted state.
        self.x = F * x + B * u

        # Calculate belief in predicted state.
        self.P = F * P * np.transpose(F) + Q
    

    def update(self):
        """ Calculate the residual and Kalman scaling factor """
        pass

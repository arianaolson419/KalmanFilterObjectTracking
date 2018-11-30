import numpy as np
import math
import matplotlib.pyplot as plt
class GeneralKalmanFilter():
    """
    Implements the multivariate Kalman Filter algorithm as described in
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/06-Multivariate-Kalman-Filters.ipynb.
    """
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
        assert self.P.shape[0] == self.P.shape[1], 'state covariance must be a square matrix'
        self.F = process_transition_function # transition previous posterior to prior
        assert self.F.shape[0] == self.F.shape[1] and self.F.shape[0] == self.x.shape[0], 'state and state transition function must be compatible sizes.'
        assert self.F.shape == self.P.shape, 'state covariance and state transition function must be the same shape.'
        self.Q = np.diag(process_covar) # process covariance
        assert self.Q.shape == self.F.shape, 'state transition function and process covariance function must be the same shape.'
        self.H = np.zeros(num_vars) # measurement (initialized at zero)
        assert self.H.shape[0] == self.x.shape[0], 'multiplication of the measurement function and the state must result in a 1x1 matrix.'
        self.R = np.diag(process_covar) # measurement covariance
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
        self.x = self.F * self.x + self.B * u

        # Calculate belief in predicted state.
        self.P = self.F * self.P * np.transpose(self.F) + self.Q
    

    def update(self, z):
        """ 
            Calculate the residual and Kalman scaling factor 
            z: measurement
        """
        S = np.dot(self.H, self.P).dot(self.H.T) + self.R
        K = np.dot(self.P, self.H.T).dot(np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x += np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), (self.P))
    
    def compute_fake_data(self, z_var, process_var, count=1, dt=1.):
        "returns track, measurements 1D ndarrays"
        x, vel = 0., 1.
        z_std = math.sqrt(z_var) 
        p_std = math.sqrt(process_var)
        xs, zs = [], []
        for _ in range(count):
            v = vel + (np.random.randn() * p_std)
            x += v*dt        
            xs.append(x)
            zs.append(x + np.random.randn() * z_std)        
        return np.array(xs), np.array(zs)

if __name__ == "__main__":
    square = np.ones((2, 2))
    diag = np.diag([1,1])
    filter = GeneralKalmanFilter(num_vars=2, state_covar=np.ones((2,)), process_covar=np.ones((2,)), process_transition_function=square, measurement_covar=diag, control_matrix=square)

    # Create fake data and use filter on it, plot results
    track, zs = filter.compute_dog_data(1, .01, 50)
    xs, cov = [], []
    for z in zs:
        filter.predict(square)
        filter.update(z)
        xs.append(filter.x)
    
    plt.plot(track, 'b')
    plt.plot(zs, 'g')
    plt.plot(np.array(xs)[:,0], 'r')
    plt.show()

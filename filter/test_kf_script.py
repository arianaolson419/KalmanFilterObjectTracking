import general_kalman_filter
import matplotlib.pyplot as plt
import numpy as np

def compute_fake_data(self, z_var, process_var, count=1, dt=1.):
    """ 
    Returns track, measurements 1D ndarray
    Function adapted from - https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/06-Multivariate-Kalman-Filters.ipynb
    """
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

# Generic matrices to be used as input for filter
initial_state = np.zeros(2,)
square = np.ones((2, 2))
diag = np.diag([1,1])

# instantiate the general filter
filter = general_kalman_filter.GeneralKalmanFilter(num_vars=2,
        initial_state,
        state_covar=np.ones((2,)),
        process_covar=np.ones((2,)),
        process_transition_function=square,
        measurement_function=np.eye(2),
        measurement_covar=diag,
        control_matrix=square)

# Create fake data and use filter on it
track, zs = compute_fake_data(1, .01, 50)
xs, cov = [], []
for z in zs:
    filter.predict(square)
    filter.update(z)
    xs.append(filter.x)

# Plot results of filter and output 
plt.plot(track, 'b', label='Measured Data (no noise)')
plt.plot(zs, 'g', label='Measured Data (noise added)')
plt.plot(np.array(xs)[:,0], 'r', label='Filtered Results')
plt.xlabel('Timestep')
plt.ylabel('State Value')
plt.legend()
plt.show()

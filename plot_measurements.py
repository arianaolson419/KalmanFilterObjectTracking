import numpy as np
import matplotlib.pyplot as plt

def plot_measurements(time, measured_states, kalman_states):
    plt.subplot(221)
    plt.plot(time, measured_states[:, 0], label='raw x')
    plt.plot(time, kalman_states[:, 0], label='filtered x')
    plt.legend()
    plt.title('x position')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (mm)')

    plt.subplot(222)
    plt.plot(time, measured_states[:, 2], label='raw z')
    plt.plot(time, kalman_states[:, 2], label='filtered z')
    plt.legend()
    plt.title('z position')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (mm)')

    plt.subplot(223)
    plt.plot(time, measured_states[:, 1], label='raw v_x')
    plt.plot(time, kalman_states[:, 1], label='filtered v_x')
    plt.legend()
    plt.title('x velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (mm/s)')

    plt.subplot(224)
    plt.plot(time, measured_states[:, 3], label='raw v_z')
    plt.plot(time, kalman_states[:, 3], label='filtered v_z')
    plt.legend()
    plt.title('z velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (mm/s)')

    plt.show()

if __name__ == "__main__":
    data = np.load('output_data.npz')
    times = data['times']
    raw = data['raw']
    filtered = data['filtered']
    plot_measurements(times, raw, filtered)

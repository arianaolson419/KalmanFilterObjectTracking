import numpy as np
import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge
from computer_vision.find_circles import CVOperations
from computer_vision.camera_calibrator import CameraCalibrator
from filter.general_kalman_filter import GeneralKalmanFilter

def calculate_process_covariance(dt, spectral_density):
    """
    Calculate the process covariance matrix given a discrete time step and a
    spectral density.

    Parameters
    ----------
    dt: the time step, in seconds.
    spectral_density: the estimated variance of the white noise in the model.

    Returns
    -------
    A square 4 by 4 numpy array.
    """
    return np.array([[dt ** 3 / 3, dt ** 2 / 2, 0, 0],
        [dt ** 2 / 2, dt, 0, 0],
        [0, 0, dt ** 3 / 3, dt ** 2 / 2],
        [0, 0, dt ** 2 / 2, dt]]) * spectral_density


class BallTrack(object):
    def __init__(self):
        rospy.init_node('ball_track')

        self.camera_sub = rospy.Subscriber("/camera/image_raw", Image, self.get_image)

        # Initialize Kalman Filter. 
        self.num_vars = 4   # The number of state variables.

        # Estimate the initial state.
        initial_state = np.array([0, 0, 1000, 0])   # x, v_x, z, v_z in mm and mm/s
        state_covar = np.array([600 ** 2, 333 ** 2, 1800 ** 2, 333 ** 2]) 

        # Values to calculate predictions.
        dt = 1. / 10.   # Seconds.
        process_transition_function = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

        self.spectral_density = 2.
        process_covar = calculate_process_covariance(self.dt, self.spectral_density)

        # Values to update prediction with measurement.
        measurement_function = np.eye(self.num_vars)
        measurement_covar = np.array([100 ** 2, 20 ** 2, 200 ** 2, 40 ** 2])

        control_matrix = np.zeros((self.num_vars, self.num_vars))

        self.kf = GeneralKalmanFilter(num_vars=self.num_vars,
                initial_state=initial_state,
                state_covar=state_covar,
                process_covar=process_covar,
                process_transition_function=process_transition_function,
                measurement_function=measurement_function,
                measurement_covar=measurement_covar,
                control_matrix=control_matrix)

        self.bridge = CvBridge()
        self.cv_op = CVOperations()
        self.calibrator = CameraCalibrator()
        self.output_window_name = 'Neato Camera Output'
        self.current_image = None
        self.ball_pos = None
        self.ball_vel = None

    def trackbar(self):
        """Allows the user to dynamically adjust the parameters of the Hough Circles algorithm
        """
        title_window = self.output_window_name
        cv2.namedWindow(title_window)

        def on_blue_trackbar(blue):
            self.cv_op.color_thresholds[0] = blue

        def on_green_trackbar(green):
            self.cv_op.color_thresholds[1] = green

        def on_red_trackbar(red):
            self.cv_op.color_thresholds[2] = red

        def on_dp_trackbar(dp):
            self.cv_op.dp = max(0.001, dp / 10.)

        def on_min_dist_trackbar(min_dist):
            self.cv_op.min_dist = max(1, min_dist)

        def on_param_one_trackbar(param_one):
            self.cv_op.param_one = param_one

        def on_param_two_trackbar(param_two):
            self.cv_op.param_two = param_two

        def on_min_radius_trackbar(min_radius):
            self.cv_op.min_radius = min_radius
        
        def on_max_radius_trackbar(max_radius): self.cv_op.max_radius = max_radius

        def on_height_trackbar(height):
            self.calibrator.camera_height = height

        def on_spectral_density_trackbar(spectral_density):
            self.process_covar = calculate_process_covariance(self.dt, spectral_density / 10.)

        # Note: sliders use integer values, so user input may be altered before
        # the CVOperations object is updated.
        cv2.createTrackbar('dp', title_window , int(self.cv_op.dp * 10), 100, on_dp_trackbar)
        cv2.createTrackbar('min_dist', title_window, int(self.cv_op.min_dist), 160, on_min_dist_trackbar)
        cv2.createTrackbar('param_one', title_window, int(self.cv_op.param_one), 500, on_param_one_trackbar)
        cv2.createTrackbar('param_two', title_window, int(self.cv_op.param_two), 500, on_param_two_trackbar)
        cv2.createTrackbar('min_radius', title_window, int(self.cv_op.min_radius), 160, on_min_radius_trackbar)
        cv2.createTrackbar('max_radius', title_window, int(self.cv_op.max_radius), 160, on_max_radius_trackbar)
        cv2.createTrackbar('blue', title_window, int(self.cv_op.color_thresholds[0]), 255, on_blue_trackbar)
        cv2.createTrackbar('green', title_window, int(self.cv_op.color_thresholds[1]), 255, on_green_trackbar)
        cv2.createTrackbar('red', title_window, int(self.cv_op.color_thresholds[2]), 255, on_red_trackbar)
        cv2.createTrackbar('camera height', title_window, int(self.calibrator.camera_height), 200, on_height_trackbar)
        cv2.createTrackbar('spectral_density', title_window, int(self.spectral_density * 10), 100, on_spectral_density_trackbar)


    def get_image(self, img):
        img = self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")
        #img = cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)
        img = np.array(img)
        self.current_image = img

    def run(self):
        r = rospy.Rate(1. / self.dt)
        self.trackbar()
        filtered_measurements = []
        while not rospy.is_shutdown():
            if self.current_image is not None:
                circle = self.cv_op.detect_circles_np_array(self.current_image, self.output_window_name, wait=50)
                if circle is not None:
                    # Only update position if there is a detected ball.
                    print("circle: ", circle)
                    new_pos = self.calibrator.get_object_distance(circle)
                    self.ball_vel = (new_pos - self.ball_pos) / self.dt
                    self.ball_pos = new_pos
                    print("distance: ", self.ball_pos)

            if self.ball_pos is not None:
                measurement = np.array([self.ball_pos[0], self.ball_vel[0], self.ball_pos[1], self.ball_vel[1]])
                self.kf.predict(np.zeros(self.num_vars))
                self.kf.update(measurement)
                filtered_measurements.append(self.kf.x)
                print(self.kf.x)

            r.sleep()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    node = BallTrack()
    node.run()




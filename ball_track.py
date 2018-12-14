import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from cv_bridge import CvBridge
import rospy
from geometry_msgs.msg import Point, Pose, Quaternion
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker

from computer_vision.camera_calibrator import CameraCalibrator
from computer_vision.find_circles import CVOperations
from filter.general_kalman_filter import GeneralKalmanFilter

FLAGS = None

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
        self.twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.vis_pub1 = rospy.Publisher('/visualization_marker1', Marker, queue_size=10)
        self.vis_pub2 = rospy.Publisher('/visualization_marker2', Marker, queue_size=10)

        self.twist = Twist()
        self.max_speed = 0.1
        self.target_z = 400

        # Initialize Kalman Filter. 
        self.num_vars = 4   # The number of state variables.

        # Estimate the initial state.
        initial_state = np.array([0, 0, 1000, 0])   # x, v_x, z, v_z in mm and mm/s
        state_covar = np.array([300 ** 2, 333 ** 2, 900 ** 2, 333 ** 2]) 
        #state_covar = np.array([600 ** 2, 333 ** 2, 1800 ** 2, 333 ** 2]) 

        # Values to calculate predictions.
        self.dt = 1. / 10.   # Seconds.
        process_transition_function = np.array([[1, self.dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.dt], [0, 0, 0, 1]])

        self.spectral_density = 0.5
        process_covar = calculate_process_covariance(self.dt, self.spectral_density)

        # Values to update prediction with measurement.
        measurement_function = np.eye(self.num_vars)
        #measurement_covar = np.array([100 ** 2, 20 ** 2, 200 ** 2, 40 ** 2])
        measurement_covar = np.array([1 ** 2, 2 ** 2, 7 ** 2, 4 ** 2])

        lin_scale = 970.0
        control_matrix = np.array([[0], [0], [-lin_scale * self.dt], [-lin_scale]])

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
        self.ball_pos = np.array([initial_state[0], initial_state[2]])
        self.ball_vel = np.array([initial_state[1], initial_state[3]])

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
    
    def move_to_ball(self):
        max_error = 50
        kp = 0.0005
        error = self.kf.x[3] - self.target_z
        if np.abs(error) < max_error:
            self.twist.linear.x = 0
        else:
            self.twist.linear.x = np.sign(error) * min(self.max_speed, np.abs(error) * kp)
        self.twist_pub.publish(self.twist)

    def visualize_ball_rviz(self):
        """ Function to visualize the ball's expected location in rViz"""

        # Define the pose for rviz marker for predicted and measured ball locations
        ball_quaternion = Quaternion(0,0,0,0) # Neither ball has orientation, so both set to all zeros
        measured_ball_point = Point(self.kf.x[0]/1000, self.kf.x[2]/1000, 0) # Divide by 1000 to convert mm to m
        measured_ball_pose = Pose(measured_ball_point, ball_quaternion)
        predicted_ball_point = Point(self.ball_pos[0]/1000, self.ball_pos[1]/1000, 0) 
        predicted_ball_pose = Pose(predicted_ball_point, ball_quaternion)

        # Define the predicted ball rviz marker's properties 
        vis_msg_pred = Marker()
        vis_msg_pred.pose = predicted_ball_pose
        vis_msg_pred.type = 2 # Sphere marker type
        vis_msg_pred.header.frame_id = "base_link"
        vis_msg_pred.scale.x = 0.5
        vis_msg_pred.scale.y = 0.5
        vis_msg_pred.scale.z = 0.5
        vis_msg_pred.color.a = 1.0
        vis_msg_pred.color.r = 1.0
        vis_msg_pred.color.g = 0.0
        vis_msg_pred.color.b = 0.0
        self.vis_pub1.publish(vis_msg_pred)

        # Define the measured ball rviz marker's properties
        vis_msg_meas = Marker()
        vis_msg_meas.pose = measured_ball_pose
        vis_msg_meas.type = 2 # Sphere marker type
        vis_msg_meas.header.frame_id = "odom"
        vis_msg_meas.scale.x = 0.5
        vis_msg_meas.scale.y = 0.5
        vis_msg_meas.scale.z = 0.5
        vis_msg_meas.color.a = 1.0
        vis_msg_meas.color.r = 0.0
        vis_msg_meas.color.g = 0.0
        vis_msg_meas.color.b = 1.0
        self.vis_pub2.publish(vis_msg_meas)

    def run(self):
        r = rospy.Rate(1. / self.dt)
        self.trackbar()
        times = []
        raw_measurements = []
        model_predictions = []
        filtered_measurements = []
        circle_radius = 0
        while not rospy.is_shutdown():
            if self.current_image is not None:
                circle = self.cv_op.detect_circles_np_array(self.current_image, self.output_window_name, wait=50)
                if circle is not None:
                    # Only update position if there is a detected ball.
                    new_pos = self.calibrator.get_object_distance(circle)
                    self.ball_vel = (new_pos - self.ball_pos) / self.dt
                    self.ball_pos = new_pos
                    circle_radius = circle[2]
                else:
                    self.ball_pos[0] = np.random.normal(self.ball_pos[0], np.abs(self.ball_pos[0]) * 0.05)
                    self.ball_pos[1] = np.random.normal(self.ball_pos[1], np.abs(self.ball_pos[1]) * 0.05)
                    self.ball_vel[0] = np.random.normal(self.ball_vel[0], np.abs(self.ball_vel[0]) * 0.05)
                    self.ball_vel[1] = np.random.normal(self.ball_vel[1], np.abs(self.ball_vel[1]) * 0.05)

            measurement = np.array([self.ball_pos[0], self.ball_vel[0], self.ball_pos[1], self.ball_vel[1]])
            if FLAGS.save_data:
                times.append(rospy.get_time())
                raw_measurements.append(measurement)

            u = np.array([self.twist.linear.x])
            predicted_state = self.kf.predict(u)

            if FLAGS.save_data:
                model_predictions.append(predicted_state)
            self.kf.update(measurement)

            if FLAGS.drive:
                self.move_to_ball()

            estimated_circle = self.calibrator.get_circle_pixel_location(self.kf.x[0], self.kf.x[2], circle_radius)
            self.cv_op.draw_circle(estimated_circle, self.current_image, color=(255, 0, 0))
            self.visualize_ball_rviz()

            if FLAGS.save_data:
                filtered_measurements.append(self.kf.x)

            r.sleep()
        cv2.destroyAllWindows()

        if FLAGS.save_data:
            np.savez('output_data',
                    times,
                    raw_measurements,
                    filtered_measurements,
                    model_predictions,
                    times=times,
                    raw=raw_measurements,
                    filtered=filtered_measurements,
                    predicted=model_predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--drive', type=bool, default=False, help="Allow robot to drive around.")
    parser.add_argument('--save_data', type=bool, default=False, help="Save data as a .npz file.")
    FLAGS, _ = parser.parse_known_args()
    node = BallTrack()
    node.run()




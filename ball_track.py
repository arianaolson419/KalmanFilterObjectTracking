import numpy as np
import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge
from computer_vision.find_circles import CVOperations

class BallTrack(object):
    def __init__(self):
        rospy.init_node('ball_track')

        self.camera_sub = rospy.Subscriber("/camera/image_raw", Image, self.find_circles)

        self.bridge = CvBridge()
        self.cv_op = CVOperations()
        self.output_window_name = 'Neato Camera Output'
        self.current_image = None
        self.pixel_pos = None

    def trackbar(self):
        """Allows the user to dynamically adjust the parameters of the Hough Circles algorithm
        """
        title_window = self.output_window_name
        cv2.namedWindow(title_window)

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
        
        def on_max_radius_trackbar(max_radius):
            self.cv_op.max_radius = max_radius

        # Note: sliders use integer values, so user input may be altered before
        # the CVOperations object is updated.
        cv2.createTrackbar('dp', title_window , int(self.cv_op.dp * 10), 100, on_dp_trackbar)
        cv2.createTrackbar('min_dist', title_window, int(self.cv_op.min_dist), 160, on_min_dist_trackbar)
        cv2.createTrackbar('param_one', title_window, int(self.cv_op.param_one), 500, on_param_one_trackbar)
        cv2.createTrackbar('param_two', title_window, int(self.cv_op.param_two), 500, on_param_two_trackbar)
        cv2.createTrackbar('min_radius', title_window, int(self.cv_op.min_radius), 160, on_min_radius_trackbar)
        cv2.createTrackbar('max_radius', title_window, int(self.cv_op.max_radius), 160, on_max_radius_trackbar)

    def find_circles(self, img):
        img = self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")
        img = cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)
        img = np.array(img)
        self.current_image = img

    def run(self):
        rospy.Rate(2)
        self.trackbar()
        while not rospy.is_shutdown():
            if self.current_image is not None:
                circle = self.cv_op.detect_circles_np_array(self.current_image, self.output_window_name, wait=50)
                if circle is not None:
                    # Only update position if there is a detected ball.
                    self.pixel_pos = circle
        cv2.destroyAllWindows()

if __name__ == "__main__":
    node = BallTrack()
    node.run()




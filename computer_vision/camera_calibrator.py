#!/usr/bin/env python
import rospy
from sensor_msgs.msg import CameraInfo

class CameraCalibrator():
    """
       This class accesses the camera calibration info (focal length and image center) and uses these
       values to find the real object location from the image data
    """
    def __init__(self):
        rospy.init_node('camera_calibrator')
        self.K = [] # Initializes the camera calibration values
        self.retrieve_camera_info()

    def retrieve_camera_info(self):
        """ Accesses the camera info from the rostopic """
        rospy.Subscriber("/sensor_msgs/camera_info", CameraInfo, self.callback)

    def callback(self, msg):
        """ Callback function to store the calibration data """
        self.K = msg.K

    def get_object_distance(self, real_height, img_height, img_center):
    """ Find the distance to the object given the object height """
        pass

if __name__ == '__main__':
    c = CameraCalibrator()
    print c.K
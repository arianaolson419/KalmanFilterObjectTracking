#!/usr/bin/env python
import rospy
from sensor_msgs.msg import CameraInfo
import time 

class CameraCalibrator():
    """
       This class accesses the camera calibration info (focal length and image center) and uses these
       values to find the real object location from the image data
    """
    def __init__(self):
        rospy.init_node('camera_calibrator')
        self.K = [] # Initialize camera parameters    
        self.camera_height = .15 # Height of camera off ground in meters

        # Subscriber accesses the camera info from the rostopic
        camera_sub = rospy.Subscriber('/camera/camera_info', CameraInfo, self.callback) 
        rospy.rostime.wallsleep(0.1) # waits to subscriber has time to access data
        
        # Obtain relevant values from K
        self.fx = self.K[0]
        self.fy = self.K[4]
        self.cx = self.K[2]
        self.cy = self.K[5]

    def callback(self, msg):
    	""" Callback function to store the calibration data """
        self.K = msg.K

    def get_object_distance(self, radius_measured, y_img):
        """ Find the distance to the object given the y value at which the object touches the ground """
        y_ground = y_img - radius_measured
        z = (-self.camera_height * self.fy)/(y_ground - self.cy)
        return z

if __name__ == '__main__':
    c = CameraCalibrator()
    print c.fy


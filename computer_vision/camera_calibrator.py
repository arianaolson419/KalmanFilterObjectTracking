#!/usr/bin/env python
import rospy
from sensor_msgs.msg import CameraInfo
import math
import numpy as np

class CameraCalibrator():
    """
       This class accesses the camera calibration info (focal length and image center) and uses these
       values to find the real object location from the image data
    """
    def __init__(self):
        #rospy.init_node('camera_calibrator')
        self.K = [] # Initialize camera parameters    
        self.camera_height = 145. # Height of camera off ground in mm

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

    def get_object_distance(self, circle):
        """ Find the distance to the object given the y value at which the object touches the ground """
        x_img = circle[0] # x pixel of center of ball
        y_img = circle[1] # y pixel of center of ball
        r = circle[2] # radius of ball

        y_ground = y_img + r # y-value of pixel where ball is at ground height
        z = (self.camera_height * self.fy)/(y_ground - self.cy) # z-axis distance to object 
        x = ((x_img-self.cx)*z)/self.fx # x-axis distance to object

        return np.array([x,z])

    def get_circle_pixel_location(self, kf_x, kf_z, radius):
        """ Calculate the position of the circle in the image from data about the distance of the ball """
        y_ground = ((self.camera_height * self.fy)/kf_z) + self.cy # y-value of bottom pixel of ball
        y_img = y_ground - radius # y-value of center pixel of ball

        x_img = ((kf_x * self.fx)/z) + self.cx 

        return (x_img, y_img, radius)
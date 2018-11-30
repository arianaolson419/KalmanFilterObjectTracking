#!/usr/bin/env python

import rospy
from sensor_msgs.msg import CameraInfo

class CameraCalibrator():
    def __init__(self):
    	rospy.init_node('camera_calibrator')
    	self.K = []
        self.retrieve_camera_info()

    def callback(self, msg):
    	print 'hi'
        self.K = msg.K

    def retrieve_camera_info(self):
        rospy.Subscriber("/sensor_msgs/camera_info", CameraInfo, self.callback)

if __name__ == '__main__':
    c = CameraCalibrator()
    print c.K
# Kalman Filters for Object Tracking
The main class for this project is located in [ball_track.py](https://github.com/arianaolson419/KalmanFilterObjectTracking/blob/master/ball_track.py). Running this python code will start up the program. 

Please note that this code is configured to work with Olin’s Neato robots. For help setting up a ROS environment and connecting to Neatos please see: https://sites.google.com/site/comprobo18/how-to/neato-etiquette and https://sites.google.com/site/comprobo18/how-to/setting-up-your-environment

Start ROS:
```roscore```

Connect to Neato:
```roslaunch neato_node bringup.launch host:=<neato IP address>```

Start ball tracking:
By default, the robot does not move and it does not save the data it collects. Also not that this code must be run with python 2.
```python2 ball_track.py```
A window will appear with a video feed from the Neato’s camera and slider bars to adjust the  some parameters of the Kalman filter and the HoughCircles algorithm. Please see the code documentation for an explanation of these parameters.

To run with robot movement:
```python2 ball_track.py --drive=true```
The robot will move back and forth to adjust its distance from the ball estimation. Please note that when the program is terminated, the robot may not stop, so it is recommended to keep a teleop program open to control the robot manually.

To save the filter data:
```python2 ball_track.py --save_data=true```
The timestamps of the data points, the raw ball location measurements, and the filtered data are saved in output_data.npz. To view the plots of the data, run
```python2 plot_measurements.py```
A figure will appear with a subplot of the data for each of the states tracked by the Kalman filter.

Please see our [website](https://arianaolson419.github.io/KalmanFilterObjectTracking/) if you'd like more details about this project, the process of developing it, or how it works.

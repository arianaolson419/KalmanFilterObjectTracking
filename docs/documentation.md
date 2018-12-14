---
title: Documentation
---

- [Home](index.md)

## Documentation
### Overview
#### What does it do?
For this project, we implemented  a Kalman filter to improve a simple computer vision algorithm with the goal of making a Neato follow around a soccer ball.

![A drawing showing the Neato estimating the position of a red ball](images/NeatoFindingBall.png)

*The Neato estimates where the ball is and keeps it a fixed distance away.*

A general block diagram of the process can be seen below. First, the position of the ball is obtained by detecting circles in video frames, identifying the circle most likely to be the ball, and translating its location in pixels to a position vector relative to the robot using camera calibration. This measured position is then input into the Kalman filter, which produces a filtered, and ideally more accurate, estimate of the ball’s position using a predictive model along with the input data. Finally, proportional control is used to direct the robot to approach or retreat from the ball to maintain a given distance. The motor controls given to the robot are also input into the Kalman filter for a more robust predictive model.

![The block diagram of the system outlining its major components](images/general_block_diagram.png)

### What is a Kalman Filter?

Because of imperfect sensors, noise affects measurements in many systems and obscures the overall behavior over time. In such systems, Kalman filters provide a method to more accurately estimate how the state of the system behaves over time. Kalman filters keep track of an estimate for the system’s true state and a variance of this estimate, which it uses, along with the actual measurement, to update it’s estimated state at the next time step. To do so, these filters first use a state transition model on the current estimate’s probability distribution to determine a prediction for the state of the system. This prediction is then weighted together with the actual measurement to determine the next estimate of the system state.


We found that [this book](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) provides a thorough introduction to the mathematics behind Kalman filters and their implementation.

### System Architecture

#### General Filter

To implement a working filter for this project, we first implemented a Python class for a general multivariate Kalman filter. Because we designed this class to be general, we are able to specify details about the system such as the number of variables. This allowed us to build up complexity throughout our project by first testing on less complex systems. For example, we first tested our filter on a system with measurements defined by f(t)=t, with noise added into each measurement. We found that our filter worked very well on this system, the results of which are shown in the plot below. 

![An example of the Kalman filter smoothing out a noisy linear function.](images/general_kalman_filter_example.png)




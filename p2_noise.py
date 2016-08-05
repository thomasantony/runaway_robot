# ----------
# Part Two
#
# Now we'll make the scenario a bit more realistic. Now Traxbot's
# sensor measurements are a bit noisy (though its motions are still
# completetly noise-free and it still moves in an almost-circle).
# You'll have to write a function that takes as input the next
# noisy (x, y) sensor measurement and outputs the best guess
# for the robot's next position.
#
# ----------
# YOUR JOB
#
# Complete the function estimate_next_pos. You will be considered
# correct if your estimate is within 0.01 stepsizes of Traxbot's next
# true position.
#
# ----------
# GRADING
#
# We will make repeated calls to your estimate_next_pos function. After
# each call, we will compare your estimated position to the robot's true
# position. As soon as you are within 0.01 stepsizes of the true position,
# you will be marked correct and we will tell you how many steps it took
# before your function successfully located the target bot.

# These import steps give you access to libraries which you may (or may
# not) want to use.
from robot import *  # Check the robot.py tab to see how this works.
from math import *
from matrix import * # Check the matrix.py tab to see how this works.
import random

from filters import extended_kalman_filter
from utils import identity_matrix


def setup_kalman_filter(z):
    """
    Setup Kalman Filter for this problem

    z : Initial measurement
    """
    # Setup 5D Kalman filter
    # initial uncertainty: 0 for positions x and y,
    # 1000 for the two velocities and accelerations
    # measurement function: reflect the fact that
    # we observe x and y
    H =  matrix([[1., 0., 0., 0., 0.],
                 [0., 1., 0., 0., 0.]])
     # measurement uncertainty: use 2x2 matrix with 0.1 as main diagonal
    R = matrix([[1., 0.0],
                [0.0, 1.]])

    u = matrix([[0.], [0.], [0.], [0.], [0.]]) # external motion

    I = identity_matrix(5)
    P =  I*1000.0  # 1000 along main diagonal

    x = matrix([[z[0]], [z[1]], [0.], [0.], [0.]])
    return [ x, u, P, H, R]

def robot_F_fn(state, dt = 1.0):
    """
    Transition matrix for robot dynamics

    Linearize dynamics about 'state' for EKF

    xdot = v*cos(theta+w)
    ydot = v*sin(theta+w)
    thetadot = w
    vdot = 0    -> step size
    wdot = 0    -> turn angle per step
    """
    x = state.value[0][0]
    y = state.value[1][0]
    theta = state.value[2][0]
    v = state.value[3][0]
    w = state.value[4][0]

    J = matrix([[0., 0.,     -v*sin(theta), cos(theta),              0.],
                [0., 0.,      v*cos(theta), sin(theta),              0.],
                [0., 0.,              0.,           0.,              1.],
                [0., 0.,              0.,           0.,              0.],
                [0., 0.,              0.,           0.,              0.],
               ])

    I = matrix([[]])
    I.identity(5)
    return I + J*dt


def robot_x_fn(state, dt=1.0):
    """
    State update for nonlinear system

    Computes next state using the non-linear dynamics
    """
    x = state.value[0][0]
    y = state.value[1][0]
    theta = state.value[2][0]
    v = state.value[3][0]
    w = state.value[4][0]

    x += v * cos(theta)*dt
    y += v * sin(theta)*dt
    theta += w*dt

    return matrix([[x],
                   [y],
                   [theta],
                   [v],
                   [w]])


def state_from_measurements(three_measurements):
    """
    Estimates state of robot from the last three measurements

    Assumes each movement of robot is a "step" and a "turn"
    Three measurements constitute two moves, from which turn angle, heading
    and step size can be inferred.
    """

    x1, y1 = three_measurements[-3]
    x2, y2 = three_measurements[-2]
    x3, y3 = three_measurements[-1]

    # Last two position vectors from measurements
    vec_1 = [x2 - x1, y2 - y1]
    vec_2 = [x3 - x2, y3 - y2]

    # Find last turning angle using dot product
    dot = sum(v1*v2 for v1,v2 in zip(vec_1, vec_2))
    mag_v1 = sqrt(sum(v**2 for v in vec_1))
    mag_v2 = sqrt(sum(v**2 for v in vec_2))

    v0 = mag_v2
    w0 = acos(dot/(mag_v1*mag_v2))
    theta0 = atan2(vec_2[1], vec_2[0]) + w0
    x0 = x3 + v0*cos(theta0)
    y0 = y3 + v0*sin(theta0)

    return matrix([[x3], [y3], [theta0], [v0], [w0]])

# This is the function you have to write. Note that measurement is a
# single (x, y) point. This function will have to be called multiple
# times before you have enough information to accurately predict the
# next position. The OTHER variable that your function returns will be
# passed back to your function the next time it is called. You can use
# this to keep track of important information over time.
def estimate_next_pos(measurement, OTHER = None):
    """Estimate the next (x, y) position of the wandering Traxbot
    based on noisy (x, y) measurements."""
    if OTHER is None:
        # Setup Kalman Filter
        [ x, u, P, H, R] = setup_kalman_filter(measurement)
        # OTHER = {'x': x, 'P': P, 'u': u, 'matrices':[H, R]}
        # est_xy = (x.value[0][0], x.value[1][0])
        OTHER = {'z_list': [measurement], 'x': None,
                 'P': P, 'u': u, 'matrices': [H, R], 'step': 1}
        return measurement, OTHER
    elif OTHER['step'] == 1:
        # Use first three measurements to seed the filter
        OTHER['step'] = 2
        OTHER['z_list'].append(measurement)
        return measurement, OTHER
    elif OTHER['step'] == 2:
        OTHER['step'] = 3
        OTHER['z_list'].append(measurement)

        # Get initial estimate of state from the three measurements
        OTHER['x'] = state_from_measurements(OTHER['z_list'])

        # Initialization complete
        OTHER['step'] = -1
        del OTHER['z_list']

        # Predict next position of robot using the dynamics and current state
        next_state = robot_x_fn(OTHER['x'])
        return (next_state.value[0][0], next_state.value[1][0]), OTHER

    x, P = extended_kalman_filter(measurement, OTHER['x'], OTHER['u'],
                        OTHER['P'], robot_F_fn, robot_x_fn, *OTHER['matrices'])
    OTHER['x'] = x
    OTHER['P'] = P

    # Predict next position of robot
    next_state = robot_x_fn(x)
    est_xy = (next_state.value[0][0], next_state.value[1][0])

    # You must return xy_estimate (x, y), and OTHER (even if it is None)
    # in this order for grading purposes.
    # xy_estimate = (3.2, 9.1)
    return est_xy, OTHER

# A helper function you may find useful.
def distance_between(point1, point2):
    """Computes distance between point1 and point2. Points are (x, y) pairs."""
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# This is here to give you a sense for how we will be running and grading
# your code. Note that the OTHER variable allows you to store any
# information that you want.
def demo_grading(estimate_next_pos_fcn, target_bot, OTHER = None):
    localized = False
    distance_tolerance = 0.01 * target_bot.distance
    ctr = 0
    # if you haven't localized the target bot, make a guess about the next
    # position, then we move the bot and compare your guess to the true
    # next position. When you are close enough, we stop checking.
    while not localized and ctr <= 1000:
        ctr += 1
        measurement = target_bot.sense()
        position_guess, OTHER = estimate_next_pos_fcn(measurement, OTHER)
        target_bot.move_in_circle()
        true_position = (target_bot.x, target_bot.y)
        error = distance_between(position_guess, true_position)
        print(error)
        if error <= distance_tolerance:
            print("You got it right! It took you ", ctr, " steps to localize.")
            localized = True
        if ctr == 1000:
            print("Sorry, it took you too many steps to localize the target.")
    return localized

def demo_grading2(estimate_next_pos_fcn, target_bot, OTHER = None):
    localized = False
    distance_tolerance = 0.01 * target_bot.distance
    ctr = 0
    # if you haven't localized the target bot, make a guess about the next
    # position, then we move the bot and compare your guess to the true
    # next position. When you are close enough, we stop checking.
    #For Visualization
    import turtle    #You need to run this locally to use the turtle module
    window = turtle.Screen()
    window.bgcolor('white')
    size_multiplier= 25.0  #change Size of animation
    broken_robot = turtle.Turtle()
    broken_robot.shape('turtle')
    broken_robot.color('green')
    broken_robot.resizemode('user')
    broken_robot.shapesize(0.1, 0.1, 0.1)
    measured_broken_robot = turtle.Turtle()
    measured_broken_robot.shape('circle')
    measured_broken_robot.color('red')
    measured_broken_robot.resizemode('user')
    measured_broken_robot.shapesize(0.1, 0.1, 0.1)
    prediction = turtle.Turtle()
    prediction.shape('arrow')
    prediction.color('blue')
    prediction.resizemode('user')
    prediction.shapesize(0.1, 0.1, 0.1)
    prediction.penup()
    broken_robot.penup()
    measured_broken_robot.penup()
    #End of Visualization
    while not localized and ctr <= 1000:
        ctr += 1
        measurement = target_bot.sense()
        position_guess, OTHER = estimate_next_pos_fcn(measurement, OTHER)
        target_bot.move_in_circle()
        true_position = (target_bot.x, target_bot.y)
        error = distance_between(position_guess, true_position)
        if error <= distance_tolerance:
            print("You got it right! It took you ", ctr, " steps to localize.")
            localized = True
        if ctr == 1000:
            print("Sorry, it took you too many steps to localize the target.")
        #More Visualization
        measured_broken_robot.setheading(target_bot.heading*180/pi)
        measured_broken_robot.goto(measurement[0]*size_multiplier, measurement[1]*size_multiplier-200)
        measured_broken_robot.stamp()
        broken_robot.setheading(target_bot.heading*180/pi)
        broken_robot.goto(target_bot.x*size_multiplier, target_bot.y*size_multiplier-200)
        broken_robot.stamp()
        prediction.setheading(target_bot.heading*180/pi)
        prediction.goto(position_guess[0]*size_multiplier, position_guess[1]*size_multiplier-200)
        prediction.stamp()
        #End of Visualization
    return localized

# This is a demo for what a strategy could look like. This one isn't very good.
def naive_next_pos(measurement, OTHER = None):
    """This strategy records the first reported position of the target and
    assumes that eventually the target bot will eventually return to that
    position, so it always guesses that the first position will be the next."""
    if not OTHER: # this is the first measurement
        OTHER = measurement
    xy_estimate = OTHER
    return xy_estimate, OTHER

# This is how we create a target bot. Check the robot.py file to understand
# How the robot class behaves.
test_target = robot(2.1, 4.3, 0.5, 2*pi / 34.0, 1.5)
measurement_noise = 0.05 * test_target.distance
test_target.set_noise(0.0, 0.0, measurement_noise)

# demo_grading(naive_next_pos, test_target)
demo_grading(estimate_next_pos, test_target)

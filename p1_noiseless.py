# ----------
# Background
#
# A robotics company named Trax has created a line of small self-driving robots
# designed to autonomously traverse desert environments in search of undiscovered
# water deposits.
#
# A Traxbot looks like a small tank. Each one is about half a meter long and drives
# on two continuous metal tracks. In order to maneuver itself, a Traxbot can do one
# of two things: it can drive in a straight line or it can turn. So to make a
# right turn, A Traxbot will drive forward, stop, turn 90 degrees, then continue
# driving straight.
#
# This series of questions involves the recovery of a rogue Traxbot. This bot has
# gotten lost somewhere in the desert and is now stuck driving in an almost-circle: it has
# been repeatedly driving forward by some step size, stopping, turning a certain
# amount, and repeating this process... Luckily, the Traxbot is still sending all
# of its sensor data back to headquarters.
#
# In this project, we will start with a simple version of this problem and
# gradually add complexity. By the end, you will have a fully articulated
# plan for recovering the lost Traxbot.
#
# ----------
# Part One
#
# Let's start by thinking about circular motion (well, really it's polygon motion
# that is close to circular motion). Assume that Traxbot lives on
# an (x, y) coordinate plane and (for now) is sending you PERFECTLY ACCURATE sensor
# measurements.
#
# With a few measurements you should be able to figure out the step size and the
# turning angle that Traxbot is moving with.
# With these two pieces of information, you should be able to
# write a function that can predict Traxbot's next location.
#
# You can use the robot class that is already written to make your life easier.
# You should re-familiarize yourself with this class, since some of the details
# have changed.
#
# ----------
# YOUR JOB
#
# Complete the estimate_next_pos function. You will probably want to use
# the OTHER variable to keep track of information about the runaway robot.
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
from robot import *
from math import sin, cos, acos, atan2
from matrix import *
import random


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
    x0 = x3 + v0*cos(theta0 + w0)
    y0 = y3 + v0*sin(theta0 + w0)

    return matrix([[x3], [y3], [theta0], [v0], [w0]])

def estimate_next_pos(measurement, OTHER = None):
    """Estimate the next (x, y) position of the wandering Traxbot
    based on noisy (x, y) measurements.

    Takes three measurements to find angle between last two steps and distance
    """

    if OTHER is None:
        # Store first measurement
        OTHER = [measurement]
        return ([measurement[0], measurement[1]], OTHER)
    elif len(OTHER) == 1:
        # Second measurement
        OTHER.append(measurement)
        return ([measurement[0], measurement[1]], OTHER)
    else:
        # Third and subsequent measurements
        OTHER.append(measurement)
        state = state_from_measurements(OTHER)

        # Estimate next position from current state
        x, y = state.value[0][0], state.value[1][0]
        theta, v, w = state.value[2][0], state.value[3][0], state.value[4][0]
        est_xy = [x + v*cos(theta),
                  y + v*sin(theta)]
        return (est_xy, OTHER)

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
    while not localized and ctr <= 10:
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
        if ctr == 10:
            print("Sorry, it took you too many steps to localize the target.")
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
test_target.set_noise(0.0, 0.0, 0.0)



#demo_grading(naive_next_pos, test_target)
demo_grading(estimate_next_pos, test_target)

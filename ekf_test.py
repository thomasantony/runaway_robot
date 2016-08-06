from robot import *
from matrix import *
import matplotlib.pyplot as plt
from filters import extended_kalman_filter
from utils import identity_matrix
from functools import reduce, partial
from collections import deque
import numpy.ma
from scipy import optimize

def setup_kalman_filter():
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
    R = matrix([[0.1, 0.0],
                [0.0, 0.1]])

    u = matrix([[0.], [0.], [0.], [0.], [0.]]) # external motion

    I = identity_matrix(5)
    # P = I*10000.0
    P = matrix([[1000.0, 0.,    1000., 1000., 0.   ],
                [0.,     1000., 1000., 1000., 0.   ],
                [0.,     0.,    1000., 0.,    1000.],
                [0.,     0.,    0.,    1000., 0.   ],
                [0.,     0.,    0.,    0.,    1000.],
    ])

    # I*1000.0  # 1000 along main diagonal
    # P.value[0][0] = 100.0
    # P.value[1][1] = 100.0
    # P.value[2][2] = 100.0
    # P.value[3][3] = 100.0
    # P.value[4][4] = 100.0

    return [ u, P, H, R]

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

import pandas as pd
import numpy as np
def ewma(values, period):
    values = np.array(values)
    return pd.ewma(values, span=period)[-1]

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
        [u, P, H, R] = setup_kalman_filter()
        # OTHER = {'x': x, 'P': P, 'u': u, 'matrices':[H, R]}
        x = matrix([[measurement[0]], [measurement[1]], [0], [0], [0]])

        OTHER = {'z_list': deque([]), 'x': x,
                 'P': P, 'u': u, 'matrices': [H, R], 'step': 1
                #  'zx': [measurement[0]]
                 }
        OTHER['z_list'].append(np.array(measurement))
    #     return measurement, OTHER
    # elif OTHER['step'] == 1:
    #     # Use first three measurements to seed the filter
    #     OTHER['step'] = 2
    #     OTHER['z_list'].append(np.array(measurement))
    #     # OTHER['zx'].append(measurement[0])
    #     # OTHER['x_list'].append(measurement)
    #     return measurement, OTHER
    # elif OTHER['step'] == 2:
    #     OTHER['step'] = 3
    #     # Get last 3 measurements
    #     OTHER['z_list'].append(np.array(measurement))
    #     # OTHER['zx'].append(measurement[0])
    #     # Get initial estimate of state from the three measurements
    #     OTHER['x'] = state_from_measurements(OTHER['z_list'])
    #
    #     # Initialization complete
    #     OTHER['step'] = -1
    #
    #     # Use last 20 measurements only
    #     num_z = 1000
    #     # OTHER['x_list'] = deque(maxlen=num_z)
    #     # OTHER['z_list'] = deque(maxlen=num_z+1)
    #
    #     # Predict next position of robot using the dynamics and current state
    #     next_state = robot_x_fn(OTHER['x'])
    #     # OTHER['x_list'].append(next_state)
    #     return (next_state.value[0][0], next_state.value[1][0]), OTHER

    OTHER['z_list'].append(np.array(measurement))
    x, P = extended_kalman_filter(measurement, OTHER['x'], OTHER['u'],
                        OTHER['P'], robot_F_fn, robot_x_fn, *OTHER['matrices'])
    # OTHER['x_list'].append(x)
    OTHER['x'] = x
    OTHER['P'] = P
    # print('Trace of P : '+str(P.trace()))
    # Predict next position of robot
    next_state = robot_x_fn(x)
    est_xy = (next_state.value[0][0], next_state.value[1][0])

    # You must return xy_estimate (x, y), and OTHER (even if it is None)
    # in this order for grading purposes.
    # xy_estimate = (3.2, 9.1)
    # return z, OTHER
    return est_xy, OTHER

def calc_R(xc, yc, Z):
    """ calculate the distance of each 2D points from the center (xc, yc)
    z -> [[x1, y1], [x2, y2], ...]
    """
    return np.array([sqrt((x-xc)**2 + (y-yc)**2) for x, y in Z])

def f_1(c, Z=[]):
    R = calc_R(*c, Z)
    return R - R.mean()

def find_closest_circle_point(Z):
    # http://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
    est_center = Z.mean(axis=0)
    f_2 = partial(f_1, Z=Z)
    center_2 = optimize.leastsq(f_2, est_center)
    R_2 = calc_R(*center_2[0], Z).mean()
    circ_Z = np.empty_like(Z)
    for i, (x, y) in enumerate(Z):
        r_vec = np.array([x - center_2[0][0], y - center_2[0][1]])
        pt_on_circle = center_2[0] + r_vec/np.linalg.norm(r_vec) * R_2
        circ_Z[i,:] = pt_on_circle
    return circ_Z

target_bot = robot(2.1, 4.3, 0.5, 2*pi / 34.0, 1.5)
measurement_noise = 2.0 * target_bot.distance
target_bot.set_noise(0.0, 0.0, measurement_noise)

num = 300
Z = np.empty((num, 2))
pos = np.empty((num, 2))
est = np.empty((num, 2))
mavg = np.empty((num, 2))
circ_Z = np.empty_like(Z)
OTHER = None
N = 5
# weights = np.ones((N,))/N
weights = np.exp(np.linspace(-1., 0., N))/N
for i in range(num):
    measurement = target_bot.sense()
    target_bot.move_in_circle()

    Z[i,:] = np.array(measurement)
    pos[i,:] = np.array([target_bot.x, target_bot.y])

    mavg_temp_x = np.convolve(Z[:i+1,0], weights, mode='same')
    mavg_temp_y = np.convolve(Z[:i+1,1], weights, mode='same')

    # mavg_temp_x = ewma(Z[:i+1, 0], N)
    # mavg_temp_y = ewma(Z[:i+1, 1], N)
    if i > 2:
        circ_fit = find_closest_circle_point(Z[:i+1])
        med_circ_z = np.median(circ_fit[-5:,:],axis=0)
        # circ_Z[i,:] = circ_fit[-1]
        circ_Z[i,:] = med_circ_z

        # est_xy, OTHER = estimate_next_pos(med_circ_z, OTHER)
        # est_xy, OTHER = estimate_next_pos([mavg_temp_x[i], mavg_temp_y[i]], OTHER)
        #
    else:
        circ_z = measurement
        est_xy = measurement
        circ_Z[i,:] = circ_z
        # est_xy, OTHER = estimate_next_pos(measurement, OTHER)

    # est[i,:] = np.array(est_xy)
    mavg[i,:] = [mavg_temp_x[i], mavg_temp_y[i]]

for i, z in enumerate(circ_Z):
    if i>=30:
        est_xy, OTHER = estimate_next_pos(z, OTHER)
    else:
        est_xy = z
    est[i,:] = np.array(est_xy)
# mavg[:,0] = np.convolve(circ_Z[:,0], weights, mode='same')
# mavg[:,1] = np.convolve(circ_Z[:,1], weights, mode='same')

# est_center = Z.mean(axis=0)
# f_2 = partial(f_1, Z=Z)
# center_2 = optimize.leastsq(f_2, est_center)
# R_2 = calc_R(*center_2[0], Z).mean()
#
# for i, (x, y) in enumerate(Z):
#     r_vec = np.array([x - center_2[0][0], y - center_2[0][1]])
#     pt_on_circle = center_2[0] + r_vec/np.linalg.norm(r_vec) * R_2
#     circ_Z[i,:] = pt_on_circle

t = range(num)
plt.plot(t, pos[:,0], t, est[:,0], t, circ_Z[:,0])
plt.legend(['Position', 'Estimate', 'Circular Regression'])
plt.title('X-Position')

plt.figure()
plt.plot(t, pos[:,1], t, est[:,1], t, circ_Z[:,1])
plt.legend(['Position', 'Estimate', 'Circular Regression'])
plt.title('Y-Position')

plt.show(True)

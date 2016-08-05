from matrix import matrix
from utils import identity_matrix

def extended_kalman_filter(z, x, u, P, F_fn, x_fn, H, R):
    """
    Applies extended kalman filter on system

    z -> measurement
    x -> last state
    u -> control vector
    P -> covariances
    F_fn -> Function that returns F matrix for given 'x'
    x_fn -> Updates 'x' using the non-linear derivatives
    H -> Measurement matrix
    R -> Measurement covariance
    """
    I = identity_matrix(x.dimx)
    # prediction
    F = F_fn(x)
    x = x_fn(x) + u
    P = F * P * F.transpose()

    # measurement update
    Z = matrix([z])
    y = Z.transpose() - (H * x)
    S = H * P * H.transpose() + R
    K = P * H.transpose() * S.inverse()
    x = x + (K * y)
    P = (I - (K * H)) * P

    return x, P

import numpy as np

def rosenbrock(u):
    """
    m(u) = 10 (u2 - u1^2)^2 + (u1 - 1)^2
    u: array-like of shape (2,)
    """
    u1, u2 = u
    return 10 * (u2 - u1**2)**2 + (u1 - 1)**2


def grad_rosenbrock(u):
    """
    Gradient of m(u), shape (2,)
    """
    u1, u2 = u

    dm_du1 = -40 * u1 * (u2 - u1**2) + 2 * (u1 - 1)
    dm_du2 = 20 * (u2 - u1**2)

    return np.array([dm_du1, dm_du2])


def hessian_rosenbrock(u):
    """
    Hessian of m(u), shape (2, 2)
    """
    u1, u2 = u

    d2m_du1du1 = 120 * u1**2 - 40 * u2 + 2
    d2m_du1du2 = -40 * u1
    d2m_du2du2 = 20

    return np.array([
        [d2m_du1du1, d2m_du1du2],
        [d2m_du1du2, d2m_du2du2]
    ])

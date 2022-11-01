import numpy as np
from typing import List

# ------------------------------------------------------------- Cartesian -------------------------------------------------------------

"""
Cartesian -> Cylindrical
"""
# returns (rho, theta, phi)
def cartesian_to_cylindrical(
    x: float, y: float, z: float, to_deg: bool = False
) -> float:
    r"""
    :param x: x-coordinate
    :param y: y-coordinate
    :param z: z-coordinate
    :param to_deg: Asks whether the resultant theta should be in degrees
    """
    r = np.sqrt(x ** 2 + y ** 2)
    theta = tan_solution(x, y, r, to_deg=to_deg)
    return np.round((r, theta, z), 2)


"""
Cartesian -> Spherical
"""
# returns (rho,theta,phi)
def cartesian_to_spherical(x: float, y: float, z: float, to_deg: bool = False) -> float:
    r"""
    :param x: x-coordinate
    :param y: y-coordinate
    :param z: z-coordinate
    :param to_deg: Asks whether the resultant theta should be in degrees
    """
    rho = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arccos(z / rho)
    r = rho * np.sin(phi)
    theta = tan_solution(x, y, r, to_deg=to_deg)
    phi = np.degrees(phi) if to_deg else np.phi
    return np.round((rho, theta, phi), 2)


# ------------------------------------------------------------- Cylindrical -------------------------------------------------------------
"""
Cylindrical -> Cartesian
"""
# returns (x,y,z)
def cylindrical_to_cartesian(
    r: float, theta: float, z: float, from_deg: bool = False
) -> float:
    r"""
    :param r: arc length of the vector
    :param theta: angle about the z-axis
    :param z: z-coordinate
    :param from_deg: Asks whether the input is given in degrees
    """
    if from_deg:
        theta = np.radians(theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.round((x, y, z), 2)


"""
Cylindrical -> Cartesian
"""
# returns (rho,theta,phi)
def cylindrical_to_spherical(
    r: float, theta: float, z: float, from_deg: bool = False, to_deg: bool = False
) -> float:
    r"""
    :param r: arc length of the vector
    :param theta: angle about the z-axis
    :param z: z-coordinate
    :param from_deg: Asks whether the input is given in degrees
    :param to_deg: Asks whether the resultant theta should be in degrees
    """
    if from_deg:
        theta = np.radians(theta)
    rho = np.sqrt(r ** 2 + z ** 2)
    if to_deg:
        phi = np.degrees(np.arccos(z / rho))
        theta = np.degrees(theta)
    else:
        phi = np.arccos(z / rho)
    return np.round((rho, theta, phi), 2)


# ------------------------------------------------------------- Spherical -------------------------------------------------------------
"""
Spherical -> Cylindrical
"""
# returns (rho,theta,z)
def spherical_to_cylindrical(
    rho: float, theta: float, phi: float, from_deg: bool = False, to_deg: bool = False
) -> float:
    r"""
    :param rho: arc length of the vector
    :param theta: angle about the z-axis
    :param phi: angle down to z-axis going up or down
    :param from_deg: Asks whether the input is given in degrees
    :param to_deg: Asks whether the resultant theta should be in degrees
    """
    if from_deg:
        theta = np.radians(theta)
        phi = np.radians(phi)
    r = rho * np.sin(phi)
    z = rho * np.cos(phi)
    if to_deg:
        theta = np.degrees(theta)
    return np.round((r, theta, z), 2)


"""
Spherical -> Cylindrical
"""
# return (x,y,z)
def spherical_to_cartesian(
    rho: float, theta: float, phi: float, from_deg: bool = False
) -> float:
    r"""
    :param rho: arc length of the vector
    :param theta: angle about the z-axis
    :param phi: angle down to z-axis going up or down
    :param from_deg: Asks whether the input is given in degrees
    """
    if from_deg:
        theta = np.radians(theta)
        phi = np.radians(phi)

    x = rho * np.sin(phi) * np.cos(theta)
    y = rho * np.sin(phi) * np.sin(theta)
    z = rho * np.cos(phi)
    return np.round((x, y, z), 2)


# ------------------------------------------------------------- Trig Functions -------------------------------------------------------------


def tan_solution(x: float, y: float, r: float, to_deg: bool = False) -> float:
    r"""
    :param x: x-coordinate
    :param y: y-coordinate
    :param r: arc length of the vector
    :param to_deg: Asks whether the resultant theta should be in degrees
    """
    angle = np.arctan2(y, x)
    angle = np.degrees(angle) if to_deg else angle
    return angle

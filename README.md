# bug-free-waffle
Computer Graphics Rendering
import numpy as np
from scipy.interpolate import CubicSpline
import math

def smooth_attitude_interpolation(Cs, Cf, ωs, ωf, T):
    """
    Smoothly interpolates between two attitude matrices Cs and Cf with rotations in any axis.

    Args:
        Cs (numpy.ndarray): The initial attitude matrix (3x3).
        Cf (numpy.ndarray): The final attitude matrix (3x3).
        ωs (numpy.ndarray): The initial angular velocity vector (3x1).
        ωf (numpy.ndarray): The final angular velocity vector (3x1).
        T (float): The time interval between Cs and Cf.

    Returns:
        List[numpy.ndarray]: A list of attitude matrices interpolating between Cs and Cf.
    """
    if not np.allclose(np.linalg.inv(Cs) @ Cs, np.eye(3)):
        raise ValueError("Cs is not a valid attitude matrix.")
    if not np.allclose(np.linalg.inv(Cf) @ Cf, np.eye(3)):
        raise ValueError("Cf is not a valid attitude matrix.")

    t = np.linspace(0, T, 3)

    def rotation_vector(t):
        return np.log(Cs.T @ Cf)

    θ_poly = CubicSpline(t, rotation_vector(t), bc_type=((1, 0.0), (1, 0.0)))

    ω = θ_poly.derivative(nu=1)
    ω_̇ = θ_poly.derivative(nu=2)

    # Set jerk at endpoints to be equal
    ω_̇(t[0]) = ω_̇(t[-1])

    θ = np.array([ω(t_val) for t_val in t])

    # Fit a cubic spline to the time matrix
    t_poly = CubicSpline(t, np.exp(t), bc_type='not-a-knot')

    # Interpolate attitude matrices
    C = [Cs]
    for i in range(len(t)):
        C.append(Cs @ RY(2 * θ[i][0], axis=0) @ RX(2 * θ[i][1], axis=1) @ RZ(2 * θ[i][2], axis=2))

    return C

def RY(θ, axis=0):
    """
    Single-qubit Y-rotation gate around a specified axis.

    Args:
        θ (float): Rotation angle in radians.
        axis (int): Axis of rotation (0 for X, 1 for Y, 2 for Z).

    Returns:
        numpy.ndarray: Single-qubit Y-rotation gate.
    """
    if axis == 0:
        return np.array([[math.cos(θ / 2), -math.sin(θ / 2)],
                         [math.sin(θ / 2), math.cos(θ / 2)]])
    elif axis == 1:
        return np.array([[math.cos(θ / 2), -math.sin(θ / 2)],
                         [math.sin(θ / 2), math.cos(θ / 2)]])
    elif axis == 2:
        return np.array([[math.cos(θ / 2), -1j * math.sin(θ / 2)],
                         [1j * math.sin(θ / 2), math.cos(θ / 2)]])
    else:
        raise ValueError("Invalid axis for RY gate.")

def RX(θ, axis=1):
    """
    Single-qubit X-rotation gate around a specified axis.

    Args:
        θ (float): Rotation angle in radians.
        axis (int): Axis of rotation (0 for X, 1 for Y, 2 for Z).

    Returns:
        numpy.ndarray: Single-qubit X-rotation gate.
    """
    if axis == 0:
        return np.array([[math.cos(θ / 2), -1j * math.sin(θ / 2)],
                         [-1j * math.sin(θ / 2), math.cos(θ / 2)]])
    elif axis == 1:
        return np.array([[math.cos(θ / 2), -math.sin(θ / 2)],
                         [math.sin(θ / 2), math.cos(θ / 2)]])
    elif axis == 2:
        return np.array([[math.cos(θ / 2), -math.sin(θ / 2)],
                         [math.sin(θ / 2), math.cos(θ / 2)]])
    else:
        raise ValueError("Invalid axis for RX gate.")

def RZ(θ, axis=2):
    """
    Single-qubit Z-rotation gate around a specified axis.

    Args:
        θ (float): Rotation angle in radians.
        axis (int): Axis of rotation (0 for X, 1 for Y, 2 for Z).

    Returns:
        numpy.ndarray: Single-qubit Z-rotation gate.
    """
    if axis == 0:
        return np.diag([math.exp(-1j * θ / 2), math.exp(1j * θ / 2)])
    elif axis == 1:
        return np.array([[math.cos(θ / 2), -1j * math.sin(θ / 2)],
                         [1j * math.sin(θ / 2), math.cos(θ / 2)]])
    elif axis == 2:
        return np.array([[math.cos(θ / 2), -math.sin(θ / 2)],
                         [math.sin(θ / 2), math.cos(θ / 2)]])
    else:
        raise ValueError("Invalid axis for RZ gate.")

def apply_quantum_gates(C, qubits):
    """
    Applies the quantum gates in C to the qubits.

    Args:
        C (List[numpy.ndarray]): A list of quantum gates.
        qubits (object): A quantum computing object (e.g., Qiskit Qubits).
    """
    for gate in C:
        qubits.unitary(gate, qubits.qubits)
    return qubits

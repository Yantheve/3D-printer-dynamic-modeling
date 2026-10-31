#Modelling the dynamics of the core xy printer through a state-space description

import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

params = {
    "mu_x": None,  # bearing friction in x [kg/s]
    "mu_y": None,  # bearing friction in y [kg/s]
    "mu_r": 8.0,   # rolling friction [kg*m^2/s]
    "J_m": 9.0,    # motor inertia [kg*m^2]
    "k_m": None,   # motor torque/back-EMF constant [N*m/A = V*s/rad]
    "R": 300.0,    # resistance [ohm]
    "L": 0.5,      # inductance [H]
    "r": 1.0,      # motor radius [m]
    "m": 0.5,      # moving mass [kg]
    "a": 1,
    "b": 1,
    "omega": 1, 
    "omega_d": 1,
    "omega_dd": 1,
}

# Initial state, ordered as [theta_l, theta_r, theta_dot_l, theta_dot_r, i_l, i_r].
initial_conditions = {
    "theta_l": 0.0,
    "theta_r": 0.0,
    "theta_dot_l": 0.0,
    "theta_dot_r": 0.0,
    "i_l": 0.0,
    "i_r": 0.0,
    "x": 0.0,
    "y": 0.0,
    "x_dot": 0.0,
    "y_dot": 0.0,
}
initial_state = np.array([
    initial_conditions["theta_l"],
    initial_conditions["theta_r"],
    initial_conditions["theta_dot_l"],
    initial_conditions["theta_dot_r"],
    initial_conditions["i_l"],
    initial_conditions["i_r"],
], dtype=float)

# Motor input vector, ordered as [u_m_l, u_m_r] [V].
u = np.array([0.0, 0.0])


def state_space_matrices(params):
    """Build the state-space matrices from the physical model parameters."""
    missing = [name for name, value in params.items() if value is None]
    if missing:
        raise ValueError(f"Set values for: {', '.join(missing)}")

    mu_x = params["mu_x"]
    mu_y = params["mu_y"]
    mu_r = params["mu_r"]
    J_m = params["J_m"]
    k_m = params["k_m"]
    R = params["R"]
    L = params["L"]
    r = params["r"]
    m = params["m"]

    mu_diag = 4 * mu_r + (r**2 / 4) * (mu_x + mu_y)
    mu_cross = (r**2 / 4) * (mu_x - mu_y)
    J = J_m + (m * r**2) / 2

    A = np.array([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, -mu_diag / J, -mu_cross / J, k_m / J, 0],
        [0, 0, -mu_cross / J, -mu_diag / J, 0, k_m / J],
        [0, 0, -k_m / L, 0, -R / L, 0],
        [0, 0, 0, -k_m / L, 0, -R / L],
    ])
    B = np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1 / L, 0],
        [0, 1 / L],
    ])
    return A, B


def system_dynamics(t, state, u, params):
    """Return state derivatives for state and motor-voltage input vectors."""
    A, B = state_space_matrices(params)
    x_dot = A @ state + B @ u
    return x_dot

def state_space_matrices_x(params):
    """Build the state-space matrices of the x-directionfrom the physical model parameters."""
    missing = [name for name, value in params.items() if value is None]
    if missing:
        raise ValueError(f"Set values for: {', '.join(missing)}")

    mu_x = params["mu_x"]
    mu_y = params["mu_y"]
    mu_r = params["mu_r"]
    J_m = params["J_m"]
    k_m = params["k_m"]
    R = params["R"]
    L = params["L"]
    r = params["r"]
    m = params["m"]

    mu_diag = 4 * mu_r + (r**2 / 4) * (mu_x + mu_y)
    mu_cross = (r**2 / 4) * (mu_x - mu_y)
    J = J_m + (m * r**2) / 2

    Ax = np.array([
        [0, 1, 0],
        [0, -mu_x/J, k_m*r/J],
        [0, -k_m/(r*L), -R/L]
    ])
    Bx = np.array([
        [0],
        [0],
        [1/L]
    ])
    return Ax, Bx


def state_space_matrices_y(params):
    """Build the state-space matrices of the y-directionfrom the physical model parameters."""
    missing = [name for name, value in params.items() if value is None]
    if missing:
        raise ValueError(f"Set values for: {', '.join(missing)}")

    mu_x = params["mu_x"]
    mu_y = params["mu_y"]
    mu_r = params["mu_r"]
    J_m = params["J_m"]
    k_m = params["k_m"]
    R = params["R"]
    L = params["L"]
    r = params["r"]
    m = params["m"]

    mu_diag = 4 * mu_r + (r**2 / 4) * (mu_x + mu_y)
    mu_cross = (r**2 / 4) * (mu_x - mu_y)
    J = J_m + (m * r**2) / 2

    Ay = np.array([
        [0, 1, 0],
        [0, -mu_y/J, k_m*r/J],
        [0, -k_m/(r*L), -R/L]
    ])
    By = np.array([
        [0],
        [0],
        [1/L]
    ])
    return Ay, By

initial_state_x = np.array([initial_conditions["x"], initial_conditions["x_dot"], initial_conditions["i_l"]], dtype=float)
initial_state_y = np.array([initial_conditions["y"], initial_conditions["y_dot"], initial_conditions["i_r"]], dtype=float)

def sys(x, u):
    """Return state derivatives for state and motor-voltage input vectors."""
    Ax, Bx = state_space_matrices_x(params)
    Ay, By = state_space_matrices_y(params)
    x_dot = Ax @ x + Bx @ u[0]
    y_dot = Ay @ x + By @ u[1]
    return np.concatenate((x_dot, y_dot))

#Now include the input vector u that is supposed to draw an ellipse
u_ellipse = np.array([0.5*((J*L)/(k_m*r)*(a*omega**3*sin(omega*t) - b*omega**3*cos(omega*t)) - ((L*mu_x + J*R)/(k_m*r))*(a*omega**2*cos(omega*t) + b*omega**2*sin(omega*t)) + ((k_m**2 + mu_x*R)/(k_m*r))*(b*omega*cos(omega*t) - a*omega*sin(omega*t)))],
                     [0.5*((J*L)/(k_m*r)*(a*omega**3*cos(omega*t) + b*omega**3*sin(omega*t)) - ((L*mu_y + J*R)/(k_m*r))*(a*omega**2*sin(omega*t) - b*omega**2*cos(omega*t)) + ((k_m**2 + mu_y*R)/(k_m*r))*(b*omega*sin(omega*t) + a*omega*cos(omega*t)))])



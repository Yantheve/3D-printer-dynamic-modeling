import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.integrate import quad, solve_ivp
import matplotlib.pyplot as plt

#parameters
T_b = 378.15 #plate bed temperature [K]
T_e = 300 #environemnt temperature [K]
T_n = 513.15 #nozzle temperature [K]
#need to be defined
#a 
#rho 
#c_p 
#k 
#h_air
#h_b
#omega 
#M
#A_path
#B_path



#Q1.4 trajectory parameters
def printing_speed(t, A_path, B_path, omega):
    vx = -A_path * omega * np.sin(omega * t)
    vy = B_path * omega * np.cos(omega * t)
    return np.sqrt(vx**2 + vy**2)

#deposited length is the integral of the printing speed over time, up to the end of the trajectory
def deposited_length(t, A_path, B_path, omega):
    t_end = 2 * np.pi / omega
    upper_time = min(t, t_end)

    length, _ = quad(
        printing_speed,
        0,
        upper_time,
        args=(A_path, B_path, omega)
    )
    return length

#spatial grid, divided into M segments
def make_grid(A_path, B_path, omega, M):
    t_end = 2 * np.pi / omega
    L = deposited_length(t_end, A_path, B_path, omega)
    ds = L / M

    faces = np.linspace(0.0, L, M + 1)
    return t_end, L, ds, faces

#deposited cell length,during filing, the length takes filled; afterwards it doesnt exceed the cell width
def deposited_cell_lengths(length, faces):
    cell_widths = np.diff(faces)
    filled_lengths = np.clip(
        length - faces[:-1],
        0.0,
        cell_widths
    )
    return filled_lengths

#conductive heat transfer rate
def conductive_power(T_left, T_right, ell_left, ell_right, k, a):
    distance = (ell_left + ell_right) / 2
    return -k * a**2 * (T_right - T_left) / distance

#heat-transfer rate with the plate and air
def plate_air_power(T, ell, h_b, h_air, a, T_b, T_e):
    P_bed = h_b * a * ell * (T_b - T)
    P_air = 3 * h_air * a * ell * (T - T_e)
    return P_bed - P_air

#starting-end heat loss
def starting_end_loss(T1, ell1, h_air, k, a, T_e):
    return (
        h_air * a**2
        / (1 + h_air * ell1 / (2 * k))
        * (T1 - T_e)
    )

#conductive heat transfer rate at the deposition front
def front_conductive_power(T_last, ell_last, T_nozzle, k, a):
    return -k * a**2 * (T_nozzle - T_last) / (ell_last / 2)

#net heat input 
def printing_heat_inputs(T, ell, k, a, h_b, h_air,
                         T_b, T_e, T_nozzle):
    Q = plate_air_power(T, ell, h_b, h_air, a, T_b, T_e)

    for i in range(len(T) - 1):
        P = conductive_power(
            T[i], T[i + 1], ell[i], ell[i + 1], k, a
        )
        Q[i] -= P
        Q[i + 1] += P

    Q[0] -= starting_end_loss(
        T[0], ell[0], h_air, k, a, T_e
    )

    Q[-1] -= front_conductive_power(
        T[-1], ell[-1], T_nozzle, k, a
    )

    return Q

#temperature rate
def printing_temperature_rates(T, ell, vp, T_nozzle,
                               rho, cp, k, a,
                               h_b, h_air, T_b, T_e):
    Q = printing_heat_inputs(
        T, ell, k, a, h_b, h_air, T_b, T_e, T_nozzle
    )

    thermal_capacity = rho * cp * a**2 * ell
    dTdt = Q / thermal_capacity

    dTdt[-1] += vp / ell[-1] * (T_nozzle - T[-1])

    return dTdt

#post-printing temperature rates
def cooling_temperature_rates(T, ds, rho, cp, k, a,
                              h_b, h_air, T_b, T_e):
    T_left = np.roll(T, 1)
    T_right = np.roll(T, -1)

    conduction = (
        k / (rho * cp * ds**2)
        * (T_left - 2 * T + T_right)
    )

    plate_air = plate_air_power(
        T, ds, h_b, h_air, a, T_b, T_e
    )
    thermal_capacity = rho * cp * a**2 * ds

    return conduction + plate_air / thermal_capacity

#implicit startup temperature 
def implicit_startup(T_old, ell_end, dt, E_in, T_nozzle_end,
                     rho, cp, k, a, h_b, h_air, T_b, T_e):
    T_old = np.asarray(T_old, dtype=float)
    ell_end = np.asarray(ell_end, dtype=float)
    n = len(ell_end)

    if n != len(T_old) + 1 or np.any(ell_end <= 0) or dt <= 0:
        raise ValueError("Startup must add exactly one positive-length cell.")

    capacity = rho * cp * a**2 * ell_end

    def heat_inputs(T):
        return printing_heat_inputs(
            T, ell_end, k, a, h_b, h_air,
            T_b, T_e, T_nozzle_end
        )

    # Q = b + A @ (T - T_e)
    reference = np.full(n, T_e, dtype=float)
    b = heat_inputs(reference)
    A = np.empty((n, n))

    for j in range(n):
        trial = reference.copy()
        trial[j] += 1.0
        A[:, j] = heat_inputs(trial) - b

    # Initially, the new cell contains no material or energy.
    old_energy = np.zeros(n)
    old_energy[:-1] = capacity[:-1] * (T_old - T_e)

    incoming_energy = np.zeros(n)
    incoming_energy[-1] = E_in

    matrix = np.diag(capacity) - dt * A
    rhs = old_energy + incoming_energy + dt * b

    return T_e + np.linalg.solve(matrix, rhs)

#startup end time and incoming material energy
def startup_inputs(t_a, N_full, ds, t_end,
                   A_path, B_path, omega,
                   rho, cp, a, T_e, nozzle_temperature,
                   eta=1e-3):
    target_length = (N_full + eta) * ds

    def length_residual(t):
        return (
            deposited_length(t, A_path, B_path, omega)
            - target_length
        )

    t_b = brentq(length_residual, t_a, t_end)

    def incoming_power(t):
        vp = printing_speed(t, A_path, B_path, omega)
        return rho * a**2 * cp * vp * (nozzle_temperature(t) - T_e)

    E_in, _ = quad(incoming_power, t_a, t_b)

    ell_end = np.full(N_full + 1, ds, dtype=float)
    ell_end[-1] = (
        deposited_length(t_b, A_path, B_path, omega)
        - N_full * ds
    )

    return t_b, ell_end, E_in

#integrate until the cell is full
def integrate_until_full(t_start, T_start, N_full, ds, t_end,
                         A_path, B_path, omega,
                         rho, cp, k, a, h_b, h_air,
                         T_b, T_e, nozzle_temperature):

    target_length = (N_full + 1) * ds
    total_length = deposited_length(
        t_end, A_path, B_path, omega
    )

    # The last cell fills exactly at the end of printing.
    if np.isclose(target_length, total_length,
                  rtol=1e-12, atol=1e-14):
        t_full = t_end
    else:
        t_full = brentq(
            lambda t: deposited_length(
                t, A_path, B_path, omega
            ) - target_length,
            t_start,
            t_end
        )

    def rhs(t, T):
        ell = np.full(N_full + 1, ds, dtype=float)
        ell[-1] = (
            deposited_length(t, A_path, B_path, omega)
            - N_full * ds
        )

        vp = printing_speed(t, A_path, B_path, omega)

        return printing_temperature_rates(
            T, ell, vp, nozzle_temperature(t),
            rho, cp, k, a, h_b, h_air, T_b, T_e
        )

    solution = solve_ivp(
        rhs,
        (t_start, t_full),
        T_start,
        method="Radau",
        rtol=1e-9,
        atol=1e-10
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return solution

#printing loop
def simulate_printing(A_path, B_path, omega, M,
                      rho, cp, k, a, h_b, h_air,
                      T_b, T_e, nozzle_temperature,
                      eta=1e-3):
    t_end, L, ds, faces = make_grid(
        A_path, B_path, omega, M
    )

    t_a = 0.0
    T = np.array([], dtype=float)

    history_t = []
    history_T = []

    def save_state(t, temperatures):
        # NaN marks material that has not yet been deposited.
        row = np.full(M, np.nan)
        row[:len(temperatures)] = temperatures
        history_t.append(t)
        history_T.append(row)

    save_state(t_a, T)

    for N_full in range(M):
        # 1. Find the startup interval and incoming energy.
        t_b, ell_end, E_in = startup_inputs(
            t_a, N_full, ds, t_end,
            A_path, B_path, omega,
            rho, cp, a, T_e, nozzle_temperature,
            eta=eta
        )

        # 2. Update existing cells and initialise the new cell.
        T_start = implicit_startup(
            T, ell_end, t_b - t_a, E_in,
            nozzle_temperature(t_b),
            rho, cp, k, a, h_b, h_air, T_b, T_e
        )
        save_state(t_b, T_start)

        # 3. Continue until the new cell is full.
        solution = integrate_until_full(
            t_b, T_start, N_full, ds, t_end,
            A_path, B_path, omega,
            rho, cp, k, a, h_b, h_air,
            T_b, T_e, nozzle_temperature
        )

        for j in range(1, len(solution.t)):
            save_state(solution.t[j], solution.y[:, j])

        t_a = solution.t[-1]
        T = solution.y[:, -1].copy()

    return {
        "t": np.asarray(history_t),
        "T": np.asarray(history_T),
        "T_end": T,
        "t_end": t_end,
        "L": L,
        "ds": ds,
        "faces": faces
    }

#post-printing cooling loop
def simulate_cooling(print_result, rho, cp, k, a,
                     h_b, h_air, T_b, T_e,
                     max_cooling_time, tolerance=0.05):
    t_start = print_result["t_end"]
    T_start = print_result["T_end"].copy()
    ds = print_result["ds"]

    if max_cooling_time <= 0 or tolerance <= 0:
        raise ValueError("Cooling time and tolerance must be positive.")

    def uniformity_event(t, T):
        return np.max(np.abs(T - np.mean(T))) - tolerance

    uniformity_event.terminal = True
    uniformity_event.direction = -1

    # Printing may already end with a nearly uniform temperature.
    if uniformity_event(t_start, T_start) <= 0:
        return {
            "t": np.array([t_start]),
            "T": T_start[None, :],
            "reached_uniform": True,
            "t_uniform": t_start
        }

    def rhs(t, T):
        return cooling_temperature_rates(
            T, ds, rho, cp, k, a,
            h_b, h_air, T_b, T_e
        )

    solution = solve_ivp(
        rhs,
        (t_start, t_start + max_cooling_time),
        T_start,
        method="Radau",
        events=uniformity_event,
        rtol=1e-6,
        atol=1e-7
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    reached_uniform = solution.t_events[0].size > 0

    return {
        "t": solution.t,
        "T": solution.y.T,
        "reached_uniform": reached_uniform,
        "t_uniform": (
            solution.t_events[0][0]
            if reached_uniform else None
        )
    }

#test parameters
if __name__ == "__main__":
    # Test trajectory: ellipse with semi-axes 20 mm and 10 mm
    A_path = 0.020       # m
    B_path = 0.010       # m
    omega = 2 * np.pi / 20.0  # rad/s: one revolution in 20 s

    # Illustrative thermal parameters — for debugging only
    a = 0.0004          # Cross-sectional side length [m]
    rho = 1200.0        # Density [kg/m^3]
    cp = 1800.0         # Specific heat capacity [J/(kg K)]
    k = 0.20            # Thermal conductivity [W/(m K)]
    h_air = 10.0        # Air convection coefficient [W/(m^2 K)]
    h_b = 100.0         # Bed contact conductance [W/(m^2 K)]

    # All temperatures in kelvin
    T_e = 25.0 + 273.15
    T_b = 60.0 + 273.15

    def nozzle_temperature(t):
        return 200.0 + 273.15

    M = 80                  # Coarse grid for the first test
    eta = 1e-2              # Startup length / full cell length
    max_cooling_time = 300.0 # s after printing
    tolerance = 0.05         # K

    print_result = simulate_printing(
        A_path, B_path, omega, M,
        rho, cp, k, a, h_b, h_air,
        T_b, T_e, nozzle_temperature,
        eta=eta
    )

    print("Printing simulation completed.")
    print(f"Path length: {print_result['L']:.6f} m")
    print(f"Printing duration: {print_result['t_end']:.2f} s")
    print(
        "Temperature range at printing completion: "
        f"{print_result['T_end'].min() - 273.15:.2f} to "
        f"{print_result['T_end'].max() - 273.15:.2f} °C"
    )

    cooling_result = simulate_cooling(
        print_result, rho, cp, k, a,
        h_b, h_air, T_b, T_e,
        max_cooling_time=max_cooling_time,
        tolerance=tolerance
    )

    final_T = cooling_result["T"][-1]
    deviation = np.max(np.abs(final_T - np.mean(final_T)))

    print(f"Nearly uniform reached: {cooling_result['reached_uniform']}")
    print(f"Final mean temperature: {np.mean(final_T) - 273.15:.3f} °C")
    print(f"Maximum deviation from mean: {deviation:.6f} K")

    if cooling_result["reached_uniform"]:
        cooling_time = cooling_result["t_uniform"] - print_result["t_end"]
        print(f"Additional cooling time: {cooling_time:.3f} s")
    else:
        print("Uniformity criterion not reached within the time limit.")
        # Cell-centre positions along the path [mm]
    s_centres = (
        0.5 * (print_result["faces"][:-1] + print_result["faces"][1:])
        * 1000
    )

    cooling_t = cooling_result["t"] - print_result["t_end"]
    cooling_T = cooling_result["T"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # 1. Temperature distribution along the deposited path
    axes[0].plot(
        s_centres,
        print_result["T_end"] - 273.15,
        label="At printing completion"
    )
    axes[0].plot(
        s_centres,
        cooling_T[-1] - 273.15,
        label="At cooling simulation end"
    )
    axes[0].set_xlabel("Arc length s [mm]")
    axes[0].set_ylabel("Cell-average temperature [°C]")
    axes[0].set_title("Temperature along the path")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Mean temperature and spatial temperature range during cooling
    mean_T = np.mean(cooling_T, axis=1) - 273.15
    min_T = np.min(cooling_T, axis=1) - 273.15
    max_T = np.max(cooling_T, axis=1) - 273.15

    axes[1].fill_between(
        cooling_t, min_T, max_T,
        alpha=0.25, label="Minimum–maximum range"
    )
    axes[1].plot(cooling_t, mean_T, label="Mean temperature")
    axes[1].set_xlabel("Time after printing completion [s]")
    axes[1].set_ylabel("Temperature [°C]")
    axes[1].set_title("Post-printing cooling")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("q1_6_cooling_results.png", dpi=300)
    fig.savefig("q1_6_cooling_results.pdf")
    plt.show()    
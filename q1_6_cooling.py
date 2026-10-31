import numpy as np
from scipy.integrate import quad, solve_ivp
import matplotlib.pyplot as plt

#1. parameters and input
rho = 1200                    #filament density [kg/m^3]
c_p = 1800                    #[J/(kg*K)]
k=0.2                         #thermal conductivity [W/(m*K)]
a=0.0004                      #cross-sectional side length [m]
h_b = 100                     #[W/m^2*K]               
h_air=10                      #heat transfer coefficient of air[W/m^2*K] 
T_b=378.15                    #plate bed temperature [K]
T_e = 300                     #environment temperature  [K]

#1.1tragetory parameters
A_path = 0.020                # ellipse semi-axis [m]
B_path = 0.010                # ellipse semi-axis [m]
omega = 2 * np.pi / 20.0      # angular frequency [rad/s]

#1.2numerical paramters
M = 40                        #number of cells
rtol, atol = 1e-6, 1e-8
tol_uniform = 0.05            #uniform ttemperature tolerance [K]
max_cooling_time = 120.0      #longest cooling simulation [s]

#1.3external inouts
def T_nozzle (t):
    return 240+273.15                #nozzle temperature [K]
def v_p(t):
    vx = -A_path * omega *np.sin(omega *t)
    vy =  B_path * omega *np.cos(omega *t)
    return np.sqrt(vx**2 +vy**2)     #printing speed


#2.0grid and filament activate 
t_end = 2 * np.pi / omega            # printing finishes after one revolution

def deposited_length(t):
    upper_time = min(t, t_end)
    length, _ = quad(v_p, 0.0, upper_time)
    return length                    #the deposited length 
L = deposited_length(t_end)          #total path length [m]
ds = L/M                             #cell length [m]
l_eps = 1e-3 *ds                 #cell shorter than this value is not exist [m]
faces = np.linspace(0.0, L, M+1)     #places in the tragetory

#2.1deposited cell length of every cell
def deposited_cell_lengths(t):
    length = deposited_length(t)
    l = length - faces [:-1]
    l = np.clip (l, 0.0, ds)          # limit every length within 0 to ds
    return l               # take out active cells, leave empty cells     

#2.2 which cells exist / are full / are growing
def cell_status(l):
    active = l > l_eps
    full = l >= ds * (1 - 1e-9)
    idx = np.where( ~full)[0]
    growing = idx[0] if idx.size > 0 else None
    return active, full, growing

#3.0temperature derivative &dT/dt
# 3.1 Heat exchange with the plate and ambient air
def plate_air_power(T, l):
    P_plate = h_b * a * l * (T_b - T)   #heat conduction betwee the filament and the plate
    P_air = 3 * h_air * a * l * (T - T_e) #convective heat transfer from the filament to air
    return P_plate - P_air
# 3.2 Conductive heat transfer between adjacent deposited cells
def conductive_power(T_left, T_right, l_left, l_right):
    distance = (l_left + l_right) / 2          #the distance between two adjacent deposited cells
    P = -k * a**2 * (T_right - T_left) / distance #the conductive heat-transfer rate across the shared boundary
    return P
# 3.3 Heat loss through the starting end face
def starting_end_loss(T_first, l_first):
    P_end_out = h_air * a**2 * (T_first - T_e) / (
        1 + h_air * l_first / (2 * k)
    )            #the conductive heat transfer rate into the face equals the convective heat-transfer rate from the face to the air
    return P_end_out

#3.4 Energy input rate from incoming filament
def growing_cell ( v_p, T_nozzle, T_e) :
    P_growing = rho * a**2 * v_p * c_p * (T_nozzle - T_e)
    return P_growing
#3.5 (E->T) Energy derivatives during printing
def printing_rates(t, E):
    l = deposited_cell_lengths(t)
    active, full, growing = cell_status(l)

    T = np.full(M, T_nozzle(t))
    T[active] = T_e + E[active] / (rho * c_p * a**2 * l[active])

    Q = np.zeros(M)

    Q[active] += plate_air_power(T[active], l[active])

    for i in range(M - 1):
        if active[i] and active[i + 1]:
            P = conductive_power(T[i], T[i + 1], l[i], l[i + 1])
            Q[i] -= P
            Q[i + 1] += P

    if active[0]:
        Q[0] -= starting_end_loss(T[0], l[0])

    dE = Q.copy()
    if growing is not None:
        dE[growing] += growing_cell(v_p(t), T_nozzle(t), T_e)

    return dE

# 3.6 Energy derivatives after path closure
def cooling_rates(t, E):
    l = np.full(M, ds)
    T = T_e + E / (rho * c_p * a**2 * l)
    Q = np.zeros(M)

    Q += plate_air_power(T,l)

    for i in range(M - 1):
        P = conductive_power(T[i], T[i + 1], l[i], l[i + 1])
        Q[i] -= P
        Q[i + 1] += P

    P = conductive_power(T[-1], T[0], l[-1], l[0])
    Q[-1] -= P
    Q[0] += P
    dE = Q.copy()
    return dE

# 4printing and cooling integration
#4.1 initial condition
E0 = np.zeros(M)
#4.2 printing integration
print_sol = solve_ivp (printing_rates, (0.0, t_end), E0,
                       method = "Radau", rtol = rtol, atol = atol )
#4.3 state after printing
E_end = print_sol.y [:, -1].copy()
#4.4 stopping condition
def uniform(t,E):
    T = T_e + E / (rho * c_p * a**2 * ds)
    return np.max(abs(T-np.mean(T)))-tol_uniform   #the uniform temperature after printing
uniform.terminal = True
uniform.direction = -1
#4.5 cooling integration
cool_sol = solve_ivp(cooling_rates, (t_end, t_end+max_cooling_time),E_end,
                     method = "Radau", events = uniform, 
                     rtol = rtol, atol = atol)
#4.6 cooling time 
reached = cool_sol.t_events[0].size >0   #if it reaches uniform
if reached:
    cooling_time = cool_sol.t_events[0][0]-t_end
else:
    cooling_time = None

# 5 output
#5.1 temperature history during printing
T_print = np.full((M, len(print_sol.t)), np.nan)    # NaN = cell does not exist yet

for j, t in enumerate(print_sol.t):
    l = deposited_cell_lengths(t)         # cell lengths at THIS time
    active = l > 0.01 * ds                    # which cells exist at this time
    E_j = print_sol.y[:, j]               # energy of all cells at this time
    T_print[active, j] = T_e + E_j[active] / (rho * c_p * a**2 * l[active])            

#5.2 temperature history during cooling
T_cool = T_e + cool_sol.y / (rho * c_p * a**2 * ds)

#5.3 splice printing and cooling, maximum deviation
t_all = np.concatenate([print_sol.t, cool_sol.t])
T_all = np.hstack([T_print, T_cool])        # shape (M, number of time points)

T_mean = np.nanmean(T_all, axis = 0)    # mean over the cells at each time, ignoring NaN (axis=0)
dev = np.nanmax (np.abs(T_all-T_mean),axis = 0)           # max over the cells of |T - T_mean| at each time, ignoring NaN (axis=0)

#plot: temperature versus cooling time (how long it takes from t_end to uniform temperature)
T_min = np.nanmin (T_all, axis = 0 )
T_max = np.nanmax (T_all, axis = 0 )
T_inf = (3 * h_air * T_e + h_b * T_b ) / (3 * h_air + h_b)
t_uniform = t_end + cooling_time
plt.figure(figsize = (7, 4))
plt.fill_between (t_all,T_min, T_max, alpha = 0.3, label = "min-max of all cells")
plt.plot( t_all, T_mean, label = "mean")
plt.axvline (t_end, linestyle = ":", label = "end of printing")
plt.axvline (t_uniform, linestyle = "--", label = f"uniform: {cooling_time:.1f} s after printing")
plt.axhline(T_inf, linestyle = "--", color = "gray", label = f"T_inf = {T_inf:.1f} K")
plt.xlabel("Time [s]")
plt.ylabel("Temperature [K]")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("temperature_vs_time.png", dpi=200)
plt.show()


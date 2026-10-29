import numpy as np
from scipy.integrate import quad
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
l_eps = 1e-9                  #cell shorter than this value is not exist [m]


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
    idx = np.where(active & ~full)[0]
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
 # 5output
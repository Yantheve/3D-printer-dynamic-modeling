import numpy as np
from scipy.integrate import solve_ivp
#nozzle model
from nozzle_model import nozzle_rhs
#from motion_model

#parameters
params = {
    "L_n":
    "C_n":
    "C_t_n":
}
#FSM

#mode
P0, P1, P2, P3, P4, P5, P6 = range(7)


#time events
def make_time_events(t_target):
    def ev(t, state):
        return t-t_target
    ev.terminal = True
    ev.direction = 1
    return ev

#initial conditions
mode = P0
t0 = 0.0
y0 = np.array([T_n, i_n, v_c])
t_final = 
t_C = 
t_move = 
T_all, Y_all = [], []
u_heat = #heating voltage 

while t0<t_final:
    if mode == P0:#Printing
        u_n = u_heat
        events = [
            make_timer_events(t_C), 
            make_timer_events(t_F)
        ]
    elif mode == P1:#Moving to wipe tower
        u_n = u_heat
        events = [make_timer_events (t0+t_move)]

    elif mode == P2:#Wiping
            u_n = U_heat
            events = [make_timer_events(t0+t_wipe)]
    elif mode == P3:#Loading new filament
            u_n = u_heat
            events = [make_timer_events(t0+t_load)]
    elif mode == P4: #heating up 
           u_n = u_heat
           events = [make_timer_events(t0+t_heat)]
    elif mode == P5:#returning to saved position
           u_n =u_heat
           events = [make_timer_events(t0+t_move)]           

    sol = solve_ivp(
        lambda t, y: nozzle_rhs(t, y, u_n, params),
        (t0, t_final), Y0,
        method = "Radau", 
        events = events
    )

    if not sol.success:
        raise RuntimeError(sol.message)
    T_all.append(sol.t)
    Y_all.append(sol.y)

  
    #After integration:update the time and continuous state
    t0 = sol.t[-1]
    y0 = sol.y[:, -1].copy()
    #change mode when a terminal event has occurred
    if sol.status == 1:
         if mode == P0:
              mode = P1
        elif mode == P1:
         mode = P2
         
T = np.concatenate(T_all)
Y = np.concatenate(Y_all, axis = 1)

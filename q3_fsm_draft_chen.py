"""
State Vector (nozzle)
y = [T_n, i_n, v_c]
motion model need to fill in

MODES match the FSM figure in report Q2.1

TRANSITIONS:
P0->P1: C or F occurs               (timer, t_C/T_F)
P1->P2: arrive at wipe tower        (placeholer: timer t_move)
P2->p3: excess filament removed     (placeholer: timer t_wipe)
P3->p4: new filament loaded         (placeholer: timer t_load)
P4->P5: T_n reaches T_target        (temperature event)
P5->P0: saved position reached      (placeholer: timer t_move)

MOTION MODEL:
position
event: arrived at wipe tower   (replace timer P1)
event: saved position reached  (replace the timer in P5)


"""




import numpy as np
from scipy.integrate import solve_ivp
#nozzle model
from nozzle_model import nozzle_rhs
#from motion_model

#parameters
params = {
    "L_n": None,
    "C_n": None,
    "R_n": None,
    "C_t_n": None,
    "R_t_n": None,
    "T_e": None,
}
#FSM
#mode
P0, P1, P2, P3, P4, P5 = range(6)

#initial conditions
mode = P0
t0 = 0.0
T_n0 = 25+273.15   #initial nozzle temperature[K]
i_n0 = 0.0  #initial inductor current [A]
v_c0 = 0.0  #initial capacitor voltage [V]
y0 = np.array([T_n, i_n, v_c])

u_heat =None   #heating voltage 
u_boost =None  #voltage while heating up [V]
T_target =None #P4 ends when T_n reaches this [K] 
t_final = None
t_C = None
t_F = None
t_move = None
t_wipe =None
t_load =None
T_all, Y_all = [], []
mode_log = []

#time events
def make_time_events(t_target):
    def ev(t, state):
        return t-t_target
    ev.terminal = True
    ev.direction = 1
    return ev
#nozzle temperature event
def make_temperature_event(T_target):
    def ev(t, state):
        return state[0]-T_target
    ev.terminal = True
    ev.direction = 1
    return ev


while t0<t_final:
    if mode == P0:#Printing
        u_n = u_heat
        events = [
            make_time_events(t_C), 
            make_time_events(t_F)
        ]
    elif mode == P1:#Moving to wipe tower
        u_n = u_heat
        events = [make_time_events (t0+t_move)]

    elif mode == P2:#Wiping
            u_n =u_heat
            events = [make_time_events(t0+t_wipe)]
    elif mode == P3:#Loading new filament
            u_n = 0
            events = [make_time_events(t0+t_load)]
    elif mode == P4: #heating up 
           u_n = u_boost
           if y0[0] >= T_target:
                mode = P5
                continue
           
           events = [make_temperature_event(T_target)]

    elif mode == P5:#returning to saved position
           u_n =u_heat
           events = [make_time_events(t0+t_move)]           

    mode_log.append((t0,mode))
    sol = solve_ivp(
        lambda t, y: nozzle_rhs(t, y, u_n, params),
        (t0, t_final), y0,
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
         elif mode == P2:
              mode = P3
         elif mode == P3:
              mode = P4
         elif mode == P4:
              mode = P5
         elif mode == P5:
              mode = P0
        

         
T = np.concatenate(T_all)
Y = np.concatenate(Y_all, axis = 1)

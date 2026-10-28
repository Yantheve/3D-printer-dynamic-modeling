params = {
    "L_n": None, #inductance [H]
    "C_n": None, #capacitance [F]
    "R_n": 8, #heating resistance [Ω]
    "R_t_n": 9, #thermal resistance [K/W]
    "C_t_n": None, #thermal capacitance [J/K]
    "T_e": 300, #environmental temperature [K]
}
def nozzle_rhs(t, state, voltage, params):
    T_n, i_n, v_c = state

    L_n = params["L_n"]
    C_n = params["C_n"]
    R_n = params["R_n"]
    R_t_n = params["R_t_n"]
    C_t_n = params["C_t_n"]
    T_e = params["T_e"]

    T_n_dot = (1/C_t_n) * (
        v_c**2/R_n-(T_n-T_e)/R_t_n
        )
    i_n_dot = (voltage-v_c)/L_n
    v_c_dot =(i_n-v_c/R_n)/C_n

    return [T_n_dot, i_n_dot, v_c_dot]
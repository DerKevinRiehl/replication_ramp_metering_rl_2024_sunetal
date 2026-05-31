TS = 10.0                   # simulation time step [s]
TD = 60.0                   # DRL time step [s]
TC = 300.0                  # MPC time step [s]
T = 9000.0                  # Total simulation horizon [s]
T_INI = 600.0               # warm-up period [s]

N_INI = int(T_INI / TS)    # number of warm-up steps
N_CTRL = int(T / TS)        # number of simulation steps

# MPC parameters
NP = 60                # prediction horizon steps : prediction window 10 min -> 600s / 10s = 60 steps
NC = 2                 # control horizon steps : control window 5 min -> 300s / 600s = update 2 times in prediction window 
M_STEP = 30            # control input update interval : 300s (control step) / 10s = 30 steps

# HFMPC parameters
NC_HF = 10                 # control horizon steps : control window 10 min -> 600s / 60s = update 10 times in prediction window 
M_STEP_HF = 6            # control input update interval : 60s (control step) / 10s = 6 steps

# DDPG parameters
DDPG_STEP = 6.0         # DRL update every 60 second / 10s = 6 steps
STEPS_PER_EP = 150      # 2.5 hr / (10s/3600s) / 6 steps = 150 steps  [total simulation hr / simulation step in hr / DDPG steps]
SEED = 40
NUM_EPISODES = 3000     # total training episodes (paper: 3000)
BUFFER_SIZE = 200_000   # replay buffer capacity
BATCH_SIZE = 512        # mini-batch size
N_MULTI_START = 30      # MPC multi-start seeds (SLSQP)

# DDPG hyperparameters (Table IV of paper)
GAMMA = 0.99             # discount factor
TAU = 0.005              # soft target update rate
LR_ACTOR = 1e-4          # actor learning rate
LR_CRITIC = 1e-3         # critic learning rate
EXPLORATION_NOISE = 0.1  # Gaussian exploration noise std
STATE_DIM = 30           # DRL state dimension
ACTION_DIM = 3           # DRL action dimension
N_DEMAND_LOOKAHEAD = 5   # number of future demand steps in state

# Normalization constants for DRL state
RHO_MAX = 180.0          # max density [veh/km/lane]
V_MAX = 102.0            # free-flow speed [km/h]
W_MAX_MAIN = 200.0       # max mainstream queue [veh]
W_MAX_RAMP = 100.0       # max ramp queue [veh]
D_MAIN_MAX = 4000.0      # max mainstream demand [veh/h]
D_RAMP_MAX = 2000.0      # max ramp demand [veh/h]
U_VSL_MAX = 102.0        # max VSL [km/h]
U_RAMP_MAX = 1.0         # max ramp metering rate

METANET_PARAMS = {
    "real": {
        # Mainstream link (m = 1)
        "v_free_1": 102.0,          # km/h
        "rho_crit_1": 33.5,         # veh/km/lane
        "rho_max_1": 180.0,         # veh/km/lane
        "a_1": 1.867,
        "tau_1": 18.0 / 3600,       # h
        "eta_1": 60.0,              # km^2/h
        "kappa_1": 40.0,            # veh/km/lane
        "lambda_1": 2,              # lanes
        "alpha_1": 0.1,             # ramp metering parameter
        "sigma_1": 0.0122,          # ramp metering parameter

        # Capacity parameters
        "C_main": 4000.0,           # veh/h per both lanes (2 lanes)
        "queue_max_main": 200.0,    # veh (queue limit)
        "C_ramp": 2000.0,           # veh/h
        "queue_max_ramp": 100.0,    # veh
        "lambda_ramp": 1,           # lanes

        # Link lengths
        "L_main": 1.0,               # km per segment
    },

    "estimated": {
        # Mainstream link (m = 1)
        "v_free_1": 102.0,          # km/h
        "rho_crit_1": 37.5,         # veh/km/lane (estimated)
        "rho_max_1": 150.0,         # veh/km/lane
        "a_1": 2.160,
        "tau_1": 14.5 / 3600,       # h
        "eta_1": 50.0,              # km^2/h
        "kappa_1": 48.0,            # veh/km/lane
        "lambda_1": 2,              # lanes
        "alpha_1": 0.08,            # ramp metering parameter
        "sigma_1": 0.01,            # ramp metering parameter

        # Capacity parameters
        "C_main": 4000.0,           # veh/h per both lanes (2 lanes)
        "queue_max_main": 200.0,    # veh (queue limit)
        "C_ramp": 2000.0,           # veh/h
        "queue_max_ramp": 100.0,    # veh
        "lambda_ramp": 1,           # lanes

        # Link lengths
        "L_main": 0.8,              # km per segment
    }
}

GUSSIAN_NOISE = {
    "Low": {
        "main": {
            "mean": 0.0,
            "std": 75
        },
        "ramp": {
            "mean": 0.0,
            "std": 30
        }
    }, 
    "Medium": {
        "main": {
            "mean": 0.0,
            "std": 150
        },
        "ramp": {
            "mean": 0.0,
            "std": 60
        }
    }, 
    "High": {
        "main": {
            "mean": 0.0,
            "std": 225
        },
        "ramp": {
            "mean": 0.0,
            "std": 90
        }
    }
}
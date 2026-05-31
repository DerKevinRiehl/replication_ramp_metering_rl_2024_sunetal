import numpy as np
import casadi as cs
import sym_metanet as metanet
from config import METANET_PARAMS, TS, N_INI, GUSSIAN_NOISE, NP, NC, M_STEP, NC_HF, M_STEP_HF
from sym_metanet import (
    Destination,
    Link,
    LinkWithVsl,
    MainstreamOrigin,
    MeteredOnRamp,
    Network,
    Node,
    engines,
)


# ----------------------------------------------------------------------
# Warm-up phase
# ----------------------------------------------------------------------

def warmup_metanet(F: cs.Function, net: Network, metanet_param: dict, warmup_steps: int = N_INI) -> cs.DM:
    # Infer dimensions from network
    n_segments = sum(link.N for _, _, link in net.links)
    n_origins = len(net.origins)

    # Initial state: empty network, free-flow speed, zero queues
    rho0 = cs.DM.zeros(n_segments, 1)  # [veh/km/lane]
    v0 = cs.DM(metanet_param["v_free_1"] * np.ones((n_segments, 1)))  # [km/h]
    w0 = cs.DM.zeros(n_origins, 1)    # [veh]
    x = cs.vertcat(rho0, v0, w0)

    for k in range(warmup_steps):
        # F takes (x, u, d) as input -> [x, ramp metering, demands]
        u_warm = cs.DM([metanet_param["v_free_1"], metanet_param["v_free_1"], 1.0])       # [VSL1, VSL2, ramp]
        x_next, q = F(x, u_warm, cs.DM([3000, 500]))
        x = x_next
    
    return x  # return the warmed-up network state


# ----------------------------------------------------------------------
# Performance metrics
# ----------------------------------------------------------------------

def compute_metrics(RHO, W, V, metanet_param: dict):
    T_hr = TS / 3600  # convert TS to hours
    veh_in_network = (RHO * metanet_param["L_main"] * metanet_param["lambda_1"]).sum(axis=1)  # [veh]
    tts_mainline = T_hr * veh_in_network.sum()                      # [veh·h]

    queues_all = W.sum(axis=1)            # [veh]
    twt_all = T_hr * queues_all.sum()     # [veh·h]

    min_speeds = V.min(axis=1).min()      # [km/h]

    violation_main = np.maximum(0.0, W[:, 0] - metanet_param["queue_max_main"])       
    violation_ramp = np.maximum(0.0, W[:, 1] - metanet_param["queue_max_ramp"])
    
    #queue_violation = (violation_main * T_hr).sum() + (violation_ramp * T_hr).sum().   -> PREV VERSION

    # Ratio of maximal exceeded queue length w.r.t. maximum allowed (percentage)
    ratio_main = violation_main / metanet_param["queue_max_main"]
    ratio_ramp = violation_ramp / metanet_param["queue_max_ramp"]
    queue_violation = np.maximum(ratio_main, ratio_ramp).max() * 100.0

    return {
        "tts_all": tts_mainline+twt_all,
        "twt_all": twt_all,
        "min_speed": min_speeds,
        "queue_violation": queue_violation,
    }
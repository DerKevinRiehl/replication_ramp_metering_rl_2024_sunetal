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
# Network construction: 6 segments, ramp merges before segment 5
# ----------------------------------------------------------------------

# From the replicated paper Page 8 - Section IV A.Setup
# 6 km - 6 segments (4 + 2) with an on-ramp merging before segment 5
# We represent the 6 mainline segments as:
#   - L1: 4 segments  -> segments 1-4 # Segment 3 and 4 have VSLs
#   - L2: 2 segments  -> segments 5-6
#
# The on-ramp merges at the node between L1 and L2, so effectively at
# the upstream boundary of segment 5.
# ----------------------------------------------------------------------

def build_network(metanet_param: dict) -> Network:
    # Nodes
    N_up = Node(name="N_up")         # upstream mainline node
    N_merge = Node(name="N_merge")   # merge node (between seg 4 and 5)
    N_down = Node(name="N_down")     # downstream node

    # Origins and destination
    O_main = MainstreamOrigin[cs.SX](name="O_main")
    O_ramp = MeteredOnRamp[cs.SX](metanet_param["C_ramp"], name="O_ramp")  # on-ramp with queue
    D_down = Destination[cs.SX](name="D_down")

    # Mainline links: 4 + 2 = 6 segments total
    L1 = LinkWithVsl[cs.SX](
        4, metanet_param["lambda_1"], metanet_param["L_main"], 
        metanet_param["rho_max_1"], metanet_param["rho_crit_1"], 
        metanet_param["v_free_1"], metanet_param["a_1"], segments_with_vsl={2, 3}, alpha=0.1, name="L1"
    )
    L2 = Link[cs.SX](
        2, metanet_param["lambda_1"], metanet_param["L_main"], 
        metanet_param["rho_max_1"], metanet_param["rho_crit_1"], 
        metanet_param["v_free_1"], metanet_param["a_1"], name="L2"
    )

    # Build network
    net = (
        Network(name="freeway_network")
        # Path: O_main -> N_up -> L1 -> N_merge -> L2 -> N_down -> D_down
        .add_path(origin=O_main, path=(N_up, L1, N_merge, L2, N_down), destination=D_down)
        # Add on-ramp origin at the merge node (merges before segment 5)
        .add_origin(O_ramp, N_merge)
    )

    net.is_valid(raises=True)
    
    return net



# ----------------------------------------------------------------------
# Build CasADi dynamics function F
# ----------------------------------------------------------------------

def build_dynamics_function(net: Network, metanet_param: dict) -> cs.Function:
    engines.use("casadi", sym_type="SX")

    # One symbolic step to instantiate the METANET dynamics
    net.step(
        T=TS/3600,
        tau=metanet_param["tau_1"],
        eta=metanet_param["eta_1"],
        kappa=metanet_param["kappa_1"],
        delta=metanet_param["sigma_1"],
        # Initial condition tweak: allow high upstream controlled speed
        init_conditions={net.origins_by_name["O_main"]: {"v_ctrl": metanet_param["v_free_1"] * 2}},
    )

    # Build CasADi function:
    #   F(x, u, d) -> (x_next, q_all)
    # where x = [rho(6); v(6); w(2)] here.
    F = metanet.engine.to_function(
        net=net,
        T=TS/3600,
        more_out=True,
        compact=2,
    )
    return F

"""
MPC Controller using csnlp + CasADi IPOPT solver.

Adapted from the previous working implementation to fit the Controller ABC.
Key differences from the previous version:
 - Implements ``Controller`` interface (get_action / reset / update_interval)
 - Accesses CasADi state via ``env.x`` (env now does 1 T_s step per call)
 - Accepts separate ``mpc_params`` for cost function (model mismatch)

The controller is generic: by changing Nc and M_step it serves as both
Low-Frequency MPC (Nc=2, M_step=30) and High-Frequency MPC (Nc=10, M_step=6).
"""

import time
import numpy as np
import casadi as cs
from csnlp import Nlp
from csnlp.wrappers import Mpc

from controllers.base_controller import Controller
from config import TS, NP, NC, M_STEP


class MPCController(Controller):
    """
    MPC controller with 2 VSL segments + 1 ramp metering action.

    Uses csnlp (CasADi NLP) with IPOPT solver, warm-starting from
    the previous solution, and optional multi-start.

    Parameters
    ----------
    env : MetanetEnv
        Environment instance (provides F, net, state).
    demand : np.ndarray, shape (n_steps, 2)
        Base (noise-free) demand profile for forecasting.
    mpc_params : dict
        Physical parameters for cost function (can differ from env for
        model mismatch, e.g. METANET_PARAMS["estimated"]).
    Np : int
        Prediction horizon in simulation steps (default 60 = 10 min).
    Nc : int
        Number of control moves in horizon (LF: 2, HF: 10).
    M_step : int
        Simulation steps per control move (LF: 30, HF: 6).
    w_tts : float
        Weight on TTS objective (default 1.0).
    w_du : float
        Weight on control smoothness penalty (default 0.4).
    num_starts : int
        Number of multi-start initialisations (default 1).
    verbose : bool
        Print solver progress.
    """

    def __init__(
        self,
        env,
        demand: np.ndarray,
        mpc_params: dict,
        Np: int = NP,
        Nc: int = NC,
        M_step: int = M_STEP,
        w_tts: float = 1.0,
        w_du: float = 0.4,
        num_starts: int = 1,
        verbose: bool = False,
    ):
        self.env = env
        self.demand = np.asarray(demand)
        self.mpc_params = mpc_params
        self.Np = Np
        self.Nc = Nc
        self.M_step = M_step
        self.w_tts = w_tts
        self.w_du = w_du
        self.num_starts = num_starts
        self.verbose = verbose
        self.solve_times: list[float] = []

        # Network dimensions
        n_seg = env.n_segments
        n_orig = env.n_origins
        self.n_seg = n_seg
        self.n_orig = n_orig

        # Cost parameters from mpc_params (may be "estimated")
        T = TS / 3600.0
        L = mpc_params["L_main"]
        lanes = mpc_params["lambda_1"]
        v_free = mpc_params["v_free_1"]
        queue_max_ramp = mpc_params["queue_max_ramp"]
        self.v_free = v_free

        # ------------------------------------------------------------------
        #  Build csnlp MPC
        # ------------------------------------------------------------------
        mpc = Mpc[cs.SX](
            nlp=Nlp[cs.SX](sym_type="SX"),
            prediction_horizon=self.Np,                # 60 sim steps
            control_horizon=self.Nc * self.M_step,     # LF: 2*30=60
            input_spacing=self.M_step,                 # LF: 30
        )

        # States
        rho, _ = mpc.state("rho", n_seg, lb=0)
        v, _   = mpc.state("v", n_seg, lb=0)
        w, _   = mpc.state(
            "w", n_orig, lb=0,
            ub=[[np.inf], [queue_max_ramp]],
        )

        # Actions: 2 VSL segments + 1 ramp metering
        vsl, _ = mpc.action("vsl", 2, lb=20, ub=v_free)
        r, _   = mpc.action("r", lb=0, ub=1)

        # Disturbances
        d = mpc.disturbance("d", n_orig)

        # Nonlinear dynamics: F(x, u, d) → (x_next, q_all)
        F = env.F
        mpc.set_nonlinear_dynamics(lambda x, u_, d_: F(x, u_, d_)[0])

        # Parameters for smoothness penalty (previous applied control)
        v_ctrl_last = mpc.parameter("v_ctrl_last", (vsl.size1(), 1))
        r_last = mpc.parameter("r_last", (r.size1(), 1))

        # Objective:  w_TTS · TTS  +  w_Δ · smoothness
        # TTS over prediction horizon
        # Smoothness: VSL normalised by v_free, ramp in [0,1] directly
        mpc.minimize(
            w_tts * T * cs.sum2(cs.sum1(rho * L * lanes) + cs.sum1(w))
            + w_du * cs.sumsqr(
                cs.diff(cs.horzcat(v_ctrl_last, vsl), 1, 1) / v_free
            )
            + w_du * cs.sumsqr(cs.diff(cs.horzcat(r_last, r)))
        )

        # IPOPT solver options
        opts = {
            "expand": True,
            "print_time": False,
            "ipopt": {
                "max_iter": 1000,
                "sb": "yes",
                "print_level": 0,
                "tol": 1e-2,
                "constr_viol_tol": 1e-2,
                "compl_inf_tol": 1e-2,
                "acceptable_tol": 1e-2,
                "acceptable_constr_viol_tol": 1e-2,
            },
        }
        
        mpc.init_solver(solver="ipopt", opts=opts)
        self.mpc = mpc

        # ------------------------------------------------------------------
        #  Internal memory
        # ------------------------------------------------------------------
        self.u_last = cs.DM([v_free, v_free, 1.0]).reshape((3, 1))
        self.sol_prev = None
        self.u_current = np.array([v_free, v_free, 1.0], dtype=float)

    # ------------------------------------------------------------------
    #  Controller ABC interface
    # ------------------------------------------------------------------
    @property
    def update_interval(self) -> int:
        return self.M_step

    def reset(self) -> None:
        v_free = self.v_free
        self.u_last = cs.DM([v_free, v_free, 1.0]).reshape((3, 1))
        self.sol_prev = None
        self.u_current = np.array([v_free, v_free, 1.0], dtype=float)
        self.solve_times.clear()

    def get_action(self, obs: np.ndarray, step_idx: int) -> np.ndarray:
        """
        Return the MPC action for the current simulation step.

        Re-solves only at ``step_idx % M_step == 0``; otherwise returns
        the cached action from the last solve.
        """
        if step_idx % self.M_step != 0:
            return self.u_current.copy()

        # --- Get CasADi state from env ---
        x = self.env.x
        rho, v, w = cs.vertsplit(
            x,
            (0, self.n_seg, 2 * self.n_seg, 2 * self.n_seg + self.n_orig),
        )

        # --- Demand forecast (noiseless, from base profile) ---
        d_hat = self.demand[step_idx : step_idx + self.Np, :]
        if d_hat.shape[0] < self.Np:
            d_hat = np.pad(
                d_hat,
                ((0, self.Np - d_hat.shape[0]), (0, 0)),
                mode="edge",
            )

        # --- Solve NLP (with optional multi-start) ---
        t0 = time.perf_counter()

        best_sol = None
        best_cost = None

        for j in range(self.num_starts):
            # Initial guess
            if j == 0 and self.sol_prev is not None:
                vals0 = self.sol_prev
            else:
                vals0 = self._make_initial_guess()

            sol_j = self.mpc.solve(
                pars={
                    "rho_0": rho,
                    "v_0": v,
                    "w_0": w,
                    "d": d_hat.T,
                    "v_ctrl_last": self.u_last[:2],
                    "r_last": self.u_last[2:],
                },
                vals0=vals0,
            )

            J_j = float(sol_j.f)
            if (best_cost is None) or (J_j < best_cost):
                best_cost = J_j
                best_sol = sol_j

        dt = time.perf_counter() - t0
        self.solve_times.append(dt)

        # --- Extract first control move ---
        if best_sol is not None:
            self.sol_prev = best_sol.vals

            vsl_seq = best_sol.vals["vsl"]   # shape (2, Nc)
            r_seq   = best_sol.vals["r"]     # shape (1, Nc)

            vsl0 = np.array(vsl_seq[:, 0]).reshape(-1)   # [vsl1, vsl2]
            r0   = float(r_seq[0])                       # first ramp rate

            u0 = np.array([vsl0[0], vsl0[1], r0], dtype=float)
            self.u_current = u0
            self.u_last = cs.DM(u0).reshape((3, 1))

        if self.verbose:
            print(
                f"  MPC step {step_idx:4d} | "
                f"J* = {best_cost:.4f} | "
                f"u = [{self.u_current[0]:.1f}, "
                f"{self.u_current[1]:.1f}, "
                f"{self.u_current[2]:.3f}] | "
                f"solve: {dt:.3f}s"
            )

        return self.u_current.copy()

    # ------------------------------------------------------------------
    #  Multi-start initial guess
    # ------------------------------------------------------------------
    def _make_initial_guess(self):
        """
        Build a perturbed initial guess from the previous solution.
        If no previous solution exists, let csnlp use its default.
        """
        if self.sol_prev is None:
            return None

        vals0 = dict(self.sol_prev)

        # Perturb VSL
        if "vsl" in vals0:
            vsl_np = np.array(vals0["vsl"], dtype=float)
            noise_vsl = 2.0 * np.random.randn(*vsl_np.shape)
            vals0["vsl"] = cs.DM(
                np.clip(vsl_np + noise_vsl, 20.0, self.v_free)
            )

        # Perturb ramp metering
        if "r" in vals0:
            r_np = np.array(vals0["r"], dtype=float)
            noise_r = 0.05 * np.random.randn(*r_np.shape)
            vals0["r"] = cs.DM(np.clip(r_np + noise_r, 0.0, 1.0))

        return vals0
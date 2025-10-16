# -*- coding: utf-8 -*-
# Cubic polynomial joint-space trajectory (rest-to-rest) with accel limit check


import math
import random
import numpy as np

PI = math.pi
DEG = PI/180.0


JOINT_LIMITS_DEG = [
    (-170, 170),   # J1
    (-120, 120),   # J2
    (-170, 170),   # J3
    (-190, 190),   # J4
    (-120, 120),   # J5
    (-360, 360),   # J6
]


A_MAX = np.array([3.0, 3.0, 3.5, 4.0, 4.0, 5.0], dtype=float)
T_DESIRED = 2.5
DT = 0.01


def rand_joint_vector():

    q = []
    for lo, hi in JOINT_LIMITS_DEG:
        q.append(random.uniform(lo, hi)*DEG)
    return np.array(q, dtype=float)

def cubic_time_scaling(T, t):

    tau = t / T
    s    = 3*tau**2 - 2*tau**3
    sdot = (6.0/T)*tau*(1.0 - tau)        # = 6 t / T^2 - 6 t^2 / T^3
    sdd  = (6.0/(T*T))*(1.0 - 2.0*tau)    
    return s, sdot, sdd

def enforce_accel_limit(q0, qf, a_max, T_desired):

    dq = np.abs(qf - q0)

    with np.errstate(divide='ignore', invalid='ignore'):
        Tj = np.sqrt(np.where(a_max>0.0, 6.0*dq/np.maximum(a_max, 1e-12), 0.0))
        Tj[np.isnan(Tj)] = 0.0
    T_eff = max(T_desired, float(np.max(Tj)))
    return T_eff

def plan_cubic_trajectory(q0, qf, a_max, T_desired, dt):

    T = enforce_accel_limit(q0, qf, a_max, T_desired)
    N = max(1, int(math.ceil(T/dt)))
    tgrid = np.linspace(0.0, T, N+1)
    dq = (qf - q0)

    q   = np.zeros((N+1, 6))
    qd  = np.zeros((N+1, 6))
    qdd = np.zeros((N+1, 6))

    for k, t in enumerate(tgrid):
        s, sdot, sdd = cubic_time_scaling(T, t)
        q[k, :]   = q0 + dq * s
        qd[k, :]  = dq * sdot
        qdd[k, :] = dq * sdd

    return tgrid, q, qd, qdd, T

def sysCall_init():
    global sim, joint_h, _traj_list, _case_idx, _cases
    sim = require("sim")
    
    CASES_DEG = [
        #1
        ([-30, -45,  60,   0,  30,   0],   [ 30,  20, -40,  90, -30,  45]),
        #2
        ([ 10, -20,  15, -90,  45, 120],   [-60,  60,  80,   0, -45, -90]),
        #3
        ([  0,   0,   0,   0,   0,   0],   [ 90, -30,  30,  60,  15, -45]),
    ]
    _cases = []
    for q0_deg, qf_deg in CASES_DEG:
        q0 = np.array(q0_deg, dtype=float) * DEG
        qf = np.array(qf_deg, dtype=float) * DEG
        _cases.append((q0, qf))
    # -------------------------------------------------------

    JOINT_PATHS = [
        "/UR5/joint",
        "/UR5/joint/link/joint",
        "/UR5/joint/link/joint/link/joint",
        "/UR5/joint/link/joint/link/joint/link/joint",
        "/UR5/joint/link/joint/link/joint/link/joint/link/joint",
        "/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint",
    ]
    joint_h = [sim.getObject(path) for path in JOINT_PATHS]
    for j in range(6):
        sim.setJointMode(joint_h[j], sim.jointmode_force, 0)

    _traj_list = []
    for idx, (q0, qf) in enumerate(_cases):
        tgrid, Q, Qd, Qdd, T_eff = plan_cubic_trajectory(q0, qf, A_MAX, T_DESIRED, DT)
        _traj_list.append((tgrid, Q, Qd, Qdd, T_eff))
        sim.addLog(sim.verbosity_scriptinfos,
            f"[Cubic] Case {idx+1}: T_eff={T_eff:.3f}s, steps={len(tgrid)}")

    _case_idx = 0

def sysCall_thread():
    global sim, joint_h, _traj_list, _case_idx
    PAUSE_BETWEEN = 0.8  

    total_cases = len(_traj_list)
    for ci in range(total_cases):
        tgrid, Q, Qd, Qdd, T_eff = _traj_list[ci]
        sim.addLog(sim.verbosity_scriptinfos, f"[Cubic] Running case {ci+1}/{total_cases}")
        for k in range(len(tgrid)):
            qk = Q[k, :]
            for j in range(6):
                sim.setJointTargetPosition(joint_h[j], float(qk[j]))
            sim.wait(DT)

        for j in range(6):
            sim.setJointTargetPosition(joint_h[j], float(Q[-1, j]))
        sim.wait(PAUSE_BETWEEN)

    sim.addLog(sim.verbosity_scriptinfos, "[Cubic] All 3 cases done. Holding last pose.")
    return
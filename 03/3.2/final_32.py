# -*- coding: utf-8 -*-
# LIPB (Trapezoidal Velocity) joint-space trajectory for 6-DOF robot in CoppeliaSim


import math
import random
import numpy as np
import time

PI  = math.pi
DEG = PI/180.0


JOINT_PATHS = [
    "/UR5/joint",
    "/UR5/joint/link/joint",
    "/UR5/joint/link/joint/link/joint",
    "/UR5/joint/link/joint/link/joint/link/joint",
    "/UR5/joint/link/joint/link/joint/link/joint/link/joint",
    "/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint",
]

JOINT_LIMITS_DEG = [
    (-170, 170),
    (-120, 120),
    (-170, 170),
    (-190, 190),
    (-120, 120),
    (-360, 360),
]

A_MAX = np.array([3.0, 3.0, 3.5, 4.0, 4.0, 5.0], dtype=float) 
T_DESIRED = 2.0  
DT = 0.01        
PAUSE_BETWEEN = 0.8  


RANDOM_THREE = False
CASES_DEG = [
    # case 1
    ([-30, -45,  60,   0,  30,   0],   [ 30,  20, -40,  90, -30,  45]),
    # case 2
    ([ 10, -20,  15, -90,  45, 120],   [-60,  60,  80,   0, -45, -90]),
    # case 3
    ([  0,   0,   0,   0,   0,   0],   [ 90, -30,  30,  60,  15, -45]),
]
# --------------------------------

def rand_joint_vector():
    q = []
    for lo, hi in JOINT_LIMITS_DEG:
        q.append(random.uniform(lo, hi)*DEG)
    return np.array(q, dtype=float)

def feasible_time_lipb_per_joint(dq_abs, a):

    with np.errstate(invalid='ignore'):
        Tmin = 2.0*np.sqrt(np.where(a>0.0, dq_abs/np.maximum(a,1e-12), 0.0))
        Tmin[np.isnan(Tmin)] = 0.0
    return Tmin

def solve_trapezoid_params(dq_abs, a, T):

    eps = 1e-12
    with np.errstate(invalid='ignore'):
        disc = 1.0 - 4.0*dq_abs/(np.maximum(a,eps)*np.maximum(T,eps)**2)
        disc = np.clip(disc, 0.0, 1.0)
        v = 0.5*a*T*(1.0 - np.sqrt(disc))
    tb = v/np.maximum(a, eps)
    tc = T - 2.0*tb
    tb = np.maximum(tb, 0.0)
    tc = np.maximum(tc, 0.0)
    return v, tb, tc

def plan_lipb_trajectory(q0, qf, a_max, T_desired, dt):
    dq   = qf - q0
    sgn  = np.sign(dq)
    dq_a = np.abs(dq)

    Tmin_j = feasible_time_lipb_per_joint(dq_a, a_max)
    T_eff  = max(T_desired, float(np.max(Tmin_j)))

    v, tb, tc = solve_trapezoid_params(dq_a, a_max, T_eff)

    N = max(1, int(math.ceil(T_eff/dt)))
    tgrid = np.linspace(0.0, T_eff, N+1)

    Q   = np.zeros((N+1, 6))
    Qd  = np.zeros((N+1, 6))
    Qdd = np.zeros((N+1, 6))

    a_dir = sgn * a_max

    for k, t in enumerate(tgrid):
        qk  = np.zeros(6)
        qdk = np.zeros(6)
        qak = np.zeros(6)

        # accel
        m1 = (t < tb)
        t1 = np.where(m1, t, 0.0)
        qk[m1]  = q0[m1] + 0.5*a_dir[m1]*t1[m1]*t1[m1]
        qdk[m1] = a_dir[m1]*t1[m1]
        qak[m1] = a_dir[m1]

        # cruise
        m2 = (t >= tb) & (t < (tb + tc))
        t2 = np.where(m2, t - tb, 0.0)
        qk[m2]  = q0[m2] + 0.5*a_dir[m2]*tb[m2]*tb[m2] + sgn[m2]*v[m2]*t2[m2]
        qdk[m2] = sgn[m2]*v[m2]
        qak[m2] = 0.0

        # decel
        m3 = (t >= (tb + tc))
        t3 = np.where(m3, t - (tb + tc), 0.0)
        qk[m3]  = qf[m3] - 0.5*a_max[m3]*(tb[m3] - t3[m3])**2 * sgn[m3]
        qdk[m3] = a_dir[m3]*(tb[m3] - t3[m3]) * (-1.0)
        qak[m3] = -a_dir[m3]

        Q[k, :]   = qk
        Qd[k, :]  = qdk
        Qdd[k, :] = qak

    Q[-1, :] = qf
    Qd[-1, :] = 0.0
    Qdd[-1, :] = 0.0

    return tgrid, Q, Qd, Qdd, T_eff, (v, tb, tc)

# ---------- CoppeliaSim callbacks ----------
def sysCall_init():
    global sim, joint_h, _traj_list
    sim = require("sim")


    cases = []
    if RANDOM_THREE:
        for _ in range(3):
            cases.append((rand_joint_vector(), rand_joint_vector()))
    else:
        for q0_deg, qf_deg in CASES_DEG:
            q0 = np.array(q0_deg, dtype=float) * DEG
            qf = np.array(qf_deg, dtype=float) * DEG
            cases.append((q0, qf))


    joint_h = [sim.getObject(p) for p in JOINT_PATHS]
    for j in range(6):
        sim.setJointMode(joint_h[j], sim.jointmode_force, 0)


    _traj_list = []  
    for idx, (q0, qf) in enumerate(cases, start=1):
        tgrid, Q, Qd, Qdd, T_eff, params = plan_lipb_trajectory(q0, qf, A_MAX, T_DESIRED, DT)
        v, tb, tc = params
        _traj_list.append((tgrid, Q, Qd, Qdd, T_eff, params))
        sim.addLog(sim.verbosity_scriptinfos,
            f"[LIPB] Case {idx}: T_desired={T_DESIRED:.3f}s, T_eff={T_eff:.3f}s, steps={len(tgrid)}")
        for j in range(6):
            dqj = Q[-1, j] - Q[0, j]
            sim.addLog(sim.verbosity_scriptinfos,
                f"[LIPB]   J{j+1}: dq={dqj:+.3f} rad, amax={A_MAX[j]:.2f}, "
                f"tb={tb[j]:.3f}, tc={tc[j]:.3f}, v={v[j]:.3f}")

def sysCall_thread():
    global sim, joint_h, _traj_list
    total = len(_traj_list)
    for ci, (tgrid, Q, Qd, Qdd, T_eff, params) in enumerate(_traj_list, start=1):
        sim.addLog(sim.verbosity_scriptinfos, f"[LIPB] Running case {ci}/{total}")
        for k in range(len(tgrid)):
            qk = Q[k, :]
            for j in range(6):
                sim.setJointTargetPosition(joint_h[j], float(qk[j]))
            sim.wait(DT)

        # hold final pose & pause
        for j in range(6):
            sim.setJointTargetPosition(joint_h[j], float(Q[-1, j]))
        sim.wait(PAUSE_BETWEEN)

    sim.addLog(sim.verbosity_scriptinfos, "[LIPB] All 3 cases done. Holding last pose.")
    return

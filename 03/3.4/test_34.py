# -*- coding: utf-8 -*-
# Task-space Linear Interpolation with Parabolic Blend (LIPB)
# - Start/End linear velocity = 0 (rest-to-rest)
# - Orientation fixed throughout the motion
# - Per-run random EE start/end positions within a box
# - Enforce max linear acceleration; auto-stretch T if needed
# - IK via Jacobian (6x6) with Damped Least Squares (DLS)

import math, time, random
import numpy as np

PI, DEG = math.pi, math.pi/180.0


JOINT_PATHS = [
    "/UR5/joint",
    "/UR5/joint/link/joint",
    "/UR5/joint/link/joint/link/joint",
    "/UR5/joint/link/joint/link/joint/link/joint",
    "/UR5/joint/link/joint/link/joint/link/joint/link/joint",
    "/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint",
]


BOX_X = (0.25, 0.65)
BOX_Y = (-0.35, 0.35)
BOX_Z = (0.18, 0.55)

A_LIN_MAX = 0.8    
T_DESIRED  = 2.0   
DT         = 0.05  
REALTIME   = True  

# IK parameters
DAMPING  = 1e-4      
Wpos, Wang = 1.0, 0.25   
DQ_MAX   = 1.0*DEG   
POS_TOL  = 1e-3      
PREP_MAX_ITERS = 800


UR5_DH = np.array([
    [-PI/2,   0.0000, 0.0892,  0.0],
    [ 0.0,    0.4251, 0.0000,  0.0],
    [ 0.0,    0.39215,0.0000,  0.0],
    [-PI/2,   0.0000, 0.1090,  0.0],
    [ PI/2,   0.0000, 0.09475, 0.0],
    [ 0.0,    0.0000, 0.0825,  0.0],
], dtype=float)

def dh_transform(alpha, a, d, theta):
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([
        [ct, -st, 0,  a],
        [st*ca, ct*ca, -sa, -sa*d],
        [st*sa, ct*sa,  ca,  ca*d],
        [0, 0, 0, 1]
    ], dtype=float)

def fk_and_frames(q):

    T = np.eye(4)
    p = [T[:3,3].copy()]
    z = [T[:3,2].copy()]
    for i in range(6):
        alpha, a, d, th0 = UR5_DH[i]
        T = T @ dh_transform(alpha, a, d, th0 + q[i])
        p.append(T[:3,3].copy())
        z.append(T[:3,2].copy())
    return T, p, z

def jacobian_full(q):

    T, p, z = fk_and_frames(q)
    pe = p[-1]
    Jv = np.zeros((3,6)); Jw = np.zeros((3,6))
    for i in range(6):
        Jv[:,i] = np.cross(z[i], pe - p[i])
        Jw[:,i] = z[i]
    return np.vstack([Jv, Jw]), T[:3,3], T[:3,:3]

def rot_to_axis_angle(R):

    eps = 1e-12
    tr = np.trace(R)
    th = np.arccos(np.clip((tr - 1.0)/2.0, -1.0, 1.0))
    if th < 1e-9:
        return np.array([0,0,0], dtype=float), 0.0
    rx = (R[2,1]-R[1,2])/(2*np.sin(th)+eps)
    ry = (R[0,2]-R[2,0])/(2*np.sin(th)+eps)
    rz = (R[1,0]-R[0,1])/(2*np.sin(th)+eps)
    axis = np.array([rx,ry,rz], dtype=float)
    n = np.linalg.norm(axis)+eps
    return axis/n, th


def feasible_time_for_distance(D, a_lin):

    if D < 1e-12: return 0.5
    return 2.0*math.sqrt(D/max(a_lin,1e-12))

def solve_trapezoid_scalar(D, a_lin, T):

    eps = 1e-12
    disc = 1.0 - 4.0*D/(max(a_lin,eps)*max(T,eps)**2)
    disc = max(0.0, min(1.0, disc))
    v = 0.5*a_lin*T*(1.0 - math.sqrt(disc))   # maximum linear speed on the path
    tb = v/max(a_lin, eps)
    tc = T - 2.0*tb
    if tc < 0.0: tc = 0.0
    return v, tb, tc

def s_lipb(t, T, tb, tc, a_lin, D):

    if D < 1e-12:
        return 1.0, 0.0, 0.0
    a_s = a_lin / D
    if t < tb:
        s = 0.5*a_s*t*t
        sd = a_s*t
        sdd = a_s
    elif t < tb + tc:
        s = 0.5*a_s*tb*tb + (a_s*tb)*(t - tb)
        sd = a_s*tb
        sdd = 0.0
    else:
        tau = t - (tb + tc)
        s = 1.0 - 0.5*a_s*(tb - tau)**2
        sd = a_s*(tb - tau)
        sdd = -a_s

    if s < 0.0: s = 0.0
    if s > 1.0: s = 1.0
    return s, sd, sdd


def rand_in_range(lo, hi):
    return lo + (hi-lo)*random.random()

def sample_random_position():
    return np.array([
        rand_in_range(*BOX_X),
        rand_in_range(*BOX_Y),
        rand_in_range(*BOX_Z),
    ], dtype=float)

def clamp_dq(dq, dq_max):
    return np.clip(dq, -dq_max, dq_max)

def ik_step_pose(q, p_target, R_target):

    J, p_now, R_now = jacobian_full(q)
    e_p = p_target - p_now
    R_err = R_now.T @ R_target
    axis, ang = rot_to_axis_angle(R_err)
    e_w = axis * ang

    e6 = np.hstack([Wpos*e_p, Wang*e_w])
    JJt = J @ J.T
    dq = J.T @ np.linalg.solve(JJt + (DAMPING*np.eye(6)), e6)
    dq = clamp_dq(dq, DQ_MAX)
    return q + dq, np.linalg.norm(e_p), np.linalg.norm(e_w)


def sysCall_init():
    global sim, joint_h, _state
    sim = require("sim")
    random.seed(int(time.time()*1000))

    
    joint_h = [sim.getObject(p) for p in JOINT_PATHS]
    for j in range(6):
        sim.setJointMode(joint_h[j], sim.jointmode_force, 0)

    
    q = np.array([sim.getJointPosition(h) for h in joint_h], dtype=float)
    T0e, _, _ = fk_and_frames(q)
    R_fixed = T0e[:3,:3].copy()

    
    p_start = sample_random_position()
    p_final = sample_random_position()

    
    _state = {
        "phase": "prep", "q": q,
        "p_start": p_start, "p_final": p_final, "R_fixed": R_fixed,
        "prep_iter": 0
    }
    sim.addLog(sim.verbosity_scriptinfos,
               f"[LIPB-TS] p_start={p_start}, p_final={p_final}")

def sysCall_thread():
    global sim, joint_h, _state

    
    if _state["phase"] == "prep":
        q = _state["q"]
        for _ in range(PREP_MAX_ITERS):
            q, ep, ew = ik_step_pose(q, _state["p_start"], _state["R_fixed"])
            for j in range(6):
                sim.setJointTargetPosition(joint_h[j], float(q[j]))
            if REALTIME: sim.wait(DT)
            else:        sim.switchThread()
            if ep < POS_TOL:
                break
        _state["q"] = q
        _state["phase"] = "follow"

        
        p0, pf = _state["p_start"], _state["p_final"]
        dvec = pf - p0
        D = float(np.linalg.norm(dvec))
        T_min = feasible_time_for_distance(D, A_LIN_MAX)
        T_eff = max(T_DESIRED, T_min)
        v_lin, tb, tc = solve_trapezoid_scalar(D, A_LIN_MAX, T_eff)

        N = max(1, int(math.ceil(T_eff/DT)))
        tgrid = np.linspace(0.0, T_eff, N+1)

        _state.update(dict(T_eff=T_eff, D=D, dvec=dvec, tgrid=tgrid, tb=tb, tc=tc, k=0))
        sim.addLog(sim.verbosity_scriptinfos,
                   f"[LIPB-TS] T_desired={T_DESIRED:.3f}s, T_eff={T_eff:.3f}s, "
                   f"D={D:.3f} m, tb={tb:.3f}s, tc={tc:.3f}s, steps={len(tgrid)}")

    
    elif _state["phase"] == "follow":
        q      = _state["q"]
        p0     = _state["p_start"]
        Rfix   = _state["R_fixed"]
        dvec   = _state["dvec"]
        D      = _state["D"]
        T_eff  = _state["T_eff"]
        tb, tc = _state["tb"], _state["tc"]
        tgrid  = _state["tgrid"]
        k      = _state["k"]

        while k < len(tgrid):
            t = tgrid[k]
            s, _, _ = s_lipb(t, T_eff, tb, tc, A_LIN_MAX, D)
            p_d = p0 + dvec * s

            q, ep, ew = ik_step_pose(q, p_d, Rfix)
            for j in range(6):
                sim.setJointTargetPosition(joint_h[j], float(q[j]))

            _state["q"] = q
            _state["k"] = k+1
            k += 1

            if REALTIME: sim.wait(DT)
            else:        sim.switchThread()

        
        for j in range(6):
            sim.setJointTargetPosition(joint_h[j], float(_state["q"][j]))
        _state["phase"] = "done"
        sim.addLog(sim.verbosity_scriptinfos, "[LIPB-TS] Done. Holding final pose.")
        return

    
    elif _state["phase"] == "done":
        sim.switchThread()
        return

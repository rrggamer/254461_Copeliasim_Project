#luaExec wrapper='pythonWrapper'
# -*- coding: utf-8 -*-

import time
import numpy as np
import math

# ---------- CONSTANTS ----------
pi  = np.pi
d2r = pi / 180.0
r2d = 180.0 / pi
sim = None  


# ======== CORE MATH ===========


def mdh_T(a, alpha, d, theta):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    T = np.empty((4, 4), dtype=float)
    T[:, 0] = [ct,      st*ca,  st*sa,  0.0]
    T[:, 1] = [-st,     ct*ca,  ct*sa,  0.0]
    T[:, 2] = [0.0,     -sa,    ca,     0.0]
    T[:, 3] = [a,       -d*sa,  d*ca,   1.0]
    return T

def so3_logmap(R):
    tr = float(np.trace(R))
    c = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = math.acos(c)
    if theta < 1e-6:
        return np.zeros(3, dtype=float)
    S = (R - R.T) * (0.5 / math.sin(theta))
    return theta * np.array([S[2,1], S[0,2], S[1,0]], dtype=float)

def so3_exponential(w):
    ang = float(np.linalg.norm(w))
    if ang < 1e-9:
        return np.eye(3)
    k = w / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang)*K + (1.0 - math.cos(ang))*(K @ K)

def slerp_rotmat(R0: np.ndarray, R1: np.ndarray, s: float) -> np.ndarray:
    Rrel = R1 @ R0.T
    w = so3_logmap(Rrel)
    return so3_exponential(s * w) @ R0

def twist_error6(T_cur, T_goal):
    dp = T_goal[:3, 3] - T_cur[:3, 3]
    dw = so3_logmap(T_goal[:3, :3] @ T_cur[:3, :3].T)
    return np.hstack((dp, dw))

_DH_BASE = np.array([[0.0, 0.0, 0.0892, -pi/2]], dtype=float) 
_DH_MAIN = np.array([
    [0.0,       0.0,      0.0,     0.0],
    [pi/2,      0.0,      0.0,     0.0],
    [0.0,       0.4251,   0.0,     0.0],
    [0.0,       0.39215,  0.11,    0.0],
    [-pi/2,     0.0,      0.09475, 0.0],
    [pi/2,      0.0,      0.0,     0.0],
], dtype=float)
_DH_EE = np.array([[0.0, 0.0, 0.26658, pi]], dtype=float)
_JOFF = np.array([0.0, pi/2, 0.0, -pi/2, 0.0, 0.0], dtype=float)


# ======== UR5 Kinematics ===========

def _compose_mdh_chain(dh_table: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for (alpha, a, d, theta) in dh_table:
        T = T @ mdh_T(a, alpha, d, theta)
    return T


def ur5_fk(q):
    dh = _DH_MAIN.copy()
    dh[:, 3] += (np.asarray(q, dtype=float) + _JOFF)
    T = _compose_mdh_chain(_DH_BASE)
    T = T @ _compose_mdh_chain(dh)
    T = T @ _compose_mdh_chain(_DH_EE)
    return T

def ur5_geo_jacobian(q):
    q = np.asarray(q, dtype=float)
    frames = [_compose_mdh_chain(_DH_BASE)]
    T = frames[0]
    for i in range(6):
        alpha, a, d, _ = _DH_MAIN[i]
        th = q[i] + _JOFF[i]
        T = T @ mdh_T(a, alpha, d, th)
        frames.append(T)

    T_ee = frames[-1] @ _compose_mdh_chain(_DH_EE)
    p_e = T_ee[:3, 3]

    J = np.zeros((6, 6), dtype=float)
    for i in range(6):
        z_i = frames[i+1][:3, 2]
        p_i = frames[i+1][:3, 3]
        J[:3, i] = np.cross(z_i, (p_e - p_i))
        J[3:, i] = z_i
    return J

# ======== Inverse Kinematics ===========

def ik_dls_update(q, T_goal, lam=0.05, pos_w=1.0, rot_w=0.6, dq_max=5*d2r):
    """One DLS step (renamed)."""
    q = np.asarray(q, dtype=float)
    T_now = ur5_fk(q)
    err = twist_error6(T_now, T_goal)

    W = np.diag([pos_w, pos_w, pos_w, rot_w, rot_w, rot_w])
    we = W @ err
    J = ur5_geo_jacobian(q)
    JW = W @ J

    H = JW @ JW.T + (lam*lam) * np.eye(6)
    dq = JW.T @ np.linalg.solve(H, we)

    m = float(np.max(np.abs(dq)))
    if m > dq_max:
        dq *= (dq_max / m)

    return q + dq, err

# ==============================
# ========= TRAJECTORIES =======
# ==============================

def smoothstep_05_10(tau: float) -> float:
    """C^2 smoothstep (renamed)."""
    t = float(np.clip(tau, 0.0, 1.0))
    return 6*t**5 - 15*t**4 + 10*t**3

def _quintic_matrix(T):
    return np.array([
        [1, 0,     0,       0,        0,         0],
        [1, T,     T**2,    T**3,     T**4,      T**5],
        [0, 1,     0,       0,        0,         0],
        [0, 1,   2*T,     3*T**2,   4*T**3,    5*T**4],
        [0, 0,     2,       0,        0,         0],
        [0, 0,     2,     6*T,     12*T**2,   20*T**3]
    ], dtype=float)

def quintic_solve_coeffs(q0, q1, v0=0.0, v1=0.0, a0=0.0, a1=0.0, Tf=1.0):
    M = _quintic_matrix(Tf)
    b = np.array([q0, q1, v0, v1, a0, a1], dtype=float)
    return np.linalg.solve(M, b)

def quintic_sample(coeffs, t):
    a0,a1,a2,a3,a4,a5 = coeffs
    tt  = t
    tt2 = tt*tt
    tt3 = tt2*tt
    tt4 = tt3*tt
    tt5 = tt4*tt
    q   = a0 + a1*tt + a2*tt2 + a3*tt3 + a4*tt4 + a5*tt5
    qd  = a1 + 2*a2*tt + 3*a3*tt2 + 4*a4*tt3 + 5*a5*tt4
    qdd = 2*a2 + 6*a3*tt + 12*a4*tt2 + 20*a5*tt3
    return q, qd, qdd

def plan_joint_quintic_path(q0: np.ndarray, q1: np.ndarray, Tf: float=2.0, dt: float=0.02,
                            v0=None, v1=None, a0=None, a1=None):
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    dof = q0.size
    v0 = np.zeros(dof) if v0 is None else np.asarray(v0, float)
    v1 = np.zeros(dof) if v1 is None else np.asarray(v1, float)
    a0 = np.zeros(dof) if a0 is None else np.asarray(a0, float)
    a1 = np.zeros(dof) if a1 is None else np.asarray(a1, float)
    coeffs = [quintic_solve_coeffs(q0[i], q1[i], v0[i], v1[i], a0[i], a1[i], Tf) for i in range(dof)]
    ts = np.arange(0.0, Tf + 1e-9, dt)
    path = np.empty((ts.size, dof), dtype=float)
    for k, t in enumerate(ts):
        for i in range(dof):
            path[k, i] = quintic_sample(coeffs[i], t)[0]
    return path, ts

def play_joint_trajectory(hdl_j: dict, q_traj: np.ndarray):
    for row in q_traj:
        for i in range(6):
            sim.setJointTargetPosition(hdl_j[i], float(row[i]))
        sim.switchThread()

def plan_cartesian_linear(T0: np.ndarray, T1: np.ndarray, Tf: float=2.0, dt: float=0.02):
    T0 = np.asarray(T0, float); T1 = np.asarray(T1, float)
    p0, R0 = T0[:3, 3], T0[:3, :3]
    p1, R1 = T1[:3, 3], T1[:3, :3]
    ts = np.arange(0.0, Tf + 1e-9, dt)
    poses = []
    for t in ts:
        s = smoothstep_05_10(t / Tf)
        p = (1.0 - s)*p0 + s*p1
        R = slerp_rotmat(R0, R1, s)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = p
        poses.append(T)
    return poses, ts

def track_taskspace_with_ik(q_seed: np.ndarray, T_traj, hdl_j: dict,
                            lam: float=0.08, pos_w: float=1.0, rot_w: float=0.6,
                            dq_max: float=5*np.pi/180.0, micro_steps: int=3):
    q = np.array(q_seed, float)
    for T_goal in T_traj:
        for _ in range(micro_steps):
            q, _ = ik_dls_update(q, T_goal, lam=lam, pos_w=pos_w, rot_w=rot_w, dq_max=dq_max)
        for i in range(6):
            sim.setJointTargetPosition(hdl_j[i], float(q[i]))
        sim.switchThread()
    return q

def move_via_jointspace(q_start, q_goal, hdl_j, Tf=2.0, dt=0.02):
    traj, _ = plan_joint_quintic_path(q_start, q_goal, Tf=Tf, dt=dt)
    play_joint_trajectory(hdl_j, traj)
    return traj[-1]

def move_via_taskspace(q_seed, T_start, T_goal, hdl_j, Tf=2.0, dt=0.02,
                       lam=0.08, pos_w=1.0, rot_w=0.6, dq_max=5*np.pi/180.0,
                       micro_steps: int=3):
    poses, _ = plan_cartesian_linear(T_start, T_goal, Tf=Tf, dt=dt)
    return track_taskspace_with_ik(q_seed, poses, hdl_j,
                                   lam=lam, pos_w=pos_w, rot_w=rot_w, dq_max=dq_max,
                                   micro_steps=micro_steps)

# ==============================
# =========== GRIPPER ==========
# ==============================

def rg2_command(hdl_gripper, action):
    if action == 1:
        sim.setJointTargetForce(hdl_gripper, 20)
        sim.setJointTargetVelocity(hdl_gripper, 0.05)
        return
    if action == 2:
        sim.setJointTargetForce(hdl_gripper, 20)
        sim.setJointTargetVelocity(hdl_gripper, -0.02)

# ==============================
# ======= SCENE UTILITIES ======
# ==============================

def get_ur5_jhandles():
    candidates = [[
        "/UR5/joint",
        "/UR5/joint/link/joint",
        "/UR5/joint/link/joint/link/joint",
        "/UR5/joint/link/joint/link/joint/link/joint",
        "/UR5/joint/link/joint/link/joint/link/joint/link/joint",
        "/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint",
    ]]
    for names in candidates:
        try:
            return {i: sim.getObject(n) for i, n in enumerate(names)}
        except Exception:
            pass
    raise RuntimeError("UR5 joints not found")

def first_available_handle(paths):
    for p in paths:
        try:
            return sim.getObject(p)
        except Exception:
            continue
    return None

def _list_all_scene_objects():
    return sim.getObjectsInTree(sim.handle_scene)

def list_cuboids():
    items = []
    for h in _list_all_scene_objects():
        try:
            name = sim.getObjectAlias(h)
            if 'Cuboid' in name:
                items.append((name, h))
        except Exception:
            pass
    items.sort(key=lambda x: x[0])
    for nm, _ in items:
        print(f"  Found cuboid: {nm}")
    return [h for _, h in items]

def list_drop_targets():
    items = []
    for h in _list_all_scene_objects():
        try:
            name = sim.getObjectAlias(h)
            if 'Drop' in name:
                items.append((name, h))
        except Exception:
            pass
    items.sort(key=lambda x: x[0])
    for nm, _ in items:
        print(f"  Found drop target: {nm}")
    return [h for _, h in items]

# ==============================
# ===== PICK & PLACE (ONE) =====
# ==============================

def do_pick_and_place_once(cuboid_hdl, drop_hdl, q_current, hdl_j, hdl_grip, hdl_att,
                           Twb, R_tilt, R_down):
    c_name = sim.getObjectAlias(cuboid_hdl)
    d_name = sim.getObjectAlias(drop_hdl)
    print(f"\n[PICK & PLACE] {c_name} -> {d_name}")

    sim.setObjectParent(cuboid_hdl, -1, True)

    # PICK
    print(f"  Moving to pick {c_name}...")
    Twc = np.array(sim.getObjectMatrix(cuboid_hdl, -1), dtype=float).reshape(3, 4)
    Twc = np.vstack((Twc, [0, 0, 0, 1]))
    T_goal_pick = Twb @ Twc
    T_goal_pick[:3, :3] = R_tilt

    T_start = ur5_fk(q_current)
    q_at_pick = move_via_taskspace(q_current, T_start, T_goal_pick, hdl_j,
                                   Tf=2.0, dt=0.02, lam=0.08, dq_max=5*d2r, micro_steps=3)

    rg2_command(hdl_grip, 2)  # close
    time.sleep(0.3)
    sim.setObjectParent(cuboid_hdl, hdl_att, True)

    # PLACE
    print(f"  Moving to place at {d_name}...")
    Twd = np.array(sim.getObjectMatrix(drop_hdl, -1), dtype=float).reshape(3, 4)
    Twd = np.vstack((Twd, [0, 0, 0, 1]))
    T_goal_place = Twb @ Twd
    T_goal_place[:3, :3] = R_down

    T_mid = ur5_fk(q_at_pick)
    q_at_place = move_via_taskspace(q_at_pick, T_mid, T_goal_place, hdl_j,
                                    Tf=2.5, dt=0.02, lam=0.08, dq_max=5*d2r, micro_steps=3)

    rg2_command(hdl_grip, 1)  # open
    time.sleep(0.3)
    sim.setObjectParent(cuboid_hdl, -1, True)

    return q_at_place

# ==============================
# ===== COPPELIASIM HOOKS ======
# ==============================

def sysCall_init():
    global sim
    sim = require("sim")
    print("[INIT] Python thread started.")

def sysCall_thread():
    print("===== UR5 Multi-Cuboid Pick-and-Place =====")

    # Handles
    hdl_j   = get_ur5_jhandles()
    hdl_base= sim.getObject("/UR5")
    hdl_grip= first_available_handle(["/UR5/RG2/openCloseJoint"])
    hdl_att = first_available_handle(["/UR5/RG2/attachPoint", "/UR5/EndPoint"])

    # Discovery
    print("\n[DISCOVERY] Searching for cuboids...")
    cuboids = list_cuboids()

    print("\n[DISCOVERY] Searching for drop targets...")
    drop_targets = list_drop_targets()

    if not cuboids:
        print("[ERROR] No cuboids found! Make sure objects are named 'Cuboid', 'Cuboid0', etc.")
        return
    if not drop_targets:
        print("[ERROR] No drop targets found! Make sure targets are named 'Drop', 'Drop0', etc.")
        return

    print(f"\n[INFO] Found {len(cuboids)} cuboid(s) and {len(drop_targets)} drop target(s)")

    # Init
    q_home = np.array([sim.getJointPosition(hdl_j[i]) for i in range(6)], dtype=float)
    _ = ur5_fk(q_home)  # parity
    rg2_command(hdl_grip, 1)
    time.sleep(0.2)

    # World/Base xform
    Tbw = np.array(sim.getObjectMatrix(hdl_base, -1), dtype=float).reshape(3, 4)
    Tbw = np.vstack((Tbw, [0, 0, 0, 1]))
    Twb = np.linalg.inv(Tbw)

    # Comfortable wrist orientation
    preset_deg = {0: 90, 4: 90, 5: 90}
    for idx, deg in preset_deg.items():
        sim.setJointTargetPosition(hdl_j[idx], deg * d2r)
    sim.switchThread()
    q_current = np.array([sim.getJointPosition(hdl_j[i]) for i in range(6)], dtype=float)
    R_tilt = ur5_fk(q_current)[:3, :3]

    # Down orientation for placing
    ex = np.array([1, 0, 0], dtype=float)
    ez = np.array([0, 0, -1], dtype=float)
    ey = np.cross(ez, ex)
    R_down = np.column_stack((ex, ey, ez))

    # Process
    num_pairs = min(len(cuboids), len(drop_targets))
    print(f"\n[START] Processing {num_pairs} pick-and-place operation(s)...\n")

    for i in range(num_pairs):
        q_current = do_pick_and_place_once(
            cuboids[i], drop_targets[i], q_current,
            hdl_j, hdl_grip, hdl_att, Twb, R_tilt, R_down
        )
        print(f"Completed {i+1}/{num_pairs}")

    # Return home
    print("\n[PHASE] RETURN HOME (joint-space)")
    _ = move_via_jointspace(q_current, q_home, hdl_j, Tf=3.0, dt=0.02)

    print("\n[DONE] All operations COMPLETE!")
    print(f"      Processed {num_pairs} cuboid(s) successfully!")
    sim.switchThread()

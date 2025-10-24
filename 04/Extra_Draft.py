#luaExec wrapper='pythonWrapper'
# -*- coding: utf-8 -*-

import time
import numpy as np
import math

# ---------- CONSTANTS ----------
pi  = np.pi
d2r = pi / 180.0
r2d = 180.0 / pi
sim = None  # set in sysCall_init()

# ======== ORIGINAL HELPERS (your kinematics) ========

def create_T_matrix_modified_dh(a, alpha, d, theta):
    cth, sth = np.cos(theta), np.sin(theta)
    cal, sal = np.cos(alpha), np.sin(alpha)
    return np.array([
        [cth,      -sth,       0,    a],
        [sth*cal,  cth*cal,   -sal, -d*sal],
        [sth*sal,  cth*sal,    cal,  d*cal],
        [0,        0,          0,    1]
    ])

def so3_log(R):
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = math.acos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3)
    w_hat = (R - R.T) * (0.5 / math.sin(theta))
    return theta * np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]])

def pose_error(T_cur, T_goal):
    p_cur = T_cur[0:3, 3]; R_cur = T_cur[0:3, 0:3]
    p_des = T_goal[0:3, 3]; R_des = T_goal[0:3, 0:3]
    dp = p_des - p_cur
    dw = so3_log(R_des @ R_cur.T)
    return np.hstack([dp, dw])

def fk_ur5(q):
    DH_Base = np.array([[0, 0, 0.0892, -pi/2]])
    DH = np.array([
        [0,           0,        0,       0],
        [np.pi/2,     0,        0,       0],
        [0,           0.4251,   0,       0],
        [0,           0.39215,  0.11,    0],
        [-np.pi/2,    0,        0.09475, 0],
        [np.pi/2,     0,        0,       0],
    ])
    DH_EE = np.array([[0, 0, 0.26658, np.pi]])
    th_offset = [0, np.pi/2, 0, -np.pi/2, 0, 0]

    T = create_T_matrix_modified_dh(DH_Base[0,1], DH_Base[0,0], DH_Base[0,2], DH_Base[0,3])
    for i in range(6):
        theta = q[i] + th_offset[i] + DH[i,3]
        T = T @ create_T_matrix_modified_dh(DH[i,1], DH[i,0], DH[i,2], theta)
    T = T @ create_T_matrix_modified_dh(DH_EE[0,1], DH_EE[0,0], DH_EE[0,2], DH_EE[0,3])
    return T

def calculate_analytical_jacobian(q):
    DH_Base = np.array([[0,0,0.0892,-np.pi/2]])
    DH = np.array([
        [0,           0,        0,       0],
        [np.pi/2,     0,        0,       0],
        [0,           0.4251,   0,       0],
        [0,           0.39215,  0.11,    0],
        [-np.pi/2,    0,        0.09475, 0],
        [np.pi/2,     0,        0,       0],
    ])
    DH_EE = np.array([[0, 0, 0.26658, np.pi]])
    th_offset = [0, np.pi/2, 0, -np.pi/2, 0, 0]

    Ts = []
    Tcur = create_T_matrix_modified_dh(DH_Base[0,1], DH_Base[0,0], DH_Base[0,2], DH_Base[0,3])
    Ts.append(Tcur)
    for i in range(6):
        theta = q[i] + th_offset[i] + DH[i,3]
        Tcur = Tcur @ create_T_matrix_modified_dh(DH[i,1], DH[i,0], DH[i,2], theta)
        Ts.append(Tcur)

    T_ee = Ts[-1] @ create_T_matrix_modified_dh(DH_EE[0,1], DH_EE[0,0], DH_EE[0,2], DH_EE[0,3])
    p_end = T_ee[0:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        Ti = Ts[i+1]
        zi = Ti[0:3, 2]
        pi = Ti[0:3, 3]
        J[0:3, i] = np.cross(zi, p_end - pi)
        J[3:6, i] = zi
    return J

def ik_dls_step(q, T_goal, lam=0.05, pos_w=1.0, rot_w=0.6, dq_max=5*d2r):
    T = fk_ur5(q)
    e = pose_error(T, T_goal)
    W = np.diag([pos_w, pos_w, pos_w, rot_w, rot_w, rot_w])
    e = W @ e
    Jw = W @ calculate_analytical_jacobian(q)
    A = Jw @ Jw.T + (lam**2) * np.eye(6)
    dq = Jw.T @ np.linalg.solve(A, e)
    m = np.max(np.abs(dq))
    if m > dq_max:
        dq *= dq_max/m
    return q + dq, e

# ---------- GRIPPER ----------

def gripper_action(hdl_gripper, action):
    # 1=open, 2=close
    if action == 1:
        sim.setJointTargetForce(hdl_gripper, 20)
        sim.setJointTargetVelocity(hdl_gripper, 0.05)
    elif action == 2:
        sim.setJointTargetForce(hdl_gripper, 20)
        sim.setJointTargetVelocity(hdl_gripper, -0.02)

# ======== TRAJECTORY PLANNING ========

def s_curve_05_10(tau: float) -> float:
    tau = float(np.clip(tau, 0.0, 1.0))
    return 6*tau**5 - 15*tau**4 + 10*tau**3

def so3_exp(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-9:
        return np.eye(3)
    k = w / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta)*K + (1-math.cos(theta))*(K@K)

def quintic_coeff(q0, q1, v0=0.0, v1=0.0, a0=0.0, a1=0.0, Tf=1.0):
    T = Tf
    M = np.array([
        [1, 0,    0,      0,       0,        0],
        [1, T,    T**2,   T**3,    T**4,     T**5],
        [0, 1,    0,      0,       0,        0],
        [0, 1,  2*T,   3*T**2,  4*T**3,   5*T**4],
        [0, 0,    2,      0,       0,        0],
        [0, 0,    2,    6*T,   12*T**2,  20*T**3]
    ], dtype=float)
    b = np.array([q0, q1, v0, v1, a0, a1], dtype=float)
    return np.linalg.solve(M, b)

def sample_quintic(coeffs, t):
    a0,a1,a2,a3,a4,a5 = coeffs
    q  = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    qd =      a1   + 2*a2*t  + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
    qdd=                2*a2 + 6*a3*t    + 12*a4*t**2 + 20*a5*t**3
    return q, qd, qdd

def plan_joint_quintic(q0: np.ndarray, q1: np.ndarray, Tf: float=2.0, dt: float=0.02,
                        v0=None, v1=None, a0=None, a1=None):
    q0 = np.asarray(q0, float); q1 = np.asarray(q1, float)
    dof = q0.size
    if v0 is None: v0 = np.zeros(dof)
    if v1 is None: v1 = np.zeros(dof)
    if a0 is None: a0 = np.zeros(dof)
    if a1 is None: a1 = np.zeros(dof)
    coeffs = [quintic_coeff(q0[i], q1[i], v0[i], v1[i], a0[i], a1[i], Tf) for i in range(dof)]
    ts = np.arange(0.0, Tf+1e-9, dt)
    q_traj  = np.zeros((ts.size, dof))
    for k,t in enumerate(ts):
        for i in range(dof):
            q,_,_ = sample_quintic(coeffs[i], t)
            q_traj[k,i] = q
    return q_traj, ts

def execute_joint_traj(hdl_j: dict, q_traj: np.ndarray):
    for k in range(q_traj.shape[0]):
        for i in range(6):
            sim.setJointTargetPosition(hdl_j[i], float(q_traj[k,i]))
        sim.switchThread()

def slerp_R(R0: np.ndarray, R1: np.ndarray, s: float) -> np.ndarray:
    R_rel = R1 @ R0.T
    w = so3_log(R_rel)
    return so3_exp(s*w) @ R0

def plan_task_linear(T0: np.ndarray, T1: np.ndarray, Tf: float=2.0, dt: float=0.02):
    T0 = np.asarray(T0, float); T1 = np.asarray(T1, float)
    p0, R0 = T0[:3,3], T0[:3,:3]
    p1, R1 = T1[:3,3], T1[:3,:3]
    ts = np.arange(0.0, Tf+1e-9, dt)
    poses = []
    for t in ts:
        s = s_curve_05_10(t/Tf)
        p = (1.0 - s)*p0 + s*p1
        R = slerp_R(R0, R1, s)
        T = np.eye(4)
        T[:3,:3] = R
        T[:3, 3] = p
        poses.append(T)
    return poses, ts

def execute_task_traj(q_seed: np.ndarray, T_traj, hdl_j: dict,
                       lam: float=0.08, pos_w: float=1.0, rot_w: float=0.6, dq_max: float=5*np.pi/180.0,
                       micro_steps: int=3):
    q = np.array(q_seed, float)
    for T_goal in T_traj:
        for _ in range(micro_steps):
            q, _ = ik_dls_step(q, T_goal, lam=lam, pos_w=pos_w, rot_w=rot_w, dq_max=dq_max)
        for i in range(6):
            sim.setJointTargetPosition(hdl_j[i], float(q[i]))
        sim.switchThread()
    return q

def move_jointspace(q_start, q_goal, hdl_j, Tf=2.0, dt=0.02):
    q_traj, ts = plan_joint_quintic(q_start, q_goal, Tf=Tf, dt=dt)
    execute_joint_traj(hdl_j, q_traj)
    return q_traj[-1]

def move_taskspace(q_seed, T_start, T_goal, hdl_j, Tf=2.0, dt=0.02,
                   lam=0.08, pos_w=1.0, rot_w=0.6, dq_max=5*np.pi/180.0,
                   micro_steps: int=3):
    T_traj, _ = plan_task_linear(T_start, T_goal, Tf=Tf, dt=dt)
    q_end = execute_task_traj(q_seed, T_traj, hdl_j,
                              lam=lam, pos_w=pos_w, rot_w=rot_w, dq_max=dq_max,
                              micro_steps=micro_steps)
    return q_end

# ======== HANDLE DISCOVERY ========

def get_ur5_joint_handles():
    names_try = [
        ["/UR5/joint",
         "/UR5/joint/link/joint",
         "/UR5/joint/link/joint/link/joint",
         "/UR5/joint/link/joint/link/joint/link/joint",
         "/UR5/joint/link/joint/link/joint/link/joint/link/joint",
         "/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint"]
    ]
    for name_set in names_try:
        try:
            h = {i: sim.getObject(nm) for i,nm in enumerate(name_set)}
            return h
        except:
            pass
    raise RuntimeError("UR5 joints not found")

def first_existing(paths):
    for p in paths:
        try:
            return sim.getObject(p)
        except:
            pass
    return None

def find_all_cuboids():
    """Find all objects with 'Cuboid' in their name"""
    cuboids = []
    all_objects = sim.getObjectsInTree(sim.handle_scene)
    for obj_handle in all_objects:
        try:
            obj_name = sim.getObjectAlias(obj_handle)
            if 'Cuboid' in obj_name:
                cuboids.append(obj_handle)
                print(f"  Found cuboid: {obj_name}")
        except:
            pass
    return cuboids

def find_drop_targets():
    """Find all drop target positions (objects with 'Drop' in name)"""
    targets = []
    all_objects = sim.getObjectsInTree(sim.handle_scene)
    for obj_handle in all_objects:
        try:
            obj_name = sim.getObjectAlias(obj_handle)
            if 'Drop' in obj_name:
                targets.append(obj_handle)
                print(f"  Found drop target: {obj_name}")
        except:
            pass
    return targets

# ======== PICK AND PLACE FOR ONE CUBOID ========

def pick_and_place_cuboid(cuboid_hdl, drop_hdl, q_current, hdl_j, hdl_grip, hdl_att, 
                          Twb, R_tilt, R_down):
    """Pick one cuboid and place it at drop target"""
    
    cuboid_name = sim.getObjectAlias(cuboid_hdl)
    drop_name = sim.getObjectAlias(drop_hdl)
    print(f"\n[PICK & PLACE] {cuboid_name} -> {drop_name}")
    
    # Detach cuboid from any parent
    sim.setObjectParent(cuboid_hdl, -1, True)
    
    # === PICK ===
    print(f"  Moving to pick {cuboid_name}...")
    Ttw_pick = np.array(sim.getObjectMatrix(cuboid_hdl, -1)).reshape(3,4)
    Ttw_pick = np.vstack([Ttw_pick, [0,0,0,1]])
    T_goal_pick = Twb @ Ttw_pick
    T_goal_pick[:3,:3] = R_tilt
    
    T_start = fk_ur5(q_current)
    q_at_pick = move_taskspace(q_current, T_start, T_goal_pick, hdl_j,
                               Tf=2.0, dt=0.02, lam=0.08, dq_max=5*d2r, micro_steps=3)
    
    # Close gripper
    gripper_action(hdl_grip, 2)
    time.sleep(0.3)
    sim.setObjectParent(cuboid_hdl, hdl_att, True)
    
    # === PLACE ===
    print(f"  Moving to place at {drop_name}...")
    Ttw_place = np.array(sim.getObjectMatrix(drop_hdl, -1)).reshape(3,4)
    Ttw_place = np.vstack([Ttw_place, [0,0,0,1]])
    T_goal_place = Twb @ Ttw_place
    T_goal_place[:3,:3] = R_down
    
    T_start = fk_ur5(q_at_pick)
    q_at_place = move_taskspace(q_at_pick, T_start, T_goal_place, hdl_j,
                                Tf=2.5, dt=0.02, lam=0.08, dq_max=5*d2r, micro_steps=3)
    
    # Open gripper
    gripper_action(hdl_grip, 1)
    time.sleep(0.3)
    sim.setObjectParent(cuboid_hdl, -1, True)
    
    return q_at_place

# ======== CoppeliaSim Callbacks ========

def sysCall_init():
    global sim
    sim = require("sim")
    print("[INIT] Python thread started.")

def sysCall_thread():
    print("===== UR5 Multi-Cuboid Pick-and-Place =====")

    # --- Handles ---
    hdl_j   = get_ur5_joint_handles()
    hdl_base= sim.getObject("/UR5")
    hdl_grip= first_existing(["/UR5/RG2/openCloseJoint"])
    hdl_att = first_existing(["/UR5/RG2/attachPoint", "/UR5/EndPoint"])
    
    # --- Find all cuboids and drop targets ---
    print("\n[DISCOVERY] Searching for cuboids...")
    cuboids = find_all_cuboids()
    
    print("\n[DISCOVERY] Searching for drop targets...")
    drop_targets = find_drop_targets()
    
    if not cuboids:
        print("[ERROR] No cuboids found! Make sure objects are named 'Cuboid', 'Cuboid0', etc.")
        return
    
    if not drop_targets:
        print("[ERROR] No drop targets found! Make sure targets are named 'Drop', 'Drop0', etc.")
        return
    
    print(f"\n[INFO] Found {len(cuboids)} cuboid(s) and {len(drop_targets)} drop target(s)")
    
    # --- Initial setup ---
    q_home = np.array([sim.getJointPosition(hdl_j[i]) for i in range(6)], float)
    T_home = fk_ur5(q_home)
    gripper_action(hdl_grip, 1)
    time.sleep(0.2)

    # --- World<->Base transformation ---
    Tbw = np.array(sim.getObjectMatrix(hdl_base, -1)).reshape(3,4)
    Tbw = np.vstack([Tbw,[0,0,0,1]])
    Twb = np.linalg.inv(Tbw)

    # --- Preset comfortable wrist orientation ---
    rotate_deg = {0: 90, 4: 90, 5: 90}
    for idx,deg in rotate_deg.items():
        sim.setJointTargetPosition(hdl_j[idx], deg*d2r)
    sim.switchThread()
    q_current = np.array([sim.getJointPosition(hdl_j[i]) for i in range(6)], float)
    R_tilt = fk_ur5(q_current)[:3,:3]
    
    # Downward orientation for placing
    x_d = np.array([1,0,0]); z_d = np.array([0,0,-1]); y_d = np.cross(z_d, x_d)
    R_down = np.column_stack([x_d, y_d, z_d])

    # === PROCESS ALL CUBOIDS ===
    num_pairs = min(len(cuboids), len(drop_targets))
    print(f"\n[START] Processing {num_pairs} pick-and-place operation(s)...\n")
    
    for i in range(num_pairs):
        q_current = pick_and_place_cuboid(
            cuboids[i], drop_targets[i], q_current,
            hdl_j, hdl_grip, hdl_att, Twb, R_tilt, R_down
        )
        print(f"  ? Completed {i+1}/{num_pairs}")
    
    # === RETURN HOME ===
    print("\n[PHASE] RETURN HOME (joint-space)")
    _ = move_jointspace(q_current, q_home, hdl_j, Tf=3.0, dt=0.02)

    print("\n[DONE] All operations COMPLETE!")
    print(f"      Processed {num_pairs} cuboid(s) successfully!")
    sim.switchThread()
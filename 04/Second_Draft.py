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


# ---------- KINEMATICS ----------
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
    if theta < 1e-6: return np.zeros(3)
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
    if m > dq_max: dq *= dq_max/m
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


# ---------- MOTION EXECUTION ----------
def move_to_pose_ik(q_start, T_goal, hdl_j, max_iter=300,
                    lam=0.1, pos_w=1.0, rot_w=0.6, dq_max=5*d2r,
                    pos_tol=1e-3, rot_tol=1e-2):
    print("Moving to new target pose...")
    q = np.array(q_start, dtype=float)
    for k in range(max_iter):
        q, e = ik_dls_step(q, T_goal, lam=lam, pos_w=pos_w, rot_w=rot_w, dq_max=dq_max)
        for i in range(6):
            sim.setJointTargetPosition(hdl_j[i], float(q[i]))
        if np.linalg.norm(e[0:3]) < pos_tol and np.linalg.norm(e[3:6]) < rot_tol:
            print(f"Target reached in {k+1} iterations. PosErr:{np.linalg.norm(e[0:3]):.4f}, RotErr:{np.linalg.norm(e[3:6]):.4f}")
            return q
        sim.switchThread()
    print("IK failed to converge.")
    return q


# ---------- HANDLE DISCOVERY ----------
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
        except: pass
    raise RuntimeError("UR5 joints not found")


def first_existing(paths):
    for p in paths:
        try: return sim.getObject(p)
        except: pass
    return None


# ---------- CoppeliaSim Callbacks ----------
def sysCall_init():
    global sim
    sim = require("sim")
    print("[INIT] Python thread started.")


def sysCall_thread():
    print("===== UR5 Pick-and-Place Start =====")

    # Handles
    hdl_j = get_ur5_joint_handles()
    hdl_base = sim.getObject("/UR5")
    hdl_box = first_existing(["/Cuboid"])
    hdl_drop = first_existing(["/Drop"])
    hdl_grip = first_existing(["/UR5/RG2/openCloseJoint"])
    hdl_attach = first_existing(["/UR5/RG2/attachPoint", "/UR5/EndPoint"])

    if hdl_box is None or hdl_grip is None or hdl_attach is None:
        print("Missing handle(s)")
        return

    sim.setObjectParent(hdl_box, -1, True)

    # Initial config
    q_home = np.array([sim.getJointPosition(hdl_j[i]) for i in range(6)], float)
    T_home = fk_ur5(q_home)
    gripper_action(hdl_grip, 1)
    time.sleep(0.3)

    # World<->Base transform
    Tbw = np.array(sim.getObjectMatrix(hdl_base, -1)).reshape(3,4)
    Tbw = np.vstack([Tbw,[0,0,0,1]])
    Twb = np.linalg.inv(Tbw)

    # Orientation preset
    rotate_deg = {0: 90, 4: 90, 5: 90}
    for idx,deg in rotate_deg.items():
        sim.setJointTargetPosition(hdl_j[idx], deg*d2r)
    sim.switchThread()

    q_pretilt = np.array([sim.getJointPosition(hdl_j[i]) for i in range(6)], float)
    R_tilt = fk_ur5(q_pretilt)[:3,:3]

    # === PICK ===
    print("Phase 1: PICK")
    Ttw_pick = np.array(sim.getObjectMatrix(hdl_box, -1)).reshape(3,4)
    Ttw_pick = np.vstack([Ttw_pick, [0,0,0,1]])
    T_goal_pick = Twb @ Ttw_pick
    T_goal_pick[:3,:3] = R_tilt

    q_at_pick = move_to_pose_ik(q_pretilt, T_goal_pick, hdl_j)
    gripper_action(hdl_grip, 2)
    time.sleep(0.4)
    sim.setObjectParent(hdl_box, hdl_attach, True)

    # === PLACE ===
    print("Phase 2: PLACE")
    Ttw_place = np.array(sim.getObjectMatrix(hdl_drop, -1)).reshape(3,4)
    Ttw_place = np.vstack([Ttw_place, [0,0,0,1]])
    T_goal_place = Twb @ Ttw_place
    x_d = np.array([1,0,0]); z_d = np.array([0,0,-1]); y_d = np.cross(z_d, x_d)
    R_down = np.column_stack([x_d, y_d, z_d])
    T_goal_place[:3,:3] = R_down

    q_at_place = move_to_pose_ik(q_at_pick, T_goal_place, hdl_j)
    gripper_action(hdl_grip, 1)
    time.sleep(0.4)
    sim.setObjectParent(hdl_box, -1, True)

    # === RETURN HOME ===
    print("Phase 3: RETURN HOME")
    _ = move_to_pose_ik(q_at_place, T_home, hdl_j)

    print("Sequence COMPLETE!")
    sim.switchThread()

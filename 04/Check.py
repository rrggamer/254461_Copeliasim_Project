#luaExec wrapper='pythonWrapper'
# -*- coding: utf-8 -*-
import time
import numpy as np
import math

# ===================== GLOBALS =====================
pi  = np.pi
d2r = pi / 180.0
r2d = 180.0 / pi
sim = None  # set in sysCall_init()

# ===================== CORE KINEMATICS =====================
def create_T_matrix_modified_dh(a, alpha, d, theta):
    cth = np.cos(theta); sth = np.sin(theta)
    cal = np.cos(alpha); sal = np.sin(alpha)
    return np.array([
        [cth,      -sth,      0,    a],
        [sth*cal,  cth*cal,  -sal, -d*sal],
        [sth*sal,  cth*sal,   cal,  d*cal],
        [0,        0,         0,    1]
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

# ===================== SIM HELPERS =====================
def gripper_action(hdl_gripper, action):
    # action: 1=open, 2=close
    if action == 1:
        sim.setJointTargetForce(hdl_gripper, 20)
        sim.setJointTargetVelocity(hdl_gripper, 0.05)
    elif action == 2:
        sim.setJointTargetForce(hdl_gripper, 20)
        sim.setJointTargetVelocity(hdl_gripper, -0.02)

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

def move_joints_smooth(hdl_j, target_deg, T=2.5, dt=0.02):
    q0 = np.array([sim.getJointPosition(hdl_j[i]) for i in range(6)], float)
    q1 = np.array([d*d2r for d in target_deg], float)
    steps = max(1, int(T/dt))
    for k in range(steps+1):
        s = k/steps
        q = q0*(1-s)+q1*s
        for i in range(6):
            sim.setJointTargetPosition(hdl_j[i], float(q[i]))
        sim.switchThread(); time.sleep(dt)

def rotate_selected_joints(hdl_j, targets_deg_map, T=2.0, dt=0.02):
    q0 = np.array([sim.getJointPosition(hdl_j[i]) for i in range(6)], float)
    q1 = q0.copy()
    for idx,deg in targets_deg_map.items(): q1[idx]=deg*d2r
    steps = max(1,int(T/dt))
    for k in range(steps+1):
        s=k/steps; q=q0*(1-s)+q1*s
        for i in range(6): sim.setJointTargetPosition(hdl_j[i], float(q[i]))
        sim.switchThread(); time.sleep(dt)

# ===================== HANDLE DISCOVERY =====================
def get_ur5_joint_handles():
    names_try = [
        ["/UR5/joint", "/UR5/joint2", "/UR5/joint3", "/UR5/joint4", "/UR5/joint5", "/UR5/joint6"],
        ["/UR5/J1", "/UR5/J2", "/UR5/J3", "/UR5/J4", "/UR5/J5", "/UR5/J6"],
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
    base = sim.getObject("/UR5")
    joint_list = sim.getObjectsInTree(base, sim.object_joint_type, 1)
    joint_list = [j for j in joint_list if "openClose" not in sim.getObjectAlias(j,1)]
    joint_list = sorted(joint_list, key=lambda j: sim.getObjectAlias(j,1))
    if len(joint_list) < 6:
        raise RuntimeError("Could not find 6 UR5 joints under /UR5.")
    return {i: joint_list[i] for i in range(6)}

def first_existing(paths):
    for p in paths:
        try: return sim.getObject(p)
        except: pass
    return None

# ===================== Coppelia callbacks =====================
def sysCall_init():
    global sim
    sim = require("sim")
    print("[INIT] Python thread up.")

def sysCall_thread():
    print("--- Phase 0: Initialization ---")
    hdl_j = get_ur5_joint_handles()
    hdl_base  = sim.getObject("/UR5")
    hdl_box   = first_existing(["/Cuboid","/UR5/RG2/Cuboid"])  # เผื่อมีลูกอยู่ที่กริปเปอร์
    hdl_drop  = sim.getObject("/Dropoff_Target")
    hdl_grip  = first_existing(["/UR5/RG2/openCloseJoint","/UR5/openCloseJoint"])
    hdl_attach_point = first_existing(["/UR5/RG2/attachPoint","/UR5/RG2/attachPoint[1]","/UR5/RG2/connection","/UR5/RG2/EndPoint"])

    if hdl_box is None: raise RuntimeError("Cannot find Cuboid")
    if hdl_grip is None: raise RuntimeError("Cannot find RG2 openCloseJoint")
    if hdl_attach_point is None: raise RuntimeError("Cannot find RG2 attach point")

    try:
        sim.setObjectParent(hdl_box, -1, True)
    except: pass

    # Home
    q_home = np.array([sim.getJointPosition(hdl_j[i]) for i in range(6)], float)
    T_home = fk_ur5(q_home)
    print("Home position saved.")
    gripper_action(hdl_grip, 1); time.sleep(0.3)

    # World<->Base
    Tbw = np.array(sim.getObjectMatrix(hdl_base, -1)).reshape(3,4)
    Tbw = np.vstack([Tbw,[0,0,0,1]])
    Twb = np.linalg.inv(Tbw)

    # ----------------- PRETILT (ใช้แค่รอบเดียวที่ถูกใจ) -----------------
    PRETILT_DEG = [ +90, -55, -75, 90, 90, 90 ]
    # ใช้เฉพาะตัวนี้ตามที่บอกว่า "หมุนถูก"
    rotate_selected_joints(hdl_j, {0:90, 4:90, 5:90}, T=1.8)
    # ❌ ตัดบรรทัดนี้ทิ้ง เพราะทำให้หมุนกากา/ซ้ำ
    # move_joints_smooth(hdl_j, PRETILT_DEG, T=2.2)

    # ท่าปัจจุบันหลังจาก pre-tilt (ให้ IK เริ่มจากตรงนี้ ไม่ใช่ q_home)
    q_pretilt = np.array([sim.getJointPosition(hdl_j[i]) for i in range(6)], float)
    R_tilt    = fk_ur5(q_pretilt)[:3,:3]

    # ----------------- Phase 1: PICK -----------------
    print("\n--- Phase 1: Pick Operation ---")
    Ttw_pick = np.array(sim.getObjectMatrix(hdl_box, -1)).reshape(3,4)
    Ttw_pick = np.vstack([Ttw_pick, [0,0,0,1]])
    T_goal_pick = Twb @ Ttw_pick
    T_goal_pick[:3,:3] = R_tilt

    # 🔧 เปลี่ยนให้เริ่ม IK จาก q_pretilt (ไม่ใช่ q_home) จะได้ไม่หมุนซ้ำ
    q_at_pick = move_to_pose_ik(q_pretilt, T_goal_pick, hdl_j)

    # คีบ + attach เหมือนเดิม
    gripper_action(hdl_grip, 2); time.sleep(0.4)
    sim.setObjectParent(hdl_box, hdl_attach_point, True)

    # ----------------- Phase 2: PLACE -----------------
    print("\n--- Phase 2: Place Operation ---")
    Ttw_place = np.array(sim.getObjectMatrix(hdl_drop, -1)).reshape(3,4)
    Ttw_place = np.vstack([Ttw_place, [0,0,0,1]])
    T_goal_place = Twb @ Ttw_place
    x_d = np.array([1,0,0]); z_d = np.array([0,0,-1.0]); y_d = np.cross(z_d, x_d)
    R_down = np.column_stack([x_d, y_d, z_d])
    T_goal_place[:3,:3] = R_down

    q_at_place = move_to_pose_ik(q_at_pick, T_goal_place, hdl_j)

    gripper_action(hdl_grip, 1); time.sleep(0.4)
    sim.setObjectParent(hdl_box, -1, True)

    # ----------------- Phase 3: RETURN HOME -----------------
    print("\n--- Phase 3: Return Home ---")
    _ = move_to_pose_ik(q_at_place, T_home, hdl_j)

    print("\n==============================")
    print("Sequence COMPLETE!")
    print("==============================")
    sim.switchThread()

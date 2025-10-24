# -*- coding: utf-8 -*-
"""
UR5 Robot Control Module + DLS-IK
Handles forward/inverse kinematics and trajectory planning for UR5 robot in CoppeliaSim
"""

import time
import numpy as np
from typing import List, Tuple

# ---------------- Constants ----------------
PI = np.pi
DEG_TO_RAD = PI / 180.0
RAD_TO_DEG = 180.0 / PI

# DH Parameters for UR5
DH_BASE_PARAMS = np.array([
    [0,    0,      0.0892, -90],
    [90,   0,      0,       90],
    [0,    0.4251, 0,        0],
    [0,    0.39215,0.11,    -90],
    [-90,  0,      0.09473,  0],
    [90,   0,      0,        0],
], dtype=float)

TOOL_OFFSET_Z = 0.431  # End effector offset (used by class FK)
EE_OFFSET_IK = 0.266   # EE offset along z for IK-only model (for fk_ur5 below)

# --------------- DLS IK helpers (new) ----------------
def _hom_from_3x4(M34: np.ndarray) -> np.ndarray:
    """3x4 (Coppelia ‘matrix’) -> 4x4 homogeneous"""
    T = np.eye(4, dtype=float)
    T[:3, :4] = M34
    return T

def _world_to_base_pos(Twb: np.ndarray, p_world: np.ndarray) -> np.ndarray:
    """Convert a world position to base frame using Twb (4x4)."""
    pw = np.ones(4); pw[:3] = p_world
    pb = Twb @ pw
    return pb[:3]

def fk_ur5(q_rad: np.ndarray) -> np.ndarray:
    """
    Simplified FK (modified DH, 6 links + EE offset along z) used by DLS-IK.
    This is independent from the class-based FK; keeps IK stable/continuous.
    """
    # alpha, a, d for each joint (modified-DH compatible)
    DH = np.array([
        [0.0,         0.0,      0.0 ],
        [np.pi/2,     0.0,      0.0 ],
        [0.0,         0.4251,   0.0 ],
        [0.0,         0.39215,  0.11],
        [-np.pi/2,    0.0,      0.09475],
        [np.pi/2,     0.0,      0.0 ],
    ], dtype=float)
    th_offset = [0, np.pi/2, 0, -np.pi/2, 0, 0]

    T = np.eye(4, dtype=float)
    for i in range(6):
        alpha, a, d = DH[i]
        theta = q_rad[i] + th_offset[i]
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        T_i = np.array([
            [ct,   -st,   0.0, a],
            [st*ca, ct*ca, -sa, -d*sa],
            [st*sa, ct*sa,  ca,  d*ca],
            [0,     0,     0,   1]
        ], dtype=float)
        T = T @ T_i
    # simple EE offset on +z of current tool frame:
    T[:3, 3] += T[:3, 2] * EE_OFFSET_IK
    return T

def so3_log(R: np.ndarray) -> np.ndarray:
    cos_th = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    th = np.arccos(cos_th)
    if th < 1e-6:
        return np.zeros(3)
    w_hat = (R - R.T) * (0.5 / np.sin(th))
    return th * np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]])

def pose_error(T_cur: np.ndarray, T_goal: np.ndarray) -> np.ndarray:
    p_cur, R_cur = T_cur[:3, 3], T_cur[:3, :3]
    p_des, R_des = T_goal[:3, 3], T_goal[:3, :3]
    dp = p_des - p_cur
    dw = so3_log(R_des @ R_cur.T)
    return np.hstack([dp, dw])

def ik_dls_step(q: np.ndarray, T_goal: np.ndarray, lam=0.05, pos_w=1.0, rot_w=0.6,
                dq_max=np.deg2rad(5)) -> Tuple[np.ndarray, np.ndarray]:
    """
    One DLS-IK iteration using class Jacobian (geometric) and FK above.
    """
    # Error
    T = fk_ur5(q)
    e = pose_error(T, T_goal)
    W = np.diag([pos_w, pos_w, pos_w, rot_w, rot_w, rot_w])
    ew = W @ e

    # Jacobian from current q using class Kinematics (expects deg in DH table)
    th_deg = {i: np.rad2deg(q[i]) for i in range(6)}
    J = Kinematics.compute_jacobian(DHTable.create(th_deg))  # 6x6
    Jw = W @ J

    # DLS
    A = Jw @ Jw.T + (lam**2) * np.eye(6)
    dq = Jw.T @ np.linalg.solve(A, ew)

    # limit per-iteration step
    m = np.max(np.abs(dq))
    if m > dq_max:
        dq = dq * (dq_max / m)

    return q + dq, e

def move_to_pose_ik(q_start: np.ndarray, T_goal: np.ndarray, hdl_j: List[int],
                    max_iter=300, lam=0.1, pos_w=1.0, rot_w=0.6,
                    pos_tol=1e-3, rot_tol=1e-2) -> np.ndarray:
    """
    Iterative IK toward T_goal; streams targets each step to the sim.
    """
    print("Moving with DLS-IK to target...")
    q = np.array(q_start, dtype=float)
    for k in range(max_iter):
        q, e = ik_dls_step(q, T_goal, lam=lam, pos_w=pos_w, rot_w=rot_w)
        for i in range(6):
            sim.setJointTargetPosition(hdl_j[i], float(q[i]))
        if np.linalg.norm(e[:3]) < pos_tol and np.linalg.norm(e[3:]) < rot_tol:
            print(f"Reached in {k+1} iters. PosErr={np.linalg.norm(e[:3]):.4f}, RotErr={np.linalg.norm(e[3:]):.4f}")
            return q
        sim.switchThread()
    print("IK: not fully converged.")
    return q

# --------------- Class modules (your originals, mostly unchanged) ---------------
class DHTable:
    @staticmethod
    def create(joint_angles_deg: dict) -> np.ndarray:
        dh_table = DH_BASE_PARAMS.copy()
        for index in range(len(dh_table)):
            dh_table[index, 3] += joint_angles_deg[index]
        return dh_table

class Kinematics:
    @staticmethod
    def forward_kinematics(dh_table: np.ndarray, start: int, stop: int) -> np.ndarray:
        T_init = np.eye(4)
        T_cal = np.eye(4)
        t_6e = np.eye(4)
        t_6e[2, 3] = TOOL_OFFSET_Z
        for i in range(start, stop):
            alpha = np.radians(dh_table[i, 0])
            theta = np.radians(dh_table[i, 3])
            T_cal[0, :] = [np.cos(theta), -np.sin(theta), 0, dh_table[i, 1]]
            T_cal[1, :] = [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*dh_table[i, 2]]
            T_cal[2, :] = [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha),  np.cos(alpha),  np.cos(alpha)*dh_table[i, 2]]
            T_cal[3, :] = [0, 0, 0, 1]
            T_init = T_init @ T_cal
        return T_init @ t_6e

    @staticmethod
    def extract_pose(homo_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = homo_mat[:3, 3]
        rot = homo_mat[:3, :3]
        ori = np.array([
            np.arctan2(-homo_mat[1, 2], homo_mat[2, 2]),
            np.arcsin(homo_mat[0, 2]),
            np.arctan2(-homo_mat[0, 1], homo_mat[0, 0]),
        ])
        return pos, rot, ori

    @staticmethod
    def compute_jacobian(dh_table: np.ndarray, type_joint: np.ndarray = None) -> np.ndarray:
        if type_joint is None:
            type_joint = np.zeros(6)

        Ts = [np.eye(4)]
        T  = np.eye(4)
        for i in range(6):
            alpha = np.radians(dh_table[i, 0])
            a     = dh_table[i, 1]
            d     = dh_table[i, 2]
            th    = np.radians(dh_table[i, 3])
            ca, sa = np.cos(alpha), np.sin(alpha)
            ct, st = np.cos(th),    np.sin(th)
            T = T @ np.array([
                [ct, -st, 0.0, a],
                [st*ca, ct*ca, -sa, -sa*d],
                [st*sa, ct*sa,  ca,  ca*d],
                [0, 0, 0, 1]
            ], dtype=float)
            Ts.append(T.copy())

        T_6e = np.eye(4); T_6e[2,3] = TOOL_OFFSET_Z
        T_e  = T @ T_6e
        p_e  = T_e[:3, 3]

        Jv = np.zeros((3, 6))
        Jw = np.zeros((3, 6))
        for i in range(6):
            z_i = Ts[i][:3, 2]
            p_i = Ts[i][:3, 3]
            Jv[:, i] = np.cross(z_i, (p_e - p_i))
            Jw[:, i] = z_i
        return np.vstack([Jv, Jw])

class TrajectoryPlanner:
    @staticmethod
    def cubic_polynomial(hdl_j: List[int], init_pos: np.ndarray, end_pos: np.ndarray,
                         end_time: float, theta: np.ndarray,
                         init_speed: float = 0.0, end_speed: float = 0.0) -> np.ndarray:
        global sim
        a0 = init_pos
        a1 = init_speed
        a2 = (3.0/(end_time**2))*(end_pos - init_pos) - (2.0/end_time)*init_speed - (1.0/end_time)*end_speed
        a3 = (-2.0/(end_time**3))*(end_pos - init_pos) + (1.0/(end_time**2))*(end_speed + init_speed)
        t0 = sim.getSimulationTime()
        theta_last = init_pos.copy()
        while True:
            t = sim.getSimulationTime() - t0
            if t >= end_time:
                ut = end_pos
                for j, h in enumerate(hdl_j):
                    sim.setJointTargetPosition(h, float(ut[j]))
                theta_last = ut
                break
            ut = a0 + a1*t + a2*(t**2) + a3*(t**3)
            for j, h in enumerate(hdl_j):
                sim.setJointTargetPosition(h, float(ut[j]))
            theta_last = ut
            sim.wait(0)
        return theta_last

class Gripper:
    @staticmethod
    def gripper_action(hdl_gripper, grip_action):
        # 1 = close, 2 = open  (ตามโค้ดเดิมของคุณ)
        if grip_action == 1:
            sim.setJointTargetForce(hdl_gripper, -20)
            sim.setJointTargetVelocity(hdl_gripper, -0.2)
        elif grip_action == 2:
            sim.setJointTargetForce(hdl_gripper, 3)
            sim.setJointTargetVelocity(hdl_gripper, 0.2)

    @staticmethod
    def get_box_poses():
        h_box = sim.getObject('/Cuboid')
        p_box = np.array(sim.getObjectPosition(h_box, -1), dtype=float)
        return h_box, p_box

    @staticmethod
    def get_current_endpose(hdl_j) -> np.ndarray:
        th_deg = {i: np.rad2deg(sim.getJointPosition(hdl_j[i])) for i in range(6)}
        dh_tbl = DHTable.create(th_deg)
        T_0E = Kinematics.forward_kinematics(dh_tbl, 0, 6)
        pos, _, ori = Kinematics.extract_pose(T_0E)
        return np.hstack([pos, ori]).astype(float)

    @staticmethod
    def move_cubic_to(hdl_j, q_target_rad: np.ndarray, end_time: float = 3.0):
        q_now = np.array([sim.getJointPosition(h) for h in hdl_j], dtype=float)
        TrajectoryPlanner.cubic_polynomial(
            hdl_j,
            init_pos=q_now,
            end_pos=q_target_rad,
            end_time=end_time,
            theta=q_now,
        )

    @staticmethod
    def wait_steps(sec: float = 0.5):
        t0 = time.time()
        while time.time() - t0 < sec:
            sim.switchThread()

# ---------------------------- CoppeliaSim Callbacks ----------------------------
def sysCall_init():
    global sim
    sim = require("sim")
    print("[INIT] Python module loaded.")

def sysCall_thread():
    global sim

    # ----- Get handles -----
    hdl_j = []
    hdl_j.append(sim.getObject("/UR5/joint"))
    hdl_j.append(sim.getObject("/UR5/joint/link/joint"))
    hdl_j.append(sim.getObject("/UR5/joint/link/joint/link/joint"))
    hdl_j.append(sim.getObject("/UR5/joint/link/joint/link/joint/link/joint"))
    hdl_j.append(sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint"))
    hdl_j.append(sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint"))
    hdl_base     = sim.getObject("/UR5")
    hdl_end      = sim.getObject("/UR5/EndPoint")
    hdl_gripper  = sim.getObject("/UR5/RG2/openCloseJoint")
    hdl_attach   = None
    try:
        hdl_attach = sim.getObject("/UR5/RG2/attachPoint")
    except:
        pass

    # ----- Box -----
    h_box, p_box_world = Gripper.get_box_poses()
    print(f"Box Position (world): {p_box_world}")

    # Open gripper first
    Gripper.gripper_action(hdl_gripper, 2)
    Gripper.wait_steps(0.3)

    # ----- World <-> Base transforms -----
    Tbw_34 = np.array(sim.getObjectMatrix(hdl_base, -1)).reshape(3,4)  # base wrt world
    Tbw = _hom_from_3x4(Tbw_34)
    Twb = np.linalg.inv(Tbw)

    # Current joints
    q_now = np.array([sim.getJointPosition(h) for h in hdl_j], dtype=float)
    R_now_base = fk_ur5(q_now)[:3, :3]  # use IK-FK for current orientation (in base)

    # ===== Sequence =====
    # 1) Above box (+0.10 m along world Z)
    p_above_world = p_box_world + np.array([0, 0, 0.10])
    p_above_base  = _world_to_base_pos(Twb, p_above_world)

    T_goal_above = np.eye(4)
    T_goal_above[:3, 3]  = p_above_base
    T_goal_above[:3, :3] = R_now_base  # keep current EE orientation (avoid big spin)

    q_now = move_to_pose_ik(q_now, T_goal_above, hdl_j, max_iter=250)
    Gripper.wait_steps(0.2)

    # 2) Touch box (go to exact box Z, still in base frame)
    p_touch_base = _world_to_base_pos(Twb, p_box_world)
    T_goal_touch = np.eye(4)
    T_goal_touch[:3, 3]  = p_touch_base
    T_goal_touch[:3, :3] = R_now_base
    q_now = move_to_pose_ik(q_now, T_goal_touch, hdl_j, max_iter=300)
    Gripper.wait_steps(0.2)

    # 3) Close gripper (clamp)
    Gripper.gripper_action(hdl_gripper, 1)
    Gripper.wait_steps(0.8)

    # (optional) parent the box to attach point so it moves with gripper
    if hdl_attach is not None:
        try:
            sim.setObjectParent(h_box, hdl_attach, True)
        except:
            pass

    # 4) Lift (+0.15 m)
    p_lift_world = p_box_world + np.array([0, 0, 0.15])
    p_lift_base  = _world_to_base_pos(Twb, p_lift_world)
    T_goal_lift = np.eye(4)
    T_goal_lift[:3, 3]  = p_lift_base
    T_goal_lift[:3, :3] = R_now_base
    q_now = move_to_pose_ik(q_now, T_goal_lift, hdl_j, max_iter=250)
    Gripper.wait_steps(0.3)

    print("Pick & lift complete.")
    sim.switchThread()

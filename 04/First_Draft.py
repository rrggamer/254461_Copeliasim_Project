# -*- coding: utf-8 -*-
"""
UR5 Robot Control Module
Handles forward/inverse kinematics and trajectory planning for UR5 robot in CoppeliaSim
"""

import time
import numpy as np
from typing import List, Tuple

# Constants
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

TOOL_OFFSET_Z = 0.431  # End effector offset


class DHTable:
    """Denavit-Hartenberg parameter table manager"""
    
    @staticmethod
    def create(joint_angles_deg: dict) -> np.ndarray:
        """
        Create DH table with joint angles applied
        
        Args:
            joint_angles_deg: Dictionary of joint angles in degrees
            
        Returns:
            6x4 DH parameter table
        """
        dh_table = DH_BASE_PARAMS.copy()
        for index in range(len(dh_table)):
            dh_table[index, 3] += joint_angles_deg[index]
        return dh_table


class Kinematics:
    """Forward and inverse kinematics solver"""
    
    @staticmethod
    def forward_kinematics(dh_table: np.ndarray, start: int, stop: int) -> np.ndarray:
        """
        Compute forward kinematics using DH parameters
        
        Args:
            dh_table: DH parameter table
            start: Starting joint index
            stop: Ending joint index
            
        Returns:
            4x4 homogeneous transformation matrix
        """
        T_init = np.eye(4)
        T_cal = np.eye(4)
        t_6e = np.eye(4)
        t_6e[2, 3] = TOOL_OFFSET_Z
        
        for i in range(start, stop):
            alpha = np.radians(dh_table[i, 0])
            theta = np.radians(dh_table[i, 3])
            
            T_cal[0, :] = [
                np.cos(theta),
                -np.sin(theta),
                0,
                dh_table[i, 1],
            ]
            T_cal[1, :] = [
                np.sin(theta) * np.cos(alpha),
                np.cos(theta) * np.cos(alpha),
                -np.sin(alpha),
                -np.sin(alpha) * dh_table[i, 2],
            ]
            T_cal[2, :] = [
                np.sin(theta) * np.sin(alpha),
                np.cos(theta) * np.sin(alpha),
                np.cos(alpha),
                np.cos(alpha) * dh_table[i, 2],
            ]
            T_cal[3, :] = [0, 0, 0, 1]
            
            T_init = T_init @ T_cal
        
        return T_init @ t_6e
    
    @staticmethod
    def extract_pose(homo_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract position, rotation matrix, and orientation from transform
        
        Args:
            homo_mat: 4x4 homogeneous transformation matrix
            
        Returns:
            Tuple of (position, rotation_matrix, euler_angles)
        """
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

        # Forward transforms from BASE -> i (0..6)
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

        # Base -> EE
        T_6e = np.eye(4); T_6e[2,3] = TOOL_OFFSET_Z
        T_e  = T @ T_6e
        p_e  = T_e[:3, 3]

        Jv = np.zeros((3, 6))
        Jw = np.zeros((3, 6))
        for i in range(6):
            z_i = Ts[i][:3, 2]      # z-axis of joint i in BASE
            p_i = Ts[i][:3, 3]      # origin of joint i in BASE
            Jv[:, i] = np.cross(z_i, (p_e - p_i))  # revolute
            Jw[:, i] = z_i
        return np.vstack([Jv, Jw])
    
    
    @staticmethod
    def _wrap_to_pi(a):
        # map any angle to (-pi, pi]
        return (a + np.pi) % (2*np.pi) - np.pi
    
    
    @staticmethod
    def _euler_xyz_to_R(rx, ry, rz):
        cx, cy, cz = np.cos([rx, ry, rz])
        sx, sy, sz = np.sin([rx, ry, rz])
        return np.array([
            [cy*cz, cz*sx*sy - cx*sz, sx*sz + cx*cz*sy],
            [cy*sz, cx*cz + sx*sy*sz, cx*sy*sz - cz*sx],
            [-sy,   cy*sx,             cx*cy]
        ], dtype=float)
        
    @staticmethod
    def _rot_to_rvec(R):
        tr = np.trace(R)
        c  = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
        th = np.arccos(c)
        if th < 1e-9: 
            return np.zeros(3)
        w = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])/(2*np.sin(th))
        return w * th
    
    @staticmethod
    def inverse_kinematics(hdl_j, end_pos: np.ndarray, init_theta: np.ndarray,
                       tolerance: float = 1e-4, alpha: float = 1.0,  # alpha now implicit in DLS
                       lambda_dls: float = 1e-3) -> np.ndarray:
        """
        One iterative IK update using Damped Least Squares on full 6D pose.
        - Uses a small orientation weight to suppress wrist spinning.
        - Keeps each step continuous by wrapping delta angles around the current.
        """
        # 1) Build DH from the *current guess* (do NOT read sim here)
        th_deg = {i: np.rad2deg(init_theta[i]) for i in range(6)}
        dh_table = DHTable.create(th_deg)

        # 2) Current pose from FK
        fk = Kinematics.forward_kinematics(dh_table, 0, 6)
        p_now, _, _ = Kinematics.extract_pose(fk)
        R_now = fk[:3, :3]

        # 3) Target pose from end_pos (x,y,z, rx,ry,rz)
        p_tgt = end_pos[:3]
        rx, ry, rz = end_pos[3:]
        R_tgt = Kinematics._euler_xyz_to_R(rx, ry, rz)

        # 4) Pose error: position + orientation (log map)
        e_pos = (p_tgt - p_now)
        e_ori = Kinematics._rot_to_rvec(R_now.T @ R_tgt)  # zero if you kept current orientation
        # weight orientation lightly to prevent wrist windup
        W = np.diag([1, 1, 1, 0.2, 0.2, 0.2])

        e = np.hstack([e_pos, e_ori])

        # 5) Geometric Jacobian at current guess
        J = Kinematics.compute_jacobian(dh_table)  # 6x6

        # 6) Damped Least Squares: (J^T W J + ?^2 I) dq = J^T W e
        A = J.T @ W @ J + (lambda_dls**2) * np.eye(6)
        b = J.T @ W @ e
        dq = np.linalg.solve(A, b)

        # 7) Limit per-iteration step and keep continuity around current
        dq = np.clip(dq, -np.deg2rad(5.0), np.deg2rad(5.0))
        new_theta = init_theta + dq
        # wrap each joint to be the nearest equivalent to the previous angle
        for i in range(6):
            new_theta[i] = init_theta[i] + Kinematics._wrap_to_pi(new_theta[i] - init_theta[i])

        # 8) Stop criterion (position only)
        if np.mean(np.abs(e_pos)) < tolerance:
            return new_theta
        return new_theta


class TrajectoryPlanner:
    """Trajectory planning with cubic polynomial and linear parabolic blend"""
    
    @staticmethod
    def cubic_polynomial(hdl_j: List, init_pos: np.ndarray, end_pos: np.ndarray,
                     end_time: float, theta: np.ndarray,
                     init_speed: float = 0.0, end_speed: float = 0.0) -> np.ndarray:
        """
        Joint-space cubic trajectory using SIMULATION time.
        Returns final joint configuration actually sent.
        """
        global sim

        # coeffs (vectorized)
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

            # cubic
            ut = a0 + a1*t + a2*(t**2) + a3*(t**3)

            for j, h in enumerate(hdl_j):
                sim.setJointTargetPosition(h, float(ut[j]))
            theta_last = ut

            sim.wait(0)  # หรือ sim.switchThread()

        return theta_last
    
    @staticmethod
    def linear_parabolic_blend(hdl_j: List, init_pos: np.ndarray, end_pos: np.ndarray,
                               end_time: float, theta: np.ndarray, accel: float = 0.005,
                               end: bool = False) -> np.ndarray:
        """
        Execute linear parabolic blend trajectory in joint space
        
        Args:
            hdl_j: Joint handles
            init_pos: Starting joint positions
            end_pos: Ending joint positions
            end_time: Segment duration
            theta: Current joint angles
            accel: Acceleration value
            end: Whether this is the final segment
            
        Returns:
            Final joint configuration
        """
        global sim
        
        print(f"go to {end_pos}")
        
        # Calculate blend time
        blended_t = (end_time / 2) + (
            np.sqrt((accel**2) * (end_time**2))
            - ((4 * accel) * (end_pos - init_pos)) / (2 * accel)
        )
        blended_t = max(blended_t)
        
        start_t = time.time()
        theta_bc = init_pos.copy()
        
        while True:
            elapsed = time.time() - start_t
            
            if elapsed < blended_t:
                # Acceleration phase
                ut = init_pos + (0.5 * accel * (elapsed**2))
                
                for count, hdl in enumerate(hdl_j):
                    sim.setJointTargetPosition(hdl, ut[count])
                
                theta_bc = ut
                
            elif blended_t <= elapsed < end_time - blended_t:
                # Constant velocity phase
                ut = (
                    init_pos
                    + (0.5 * accel * (blended_t**2))
                    + (accel * blended_t * (elapsed - blended_t))
                )
                
                for count, hdl in enumerate(hdl_j):
                    sim.setJointTargetPosition(hdl, ut[count])
                
                theta_bc = ut
                
            elif end_time - blended_t <= elapsed < end_time:
                # Deceleration phase
                ut = (
                    end_pos
                    - (0.5 * accel * (end_time**2))
                    + (accel * end_time * elapsed)
                    - (0.5 * accel * (elapsed**2))
                )
                
                for count, hdl in enumerate(hdl_j):
                    sim.setJointTargetPosition(hdl, ut[count])
                
                theta_bc = ut
                
            else:
                # End of trajectory
                if not end:
                    break
                for count, hdl in enumerate(hdl_j):
                    sim.setJointTargetPosition(hdl, theta_bc[count])
            
            sim.switchThread()
        
        return theta_bc
    
    @staticmethod
    def execute_path(hdl_j: List, positions: np.ndarray, speeds: List[float], 
                    loop: bool = False):
        """
        Execute multi-segment trajectory using linear parabolic blend
        
        Args:
            hdl_j: Joint handles
            positions: Array of joint configurations
            speeds: Duration for each segment
            loop: Whether to loop indefinitely
        """
        theta = np.zeros(6)
        
        while True:
            for count in range(len(positions) - 1):
                theta = TrajectoryPlanner.linear_parabolic_blend(
                    hdl_j,
                    positions[count],
                    positions[count + 1],
                    speeds[count],
                    theta,
                )
                
                
                
class Gripper:
    
    
    
    @staticmethod
    def gripper_action(hdl_gripper, grip_action):
        if grip_action == 1:
            sim.setJointTargetForce(hdl_gripper,-20)
            sim.setJointTargetVelocity(hdl_gripper,-0.2)
        elif grip_action == 2:
            sim.setJointTargetForce(hdl_gripper,3)
            sim.setJointTargetVelocity(hdl_gripper,0.2)
           
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
    def build_target_endpos_from_point(hdl_j, p_world: np.ndarray,
                                    z_offset: float = 0.06,
                                    keep_current_orientation: bool = True,
                                    euler_override: np.ndarray = None) -> np.ndarray:
        x, y, z = float(p_world[0]), float(p_world[1]), float(p_world[2] + z_offset)

        if euler_override is not None:
            rx, ry, rz = map(float, euler_override)
        elif keep_current_orientation:
            cur = Gripper.get_current_endpose(hdl_j)
            rx, ry, rz = float(cur[3]), float(cur[4]), float(cur[5])
        else:
            rx = ry = rz = 0.0

        return np.array([x, y, z, rx, ry, rz], dtype=float)

    @staticmethod
    def solve_ik_to_point(hdl_j,
                      p_world: np.ndarray,
                      z_offset: float = 0.06,
                      iters: int = 100,
                      keep_current_orientation: bool = True,
                      euler_override: np.ndarray = None) -> np.ndarray:

        end_pos = Gripper.build_target_endpos_from_point(
            hdl_j, p_world,
            z_offset=z_offset,
            keep_current_orientation=keep_current_orientation,
            euler_override=euler_override
        )

        theta = np.array([sim.getJointPosition(h) for h in hdl_j], dtype=float)
        for i in range(150):
            theta_prev = theta.copy()
            theta = Kinematics.inverse_kinematics(hdl_j, end_pos, theta)
            if np.linalg.norm(theta - theta_prev) < 1e-4:
                print(f"IK converged in {i} iterations")
                break
        return theta

    @staticmethod
    def move_cubic_to(hdl_j, q_target_rad: np.ndarray, end_time: float = 3.0):
        q_now = np.array([sim.getJointPosition(h) for h in hdl_j], dtype=float)
        print("Move from:", np.degrees(q_now))
        print("To:", np.degrees(q_target_rad))
        TrajectoryPlanner.cubic_polynomial(
            hdl_j,
            init_pos=q_now,
            end_pos=q_target_rad,
            end_time=end_time,
            theta=q_now,
        )

        # Debug EE position every step
        h_robot = sim.getObject('/UR5/EndPoint')
        for _ in range(30):
            pos = sim.getObjectPosition(h_robot, -1)
            print(f"EE pos: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
            sim.wait(0.1)
    @staticmethod
    def wait_steps(sec: float = 0.5):
        t0 = time.time()
        while time.time() - t0 < sec:
            sim.switchThread()

    @staticmethod
    def pose_to_matrix(end_pos: np.ndarray) -> np.ndarray:
        x, y, z, rx, ry, rz = end_pos
        cx, cy, cz = np.cos([rx, ry, rz])
        sx, sy, sz = np.sin([rx, ry, rz])
        R = np.array([
            [cy*cz, cz*sx*sy - cx*sz, sx*sz + cx*cz*sy],
            [cy*sz, cx*cz + sx*sy*sz, cx*sy*sz - cz*sx],
            [-sy,   cy*sx,             cx*cy]
        ], dtype=float)
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3]  = [x, y, z]
        return T


# ---------------------------- CoppeliaSim Callbacks ----------------------------

def sysCall_init():
    global sim
    sim = require("sim")


def sysCall_thread():
    global sim
    
    # Define handles for joints And Gripper
    hdl_j = []
    hdl_j.append(sim.getObject("/UR5/joint"))
    hdl_j.append(sim.getObject("/UR5/joint/link/joint"))
    hdl_j.append(sim.getObject("/UR5/joint/link/joint/link/joint"))
    hdl_j.append(sim.getObject("/UR5/joint/link/joint/link/joint/link/joint"))
    hdl_j.append(sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint"))
    hdl_j.append(sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint")) 
    hdl_end = sim.getObject("/UR5/EndPoint")
    hdl_gripper = sim.getObject("/UR5/RG2/openCloseJoint")
    sim.setJointTargetForce(hdl_gripper,3)
    sim.setJointTargetVelocity(hdl_gripper,0.2)
   
   
    # ---------- Get target and pose ----------
    
    h_robot = sim.getObject('/UR5/EndPoint')
    h_box, p_box = Gripper.get_box_poses()
    print(f"Box Position : {p_box}")
    
    
     # ตั้งค่าท่าปัจจุบัน
    q_now = np.array([sim.getJointPosition(h) for h in hdl_j.values()], float)
    T_now = fk_ur5(q_now)

    # จุดเป้าหมาย
    p_box = np.array(sim.getObjectPosition(hdl_box, -1))
    T_goal = np.eye(4)
    T_goal[:3, 3] = p_box + np.array([0, 0, 0.10])  
    T_goal[:3, :3] = T_now[:3, :3]                  

    # ขยับด้วย iterative IK
    q_new = move_to_pose_ik(q_now, T_goal, hdl_j)

    # หนีบกล่อง
    gripper_action(hdl_grip, 2)
    sim.switchThread()
   
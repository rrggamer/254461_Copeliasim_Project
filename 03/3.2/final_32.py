# -*- coding: utf-8 -*-
"""
UR5 Robot Control Module
Handles forward/inverse kinematics and Cartesian trajectory planning for UR5 robot in CoppeliaSim
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
        """
        Compute the Jacobian matrix
        
        Args:
            dh_table: DH parameter table
            type_joint: Array indicating joint type (0=revolute, 1=prismatic)
            
        Returns:
            6xN Jacobian matrix
        """
        if type_joint is None:
            type_joint = np.zeros(6)
        
        j_array = []
        for index in range(len(dh_table)):
            homo_mat = Kinematics.forward_kinematics(dh_table, index, len(dh_table))
            k_0i = Kinematics.extract_pose(homo_mat)[1] @ np.array([0, 0, 1])
            r_ie = Kinematics.extract_pose(homo_mat)[0]
            j_1 = ((1 - type_joint[index]) * np.cross(k_0i, r_ie)) + (
                type_joint[index] * k_0i
            )
            j_2 = (1 - type_joint[index]) * k_0i
            j_array.append(j_1 + j_2)
        
        return np.array(j_array).T
    
    @staticmethod
    def inverse_kinematics(hdl_j, end_pos: np.ndarray, init_theta: np.ndarray,
                          tolerance: float = 1e-4, alpha: float = 0.1) -> np.ndarray:
        """
        Compute inverse kinematics using Jacobian pseudo-inverse
        
        Args:
            hdl_j: Joint handles from simulator
            end_pos: Target end effector pose [x, y, z, rx, ry, rz]
            init_theta: Current joint angles in radians
            tolerance: Convergence tolerance
            alpha: Step size for gradient descent
            
        Returns:
            Updated joint angles in radians
        """
        global sim
        
        th = {}
        for i in range(6):
            th[i] = round(sim.getJointPosition(hdl_j[i]) * RAD_TO_DEG, 2)
            init_theta[i] = np.deg2rad(th[i])
        
        dh_table = DHTable.create(th)
        fk = Kinematics.forward_kinematics(dh_table, 0, 6)
        pos, rot, ori = Kinematics.extract_pose(fk)
        init_pos = np.hstack((pos, ori))
        
        jacobian = Kinematics.compute_jacobian(dh_table)
        jacobian_inv = np.linalg.pinv(jacobian)
        
        error_pos = init_pos - end_pos
        error_pos[1] = init_pos[1] - end_pos[1]
        
        deltha_theta = alpha * (jacobian_inv @ error_pos[:3])
        init_theta += deltha_theta
        
        if np.mean(error_pos) < tolerance:
            return init_theta
        
        return init_theta


class TrajectoryPlanner:
    """Cubic polynomial trajectory planning in Cartesian space"""
    
    @staticmethod
    def compute_cubic_coefficients(init_pos: np.ndarray, end_pos: np.ndarray, 
                                   end_time: float, init_speed: float = 0, 
                                   end_speed: float = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute cubic polynomial coefficients for smooth trajectory
        
        Args:
            init_pos: Starting position array
            end_pos: Ending position array
            end_time: Time duration
            init_speed: Starting velocity
            end_speed: Ending velocity
            
        Returns:
            Tuple of coefficient arrays (a0, a1, a2, a3)
        """
        a0 = init_pos
        a1 = init_speed
        a2 = (
            (3 / (end_time**2)) * (end_pos - init_pos)
            - ((3 / end_time) * init_speed)
            - ((1 / end_time) * end_speed)
        )
        a3 = ((-2 / (end_time**3)) * (end_pos - init_pos)) + (
            (1 / (end_time**2)) * (end_speed + 2 * init_speed)
        )
        return a0, a1, a2, a3
    
    @staticmethod
    def execute_cubic_segment(hdl_j: List, init_pos: np.ndarray, end_pos: np.ndarray,
                             end_time: float, theta: np.ndarray, end: bool = False,
                             init_speed: float = 0, end_speed: float = 0) -> np.ndarray:
        """
        Execute a single Cartesian trajectory segment using inverse kinematics
        
        Args:
            hdl_j: Joint handles
            init_pos: Starting Cartesian pose
            end_pos: Ending Cartesian pose
            end_time: Segment duration
            theta: Current joint angles
            end: Whether this is the final segment
            init_speed: Starting velocity
            end_speed: Ending velocity
            
        Returns:
            Final joint configuration
        """
        global sim
        
        print(f"Moving To  {end_pos}")
        
        a0, a1, a2, a3 = TrajectoryPlanner.compute_cubic_coefficients(
            init_pos, end_pos, end_time, init_speed, end_speed
        )
        
        start_t = time.time()
        theta_bc = theta.copy()
        
        while True:
            elapsed = time.time() - start_t
            
            if elapsed > end_time:
                if not end:
                    break
                for count, hdl in enumerate(hdl_j):
                    if count == 5:
                        sim.setJointTargetPosition(hdl, 0)
                    else:
                        sim.setJointTargetPosition(hdl, theta_bc[count])
            else:
                t = elapsed
                ut = a0 + a1 * t + a2 * (t**2) + a3 * (t**3)
                theta = Kinematics.inverse_kinematics(hdl_j, ut, theta)
                
                for count, hdl in enumerate(hdl_j):
                    if count == 5:
                        sim.setJointTargetPosition(hdl, 0)
                    else:
                        sim.setJointTargetPosition(hdl, theta[count])
                
                theta_bc = theta.copy()
            
            sim.switchThread()
        
        return theta
    
    @staticmethod
    def execute_path(hdl_j: List, positions: np.ndarray, speeds: List[float], 
                    loop: bool = False):
        """
        Execute multi-segment Cartesian trajectory
        
        Args:
            hdl_j: Joint handles
            positions: Array of Cartesian poses [x, y, z, rx, ry, rz]
            speeds: Duration for each segment
            loop: Whether to loop indefinitely
        """
        theta = np.zeros(6)
        
        while True:
            for count in range(len(positions) - 1):
                theta = TrajectoryPlanner.execute_cubic_segment(
                    hdl_j,
                    positions[count],
                    positions[count + 1],
                    speeds[count],
                    theta,
                )


# ---------------------------- CoppeliaSim Callbacks ----------------------------

def sysCall_init():
    """Initialize simulation"""
    global sim
    sim = require("sim")


def sysCall_thread():
    """Main simulation thread"""
    global sim
    
    # Define handles for joints
    hdl_j = []
    hdl_j.append(sim.getObject("/UR5/joint"))
    hdl_j.append(sim.getObject("/UR5/joint/link/joint"))
    hdl_j.append(sim.getObject("/UR5/joint/link/joint/link/joint"))
    hdl_j.append(sim.getObject("/UR5/joint/link/joint/link/joint/link/joint"))
    hdl_j.append(sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint"))
    hdl_j.append(sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint"))
    
    hdl_end = sim.getObject("/UR5/EndPoint")
    
    # Initialize joint angles
    new_theta = np.zeros(6)
    
    # Define trajectory parameters (time durations)
    speeds = [10, 10, 10, 10]
    
    # Define Cartesian waypoints [x, y, z, rx, ry, rz]
    positions = [
        [-0.5, 0.3, 0.3, 0, 0, 0],
        [-0.6, 0.3, 0.3, 0, 0, 0],
        [-0.7, 0.3, 0.4, 0, 0, 0],
        [-0.8, 0.3, 0.5, 0, 0, 0],
        [-0.9, 0.3, 0.6, 0, 0, 0],
    ]
    
    # Execute Cartesian path planning
    TrajectoryPlanner.execute_path(hdl_j, np.array(positions), speeds)
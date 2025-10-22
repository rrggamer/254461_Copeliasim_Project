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
    @staticmethod
    def create(joint_angles_deg: np.ndarray) -> np.ndarray:
        dh_table = DH_BASE_PARAMS.copy()
        dh_table[:, 3] += joint_angles_deg
        return dh_table


class Kinematics:
    @staticmethod
    def forward_kinematics(dh_table: np.ndarray, start: int, stop: int) -> np.ndarray:
        transform = np.eye(4)
        
        for i in range(start, stop):
            alpha = np.radians(dh_table[i, 0])
            a = dh_table[i, 1]
            d = dh_table[i, 2]
            theta = np.radians(dh_table[i, 3])
            
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
            
            T_i = np.array([
                [cos_theta, -sin_theta, 0, a],
                [sin_theta * cos_alpha, cos_theta * cos_alpha, -sin_alpha, -sin_alpha * d],
                [sin_theta * sin_alpha, cos_theta * sin_alpha,  cos_alpha,  cos_alpha * d],
                [0, 0, 0, 1]
            ])
            
            transform = transform @ T_i
        
        tool_transform = np.eye(4)
        tool_transform[2, 3] = TOOL_OFFSET_Z
        
        return transform @ tool_transform
    
    @staticmethod
    def extract_pose(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        position = transform[:3, 3]
        rotation = transform[:3, :3]
        
        
        orientation = np.array([
            np.arctan2(-transform[1, 2], transform[2, 2]),
            np.arcsin(transform[0, 2]),
            np.arctan2(-transform[0, 1], transform[0, 0]),
        ])
        
        return position, rotation, orientation
    
    @staticmethod
    def compute_jacobian(dh_table: np.ndarray, joint_types: np.ndarray = None) -> np.ndarray:
        if joint_types is None:
            joint_types = np.zeros(len(dh_table))
        
        jacobian_columns = []
        
        for i in range(len(dh_table)):
            T_i_e = Kinematics.forward_kinematics(dh_table, i, len(dh_table))
            _, rotation_i_e, _ = Kinematics.extract_pose(T_i_e)
            
            # Joint axis in base frame
            k_0i = rotation_i_e @ np.array([0, 0, 1])
            
            # Position vector from joint i to end effector
            r_ie = Kinematics.extract_pose(T_i_e)[0]
            
            # Compute linear and angular velocity contributions
            j_linear = ((1 - joint_types[i]) * np.cross(k_0i, r_ie)) + (joint_types[i] * k_0i)
            j_angular = (1 - joint_types[i]) * k_0i
            
            jacobian_columns.append(np.concatenate([j_linear, j_angular]))
        
        return np.array(jacobian_columns).T
    
    @staticmethod
    def inverse_kinematics(hdl_joints, target_pose: np.ndarray, current_angles: np.ndarray,
                          tolerance: float = 1e-4, step_size: float = 0.1) -> np.ndarray:
        global sim
        
        # Get current joint angles from simulator
        joint_angles_deg = {}
        for i in range(6):
            joint_angles_deg[i] = round(sim.getJointPosition(hdl_joints[i]) * RAD_TO_DEG, 2)
            current_angles[i] = np.deg2rad(joint_angles_deg[i])
        
        # Compute forward kinematics
        dh_table = DHTable.create(np.array([joint_angles_deg[i] for i in range(6)]))
        transform = Kinematics.forward_kinematics(dh_table, 0, 6)
        position, _, orientation = Kinematics.extract_pose(transform)
        current_pose = np.hstack((position, orientation))
        
        # Compute Jacobian and its pseudo-inverse
        jacobian = Kinematics.compute_jacobian(dh_table)
        jacobian_pinv = np.linalg.pinv(jacobian)
        
        # Compute error and update
        error = current_pose - target_pose
        error[1] = current_pose[1] - target_pose[1]  # Preserve original behavior
        delta_theta = step_size * (jacobian_pinv @ error[:3])
        current_angles += delta_theta
        
        if np.mean(error) < tolerance:
            return current_angles
        
        return current_angles


class TrajectoryPlanner:
    """Cubic polynomial trajectory planning"""
    
    @staticmethod
    def compute_cubic_coefficients(p0: float, p1: float, duration: float,
                                   v0: float = 0.0, v1: float = 0.0) -> Tuple[float, float, float, float]:
        """
        Compute cubic polynomial coefficients for smooth trajectory
        
        Args:
            p0: Starting position
            p1: Ending position
            duration: Time duration
            v0: Starting velocity
            v1: Ending velocity
            
        Returns:
            Tuple of coefficients (a0, a1, a2, a3)
        """
        T = duration
        a0 = p0
        a1 = v0
        a2 = (3.0 / (T**2)) * (p1 - p0) - (3.0 / T) * v0 - (1.0 / T) * v1
        a3 = (-2.0 / (T**3)) * (p1 - p0) + (1.0 / (T**2)) * (v1 + 2.0 * v0)
        return a0, a1, a2, a3
    
    @staticmethod
    def evaluate_cubic(coeffs: Tuple[float, float, float, float], t: float) -> float:
        """Evaluate cubic polynomial at time t"""
        a0, a1, a2, a3 = coeffs
        return a0 + a1 * t + a2 * (t**2) + a3 * (t**3)
    
    @staticmethod
    def execute_segment(hdl_joints: List, q_start: np.ndarray, q_end: np.ndarray, 
                       duration: float) -> np.ndarray:
        """
        Execute a single trajectory segment
        
        Args:
            hdl_joints: Joint handles
            q_start: Starting joint configuration
            q_end: Ending joint configuration
            duration: Segment duration
            
        Returns:
            Final joint configuration
        """
        global sim
        
        print(f"Moving To {q_end}")
        
        # Compute coefficients for each joint
        coefficients = [
            TrajectoryPlanner.compute_cubic_coefficients(q_start[i], q_end[i], duration)
            for i in range(len(q_start))
        ]
        
        start_time = time.time()
        last_command = q_start.copy()
        
        while True:
            elapsed_time = time.time() - start_time
            
            if elapsed_time > duration:
                # Send final position
                for i, handle in enumerate(hdl_joints):
                    sim.setJointTargetPosition(handle, float(last_command[i]))
                break
            
            # Evaluate trajectory at current time
            current_position = np.array([
                TrajectoryPlanner.evaluate_cubic(coefficients[i], elapsed_time)
                for i in range(len(q_start))
            ])
            
            # Send commands to joints
            for i, handle in enumerate(hdl_joints):
                sim.setJointTargetPosition(handle, float(current_position[i]))
            
            last_command = current_position
            sim.switchThread()
        
        return last_command
    
    @staticmethod
    def execute_path(hdl_joints: List, waypoints: List[np.ndarray], durations: List[float]):
        """
        Execute multi-segment trajectory
        
        Args:
            hdl_joints: Joint handles
            waypoints: List of joint configurations
            durations: Duration for each segment
        """
        last_position = waypoints[0].copy()
        
        while True:
            for k in range(len(waypoints) - 1):
                q_start = waypoints[k]
                q_end = waypoints[k + 1]
                segment_duration = float(durations[k])
                
                last_position = TrajectoryPlanner.execute_segment(
                    hdl_joints, q_start, q_end, segment_duration
                )


# ---------------------------- CoppeliaSim Callbacks ----------------------------

def sysCall_init():
    global sim
    sim = require("sim")


def sysCall_thread():
    global sim
    
    # Get joint handles
    joint_handles = [
        sim.getObject("/UR5/joint"),
        sim.getObject("/UR5/joint/link/joint"),
        sim.getObject("/UR5/joint/link/joint/link/joint"),
        sim.getObject("/UR5/joint/link/joint/link/joint/link/joint"),
        sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint"),
        sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint"),
    ]
    
    # Get end effector handle
    end_effector_handle = sim.getObject("/UR5/EndPoint")
    
    # Define trajectory parameters
    segment_durations = [10, 10, 10, 10]
    waypoint_positions = np.random.uniform(low=0.0, high=1.884, size=(5, 6))
    
    # Execute trajectory
    TrajectoryPlanner.execute_path(joint_handles, waypoint_positions, segment_durations)
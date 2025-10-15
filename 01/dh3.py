import time
import numpy as np
import math

pi = np.pi
d2r = pi/180
r2d = 1/d2r
def dh_transform_matrix(alpha, a, d, theta):
    ca = math.cos(alpha); sa = math.sin(alpha)
    ct = math.cos(theta); st = math.sin(theta)
    A = np.array([[   ct,   -st,    0,     a],
                  [st*ca, ct*ca, -sa, -d*sa],
                  [st*sa, ct*sa,  ca,  d*ca],
                  [    0,     0,   0,     1]])
    return A
def rotmat_to_euler(R):
    EPS = 1e-9
    r02 = max(-1.0, min(1.0, float(R[0, 2])))
    beta = math.asin(r02)
    if abs(r02) < 1.0 - EPS:
        alpha = math.atan2(-R[1, 2], R[2, 2])
        gamma = math.atan2(-R[0, 1], R[0, 0])
    else:
        alpha = 0.0
        gamma = math.atan2(R[1, 0], R[1, 1])

    return np.array([alpha, beta, gamma])

def ur5_fk(theta):
    theta1, theta2, theta3, theta4, theta5, theta6 = theta
    T_01 = dh_transform_matrix(0.0      , 0.0      , 0.0892 , theta1 - np.pi/2)
    T_12 = dh_transform_matrix(np.pi/2  , 0.0      , 0.0    , theta2 + np.pi/2)
    T_23 = dh_transform_matrix(0.0      , 0.4251   , 0.0    , theta3)
    T_34 = dh_transform_matrix(0.0      , 0.39215  , 0.11   , theta4 + np.pi/2)
    T_45 = dh_transform_matrix(np.pi/2  , 0.0      , 0.09475, theta5)
    T_56 = dh_transform_matrix(-np.pi/2 , 0.0      , 0.26658, theta6)
    T_6E = np.eye(4)
    T_0E = T_01 @ T_12 @ T_23 @ T_34 @ T_45 @ T_56 @ T_6E
    position = T_0E[:3, 3]
    R = T_0E[:3, :3]
    euler_angles = rotmat_to_euler(R)
    return position, euler_angles, T_0E

def sysCall_init():
    sim=require("sim")


def sysCall_thread():
    hdl_j = {}
    hdl_j[0] = sim.getObject("/UR5/joint")
    hdl_j[1] = sim.getObject("/UR5/joint/link/joint")
    hdl_j[2] = sim.getObject("/UR5/joint/link/joint/link/joint")
    hdl_j[3] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint")
    hdl_j[4] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint")
    hdl_j[5] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint")
    
    hdl_end = sim.getObject("/UR5/EndPoint")
    
    t = 0
    t1 = time.time()
    
    print("UR5 Forward Kinematics and Orientation Display")
    print("=" * 60)
    
    while t < 10:
        base_angle = 45 * d2r * np.sin(0.01 * pi * t)
        joint_angles = np.array([
            base_angle,
            base_angle * 0.8,
            base_angle * 0.6,
            base_angle * 0.4,
            base_angle * 0.3,
            base_angle * 0.2
        ])
        
        for i in range(6):
            sim.setJointTargetPosition(hdl_j[i], joint_angles[i])
            
        actual_angles = {}
        for i in range(6):
            actual_angles[i] = sim.getJointPosition(hdl_j[i])
        
        theta_array = [actual_angles[i] for i in range(6)]
        calc_position, calc_orientation, T_0E = ur5_fk(theta_array) 
        
        sim_position = sim.getObjectPosition(hdl_end, -1)
        sim_orientation = sim.getObjectOrientation(hdl_end, -1) 
        
        calc_orientation_deg = calc_orientation * r2d
        sim_orientation_deg = np.array(sim_orientation) * r2d
        
        print(f"\nTime: {t:.2f}s")
        print("-" * 40)
        
        joint_deg = {i: round(actual_angles[i] * r2d, 2) for i in range(6)}
        print(f"Joint Angles (deg): {joint_deg}")
        
        # Calculated FK
        print("\nCALCULATED Forward Kinematics:")
        print(f"Position: X={calc_position[0]:.4f}, Y={calc_position[1]:.4f}, Z={calc_position[2]:.4f}")
        print(f"Orientation (XYZ Euler angles):")  # match CoppeliaSim
        print(f"  Roll (X): {calc_orientation_deg[0]:.2f}°")
        print(f"  Pitch (Y): {calc_orientation_deg[1]:.2f}°") 
        print(f"  Yaw (Z): {calc_orientation_deg[2]:.2f}°")
        
        # CoppeliaSim actual
        print("\nCOPPELIASIM Actual Values:")
        print(f"Position: X={sim_position[0]:.4f}, Y={sim_position[1]:.4f}, Z={sim_position[2]:.4f}")
        print(f"Orientation (XYZ Euler angles):")
        print(f"  Roll (X): {sim_orientation_deg[0]:.2f}°")
        print(f"  Pitch (Y): {sim_orientation_deg[1]:.2f}°")
        print(f"  Yaw (Z): {sim_orientation_deg[2]:.2f}°")

        # Update time
        t = time.time() - t1
        time.sleep(0.1) 
        sim.switchThread() 
        

def sysCall_actuation():
    pass


def sysCall_sensing():
    pass


def sysCall_cleanup():
    pass
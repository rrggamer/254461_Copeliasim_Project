import time
import math
import numpy as np

# ---------- Constants ----------
PI       = np.pi
DEG2RAD  = PI / 180.0
RAD2DEG  = 1.0 / DEG2RAD
EPS      = 1e-9

# ---------- Math helpers ----------
def dh_transform(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    """Denavit?Hartenberg transform."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -d * sa],
        [st * sa, ct * sa,  ca,  d * ca],
        [0.0, 0.0, 0.0, 1.0],
    ])

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def euler_from_rotmat_xyz(R: np.ndarray) -> np.ndarray:
    """XYZ (roll, pitch, yaw) intrinsic Euler angles from rotation matrix."""
    r02 = clamp(float(R[0, 2]), -1.0, 1.0)
    beta = math.asin(r02)
    if abs(r02) < 1.0 - EPS:
        alpha = math.atan2(-R[1, 2], R[2, 2])
        gamma = math.atan2(-R[0, 1], R[0, 0])
    else:
        alpha = 0.0
        gamma = math.atan2(R[1, 0], R[1, 1])
    return np.array([alpha, beta, gamma])

# ---------- UR5 Forward Kinematics ----------
def ur5_fk(joint_rad: np.ndarray):
    """UR5 forward kinematics -> (pos, euler_xyz, T_0E)."""
    j1, j2, j3, j4, j5, j6 = joint_rad

    T_01 = dh_transform(0.0,   0.0,     0.0892,  j1 - PI/2)
    T_12 = dh_transform(PI/2,  0.0,     0.0,     j2 + PI/2)
    T_23 = dh_transform(0.0,   0.4251,  0.0,     j3)
    T_34 = dh_transform(0.0,   0.39215, 0.11,    j4 + PI/2)
    T_45 = dh_transform(PI/2,  0.0,     0.09475, j5)
    T_56 = dh_transform(-PI/2, 0.0,     0.26658, j6)

    T_6E = np.eye(4)
    T_0E = T_01 @ T_12 @ T_23 @ T_34 @ T_45 @ T_56 @ T_6E

    pos = T_0E[:3, 3]
    R   = T_0E[:3, :3]
    eul = euler_from_rotmat_xyz(R)
    return pos, eul, T_0E

# ---------- CoppeliaSim callbacks ----------
def sysCall_init():
    global sim
    sim = require("sim")

def sysCall_thread():
    joint_paths = [
        "/UR5/joint",
        "/UR5/joint/link/joint",
        "/UR5/joint/link/joint/link/joint",
        "/UR5/joint/link/joint/link/joint/link/joint",
        "/UR5/joint/link/joint/link/joint/link/joint/link/joint",
        "/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint",
    ]
    joint_handles = [sim.getObject(p) for p in joint_paths]
    ee_handle = sim.getObject("/UR5/EndPoint")

    print("\n=== UR5 Forward Kinematics and Orientation Display ===")
    print("Running simulation... Press Stop to end.\n")

    start_time = time.time()
    t = 0.0

    while t < 10.0:
        base = 45.0 * DEG2RAD * np.sin(0.01 * PI * t)
        targets = np.array([base, 0.8*base, 0.6*base, 0.4*base, 0.3*base, 0.2*base])

        # Command joints
        for h, q in zip(joint_handles, targets):
            sim.setJointTargetPosition(h, float(q))


        # Read actual positions
        actual = np.array([sim.getJointPosition(h) for h in joint_handles], dtype=float)

        # Forward Kinematics
        fk_pos, fk_eul, _ = ur5_fk(actual)

        # Actual from CoppeliaSim
        sim_pos = sim.getObjectPosition(ee_handle, -1)
        sim_eul = sim.getObjectOrientation(ee_handle, -1)

        # Degrees
        fk_eul_deg  = fk_eul * RAD2DEG
        sim_eul_deg = np.array(sim_eul) * RAD2DEG
        actual_deg  = actual * RAD2DEG

        # ---------- Pretty print ----------
        print("\n" + "=" * 65)
        print(f"Time: {t:6.2f} s")
        print("-" * 65)
        print("Joint Angles (deg):")
        print(" ".join([f"J{i+1}:{ang:7.2f}" for i, ang in enumerate(actual_deg)]))
        print("-" * 65)
        print("CALCULATED Forward Kinematics")
        print(f"  Position [m]:  X={fk_pos[0]:.4f}  Y={fk_pos[1]:.4f}  Z={fk_pos[2]:.4f}")
        print(f"  Orientation : Roll={fk_eul_deg[0]:7.2f}  Pitch={fk_eul_deg[1]:7.2f}  Yaw={fk_eul_deg[2]:7.2f}")
        print("-" * 65)
        print("COPPELIASIM Actual Values")
        print(f"  Position [m]:  X={sim_pos[0]:.4f}  Y={sim_pos[1]:.4f}  Z={sim_pos[2]:.4f}")
        print(f"  Orientation : Roll={sim_eul_deg[0]:7.2f}  Pitch={sim_eul_deg[1]:7.2f}  Yaw={sim_eul_deg[2]:7.2f}")
        print("=" * 65)

        time.sleep(0.1)
        sim.switchThread()
        t = time.time() - start_time

def sysCall_actuation():
    pass

def sysCall_sensing():
    pass

def sysCall_cleanup():
    pass

#luaExec wrapper='pythonWrapper' -- using the old wrapper for backw. compat.
# (As requested, we also call sim=require('sim') in sysCall_init.)
import time
import math
import numpy as np

pi  = np.pi
d2r = pi/180.0
r2d = 1.0/d2r

def sysCall_init():

    sim = require('sim')
    globals()['sim'] = sim  # ensure visible to thread

# ---------- helpers ----------
def eulerXYZ_from_R(R):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-9:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0.0
    return np.array([x, y, z], dtype=float)

def T_of_DH(a, alpha, d, theta):
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([
        [ ct, -st*ca,  st*sa, a*ct],
        [ st,  ct*ca, -ct*sa, a*st],
        [  0,     sa,     ca,    d],
        [  0,      0,      0,    1],
    ], dtype=float)

def sysCall_thread():
    # ----- original handle definitions (kept) -----
    hdl_j={}
    hdl_j[0] = sim.getObject("/UR5/joint")
    hdl_j[1] = sim.getObject("/UR5/joint/link/joint")
    hdl_j[2] = sim.getObject("/UR5/joint/link/joint/link/joint")
    hdl_j[3] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint")
    hdl_j[4] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint")
    hdl_j[5] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint")
    hdl_end = sim.getObject("/UR5/EndPoint")

    # ----- DH (classic; a_{i-1}, alpha_{i-1}, d_i, theta_i) from your world-frame list -----
    a     = np.array([0.00000,  0.42510, 0.39215, 0.00000, 0.00000, 0.0000], dtype=float)
    alpha = np.array([math.pi/2, 0.00000, 0.00000, math.pi/2, -math.pi/2, 0.0000], dtype=float)
    d     = np.array([0.0892,   0.00000,  0.00000, 0.10915, 0.094, 0.0823], dtype=float)
    thOfs = np.array([math.pi/2, -math.pi/2, 0.00000, math.pi/2, 0.00000,0.00], dtype=float)
    # actual ?_i used in FK each step: ? = q_meas + thOfs

    # ----- original time + print state -----
    t  = 0.0
    t0 = time.time()
    th_deg = {}

    while t < 10.0:
        # original target (?45? at 0.1 Hz)
        p = 45.0*d2r*np.sin(0.2*pi*t)

        # drive all 6 joints (use range(6); original had 0..5 which skipped joint 6)
        for i in range(6):
            sim.setJointTargetPosition(hdl_j[i], p)

        # readback joint positions (deg) for print
        for i in range(6):
            th_deg[i] = round(sim.getJointPosition(hdl_j[i]) * r2d, 2)

        # built-in pose
        end_pos = np.array(sim.getObjectPosition(hdl_end, -1), dtype=float)
        end_ori = np.array(sim.getObjectOrientation(hdl_end, -1), dtype=float)  # XYZ radians

        # ---------- FK via DH (compare) ----------
        q_meas = np.array([sim.getJointPosition(hdl_j[i]) for i in range(6)], dtype=float)
        thetas = q_meas + thOfs

        T = np.eye(4)
        for i in range(6):
            T = T @ T_of_DH(a[i], alpha[i], d[i], thetas[i])

        p_dh   = T[:3, 3]
        eul_dh = eulerXYZ_from_R(T[:3, :3])

        # ---------- prints (kept original prints + added DH compare) ----------
        print("-----------------------")
        print("Joint Position [deg]: {}".format(th_deg))
        print("SCN End position [m]: {}".format(np.array(end_pos).round(4)))
        print("SCN End eul XYZ [deg]: {}".format(np.array(end_ori*r2d).round(2)))

        print("DH  End position [m]: {}".format(np.array(p_dh).round(4)))
        print("DH  End eul XYZ [deg]: {}".format(np.array(eul_dh*r2d).round(2)))

        print("?pos [m]: {}".format(np.array(p_dh - end_pos).round(6)))
        print("?eul [deg]: {}".format(np.array((eul_dh - end_ori)*r2d).round(3)))

        # time update + yield
        t = time.time() - t0
        sim.switchThread()  # resume in next simulation step

    pass

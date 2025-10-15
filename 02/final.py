import time
import numpy as np

pi = np.pi
deg_to_rad = pi / 180
rad_to_deg = 1 / deg_to_rad


def create_dh_parameters(joint_angles):
    parameters = np.array(
        [
            [0, 0, 0.0892, -90],
            [90, 0, 0, 90],
            [0, 0.4251, 0, 0],
            [0, 0.39215, 0.11, -90],
            [-90, 0, 0.09473, 0],
            [90, 0, 0, 0],
        ]
    )
    for idx, row in enumerate(parameters):
        row[3] += joint_angles[idx]
    return parameters


def rotation_x(angle):
    angle_rad = np.radians(angle)
    matrix = np.eye(4)
    matrix[1, :] = [0, np.cos(angle_rad), -np.sin(angle_rad), 0]
    matrix[2, :] = [0, np.sin(angle_rad), np.cos(angle_rad), 0]
    return matrix


def rotation_y(angle):
    angle_rad = np.radians(angle)
    matrix = np.eye(4)
    matrix[0, :] = [np.cos(angle_rad), 0, np.sin(angle_rad), 0]
    matrix[2, :] = [-np.sin(angle_rad), 0, np.cos(angle_rad), 0]
    return matrix


def rotation_z(angle):
    angle_rad = np.radians(angle)
    matrix = np.eye(4)
    matrix[0, :] = [np.cos(angle_rad), -np.sin(angle_rad), 0, 0]
    matrix[1, :] = [np.sin(angle_rad), np.cos(angle_rad), 0, 0]
    return matrix


def compute_transformation(parameters, start_idx, end_idx):
    result = np.eye(4)
    temp = np.eye(4)
    for i in range(start_idx, end_idx):
        link_twist = np.radians(parameters[i][0])
        joint_angle = np.radians(parameters[i][3])
        temp[0, :] = [
            np.cos(joint_angle),
            -np.sin(joint_angle),
            0,
            parameters[i][1],
        ]
        temp[1, :] = [
            np.sin(joint_angle) * np.cos(link_twist),
            np.cos(joint_angle) * np.cos(link_twist),
            -np.sin(link_twist),
            -np.sin(link_twist) * parameters[i][2],
        ]
        temp[2, :] = [
            np.sin(joint_angle) * np.sin(link_twist),
            np.cos(joint_angle) * np.sin(link_twist),
            np.cos(link_twist),
            np.cos(link_twist) * parameters[i][2],
        ]
        temp[3, :] = [
            0,
            0,
            0,
            1,
        ]
        result = result @ temp
    return result


def extract_pose(transform_matrix):
    position = transform_matrix[:3, 3]
    rotation = transform_matrix[:3, :3]
    orientation = np.array(
        [
            np.arctan2(-transform_matrix[1, 2], transform_matrix[2, 2]),
            np.arcsin(transform_matrix[0, 2]),
            np.arctan2(-transform_matrix[0, 1], transform_matrix[0, 0]),
        ]
    )
    return position, rotation, orientation


def calculate_jacobian(parameters, end_transform, joint_types):

    columns = []
    for idx in range(len(parameters)):
        transform = compute_transformation(parameters, idx, len(parameters)) @ end_transform
        axis = extract_pose(transform)[1] @ np.array([0, 0, 1])
        displacement = extract_pose(transform)[0]
        linear = ((1 - joint_types[idx]) * np.cross(axis, displacement)) + (
            joint_types[idx] * axis
        )
        angular = (1 - joint_types[idx]) * axis
        columns.append(linear + angular)
    return np.array(columns).T


def solve_inverse_kinematics(current_pose, target_pose, current_angles, jacobian_matrix, step_size=0.05):
    pose_error = current_pose - target_pose
    jacobian_pseudo_inv = np.linalg.pinv(jacobian_matrix)
    angle_delta = jacobian_pseudo_inv @ pose_error[:3]
    updated_angles = current_angles + (step_size * angle_delta)
    return updated_angles


def sysCall_init():
    sim = require("sim")


def sysCall_thread():

    joint_handles = {}
    joint_handles[0] = sim.getObject("/UR5/joint")
    joint_handles[1] = sim.getObject("/UR5/joint/link/joint")
    joint_handles[2] = sim.getObject("/UR5/joint/link/joint/link/joint")
    joint_handles[3] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint")
    joint_handles[4] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint")
    joint_handles[5] = sim.getObject(
        "/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint"
    )
    endpoint_handle = sim.getObject("/UR5/EndPoint")

    target = sim.getObject("/Cuboid")

    elapsed_time = 0
    start_time = time.time()
    angles = {}


    end_effector_offset = np.eye(4)
    end_effector_offset[2, 3] = 0.431
    end_effector_offset = end_effector_offset
    forward_kinematics = np.eye(4)
    offset_matrix = end_effector_offset
    jacobian_mat = np.array([])
    joint_type_array = np.zeros(6)
    updated_angles = np.zeros(6)

    while True:

        for i in range(0, 6):
            angles[i] = round(sim.getJointPosition(joint_handles[i]) * rad_to_deg, 2)
            updated_angles[i] = np.deg2rad(angles[i])


        dh_params = create_dh_parameters(angles)
        forward_kinematics = compute_transformation(dh_params, 0, 6) @ offset_matrix
        position, rotation, orientation = extract_pose(forward_kinematics)
        jacobian_mat = calculate_jacobian(dh_params, offset_matrix, joint_type_array)

        endpoint_position = sim.getObjectPosition(endpoint_handle, -1)
        target_position = sim.getObjectPosition(target, -1)

        endpoint_orientation = sim.getObjectOrientation(endpoint_handle, -1)
        target_orientation = sim.getObjectOrientation(target, -1)

        current_pose = np.hstack((position, orientation))
        target_pose_array = np.hstack((target_position, target_orientation))
        updated_angles = solve_inverse_kinematics(current_pose, target_pose_array, updated_angles, jacobian_mat)
        for i in range(0, 6):
            sim.setJointTargetPosition(joint_handles[i], updated_angles[i])



        elapsed_time = time.time() - start_time

        sim.switchThread()  
    pass
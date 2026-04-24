"""
PD Control for Forward Kinematics - UR10e Robot
================================================

This script demonstrates PD (Proportional-Derivative) torque control for direct joint position control.
The robot moves to a desired joint configuration using PD control.
"""

import mujoco as mj
import mujoco.viewer
import numpy as np
import os
from generic_ik_solver import IKSolver

# Path to XML model
current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "ur10e", "ur10e_custom_gripper_scene.xml")

# Control parameters
TIME_STEP = 0.002
KP_JOINTS = np.array([300, 250, 90, 100, 70, 100])
KD_JOINTS = np.array([5.0, 40.0, 35.0, 40.0, 5.0, 3.5])

# Initial and target joint positions (6 DOF for arm)
initial_qpos = np.array([-1.5708, -2.0708, 1.2708, -2.0, 1.5708, 0.0])

# End-effector frame used for IK
IK_SITE_NAME = "gripperframe"

# Cup geometry (from XML)
CUP_BODY_NAME = "solo_cup"
CUP_CENTER_Z_OFFSET = 0.035
CUP_RADIUS = 0.035

# Side approach tuning (meters)
SIDE_PREGRASP_GAP = 0.055
SIDE_GRASP_DEPTH = 0.004
LIFT_HEIGHT = 0.16

# Gripper positions (radians)
GRIPPER_OPEN = 1.5   # Open position (minimum)
GRIPPER_CLOSED = 0.6      # Closed position - more squeeze


def pd_control(desired_qpos, current_qpos, current_qvel, kp, kd):
    """
    Compute PD control torques.
    
    Args:
        desired_qpos: Desired joint positions
        current_qpos: Current joint positions
        current_qvel: Current joint velocities
        kp: Proportional gain
        kd: Derivative gain
    
    Returns:
        Control torques
    """
    pos_error = desired_qpos - current_qpos
    vel_error = -current_qvel
    return kp * pos_error + kd * vel_error


def interpolate_qpos(initial, target, alpha):
    """Smooth interpolation between initial and target joint positions."""
    return initial + alpha * (target - initial)


def normalize(vec):
    """Return a unit vector, preserving zeros."""
    nrm = np.linalg.norm(vec)
    if nrm < 1e-8:
        return vec.copy()
    return vec / nrm


def compute_side_grasp_pose(model, data):
    """
    Compute a side-approach grasp pose.

    We project the initial cup-to-end-effector vector to XY and use it as the
    horizontal approach direction so the gripper comes from the side (not top-down).
    """
    cup_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, CUP_BODY_NAME)
    ee_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, IK_SITE_NAME)

    cup_body_pos = data.xpos[cup_id].copy()
    ee_pos = data.site_xpos[ee_site_id].copy()
    cup_center = cup_body_pos + np.array([0.0, 0.0, CUP_CENTER_Z_OFFSET])

    side_vec_xy = ee_pos[:2] - cup_center[:2]
    side_dist = np.linalg.norm(side_vec_xy)
    if side_dist < 1e-6:
        side_xy = np.array([1.0, 0.0])
    else:
        side_xy = side_vec_xy / side_dist

    approach_axis = np.array([side_xy[0], side_xy[1], 0.0])
    world_up = np.array([0.0, 0.0, 1.0])

    z_axis = normalize(approach_axis)
    x_axis = normalize(np.cross(world_up, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))

    target_rot = np.column_stack([x_axis, y_axis, z_axis])
    base_yaw = np.arctan2(approach_axis[1], approach_axis[0])
    base_dist = np.linalg.norm(cup_center[:2])

    return cup_center, target_rot, z_axis, base_dist, base_yaw


def compute_planar_seed(cup_center, radial_dist, base_yaw):
    """Compute a geometric seed for the first 3 joints from base-to-cup distance."""
    shoulder_height = 0.181
    shoulder_radius_offset = 0.176
    link_1 = 0.613
    link_2 = 0.571

    z = cup_center[2] - shoulder_height
    radial_eff = max(0.05, radial_dist - shoulder_radius_offset)
    reach = np.sqrt(radial_eff**2 + z**2)
    reach = np.clip(reach, 0.05, link_1 + link_2 - 1e-4)

    cos_elbow = np.clip((reach**2 - link_1**2 - link_2**2) / (2.0 * link_1 * link_2), -1.0, 1.0)
    elbow_inner = np.arccos(cos_elbow)

    shoulder_line = np.arctan2(z, radial_eff)
    shoulder_offset = np.arctan2(link_2 * np.sin(elbow_inner), link_1 + link_2 * np.cos(elbow_inner))

    q1 = base_yaw
    q2 = shoulder_line - shoulder_offset - np.pi / 2.0
    q3 = np.pi / 2.0 - elbow_inner

    return np.array([q1, q2, q3, -1.7, 1.57, 0.0])


def solve_pose_ik(model, data, target_pos, target_rot, initial_guess):
    """Solve orientation-constrained IK for a site pose."""
    data.qpos[:6] = initial_guess
    mj.mj_forward(model, data)

    solver = IKSolver(model, data, n_dof=6, verbose=False)
    result = solver.solve_ik(
        target_pos=target_pos,
        target_rot_matrix=target_rot,
        site_name=IK_SITE_NAME,
        rot_weight=0.3,
        max_steps=8000,
        tol=5e-5,
        regularization_strength=2e-2,
        regularization_threshold=0.08,
        progress_thresh=50.0,
        max_update_norm=0.25,
        progress_check_delay=20,
    )
    return result["qpos"][:6], result["err_norm"], result["success"]


def main():
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    model.opt.timestep = TIME_STEP

    # Dedicated IK state avoids disturbing the live simulation state.
    model_ik = mj.MjModel.from_xml_path(xml_path)
    data_ik = mj.MjData(model_ik)

    gripper_actuator_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "gripper")

    data.qpos[:6] = initial_qpos
    mj.mj_forward(model, data)
    data_ik.qpos[:6] = initial_qpos
    mj.mj_forward(model_ik, data_ik)
    
    cup_center, target_rot, approach_axis, radial_dist, base_yaw = compute_side_grasp_pose(model_ik, data_ik)
    seed_qpos = compute_planar_seed(cup_center, radial_dist, base_yaw)

    pregrasp_offset = CUP_RADIUS + SIDE_PREGRASP_GAP
    grasp_depth = SIDE_GRASP_DEPTH
    pregrasp_pos = cup_center - approach_axis * pregrasp_offset
    grasp_pos = cup_center - approach_axis * grasp_depth
    lift_pos = grasp_pos + np.array([0.0, 0.0, LIFT_HEIGHT])

    print("UR10e Cup Pickup Demo")
    print(f"Cup center: {cup_center}")
    print("Using generic_ik_solver.IKSolver for side approach")
    print(f"Cup radial distance: {radial_dist:.4f} m")
    print(f"Side approach yaw (joint 1 seed): {base_yaw:.4f} rad ({np.degrees(base_yaw):.2f} deg)")
    print(f"Approach axis (XY side-on): {approach_axis}")

    pregrasp_qpos, pre_err, pre_ok = solve_pose_ik(model_ik, data_ik, pregrasp_pos, target_rot, seed_qpos)
    grasp_qpos, grasp_err, grasp_ok = solve_pose_ik(model_ik, data_ik, grasp_pos, target_rot, pregrasp_qpos)
    lift_qpos, lift_err, lift_ok = solve_pose_ik(model_ik, data_ik, lift_pos, target_rot, grasp_qpos)

    print(f"Pre-grasp IK success={pre_ok}, error={pre_err:.6f}")
    print(f"Grasp IK success={grasp_ok}, error={grasp_err:.6f}")
    print(f"Lift IK success={lift_ok}, error={lift_err:.6f}")

    print("Joint targets (radians):")
    print(f"  pre-grasp: {pregrasp_qpos}")
    print(f"  grasp:     {grasp_qpos}")
    print(f"  lift:      {lift_qpos}")

    print("Joint targets (degrees):")
    print(f"  pre-grasp: {np.degrees(pregrasp_qpos)}")
    print(f"  grasp:     {np.degrees(grasp_qpos)}")
    print(f"  lift:      {np.degrees(lift_qpos)}")

    site_id = mj.mj_name2id(model_ik, mj.mjtObj.mjOBJ_SITE, IK_SITE_NAME)
    for label, q in [("pre-grasp", pregrasp_qpos), ("grasp", grasp_qpos), ("lift", lift_qpos)]:
        data_ik.qpos[:6] = q
        mj.mj_forward(model_ik, data_ik)
        achieved = data_ik.site_xpos[site_id]
        target = pregrasp_pos if label == "pre-grasp" else grasp_pos if label == "grasp" else lift_pos
        print(f"{label} position error: {np.linalg.norm(achieved - target):.4f} m")

    APPROACH_PREGRASP = 4.0
    APPROACH_GRASP = 2.0
    GRASP_PHASE = 1.5
    LIFT_PHASE = 3.0
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            phase_time = data.time
            
            # Phase 1: Move to pre-grasp point
            if phase_time < APPROACH_PREGRASP:
                alpha = phase_time / APPROACH_PREGRASP
                smooth_alpha = 3 * alpha**2 - 2 * alpha**3
                desired_qpos = interpolate_qpos(initial_qpos, pregrasp_qpos, smooth_alpha)
                gripper_goal = GRIPPER_OPEN

            # Phase 2: Move straight into cup centerline
            elif phase_time < APPROACH_PREGRASP + APPROACH_GRASP:
                alpha = (phase_time - APPROACH_PREGRASP) / APPROACH_GRASP
                smooth_alpha = 3 * alpha**2 - 2 * alpha**3
                desired_qpos = interpolate_qpos(pregrasp_qpos, grasp_qpos, smooth_alpha)
                gripper_goal = GRIPPER_OPEN
            
            # Phase 3: Close gripper on cup
            elif phase_time < APPROACH_PREGRASP + APPROACH_GRASP + GRASP_PHASE:
                desired_qpos = grasp_qpos
                alpha = (phase_time - APPROACH_PREGRASP - APPROACH_GRASP) / GRASP_PHASE
                smooth_alpha = 3 * alpha**2 - 2 * alpha**3
                gripper_goal = interpolate_qpos(GRIPPER_OPEN, GRIPPER_CLOSED, smooth_alpha)
            
            # Phase 4: Lift cup while keeping grasp
            else:
                alpha = min(1.0, (phase_time - APPROACH_PREGRASP - APPROACH_GRASP - GRASP_PHASE) / LIFT_PHASE)
                smooth_alpha = 3 * alpha**2 - 2 * alpha**3
                desired_qpos = interpolate_qpos(grasp_qpos, lift_qpos, smooth_alpha)
                gripper_goal = GRIPPER_CLOSED
            
            # Compute PD control torques for arm
            current_qpos = data.qpos[:6]
            current_qvel = data.qvel[:6]
            torques = pd_control(desired_qpos, current_qpos, current_qvel, KP_JOINTS, KD_JOINTS)

            data.ctrl[:6] = torques
            
            if gripper_actuator_id >= 0:
                data.ctrl[gripper_actuator_id] = gripper_goal
            
            # Step simulation
            mj.mj_step(model, data)
            viewer.sync()
    
    print("Pickup completed")


if __name__ == "__main__":
    main()


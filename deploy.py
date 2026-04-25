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
KP_JOINTS = np.array([300, 250, 90, 100, 70, 100])   # Proportional gains per joint (reduced for smoother motion)
KD_JOINTS = np.array([5.0, 40.0, 35.0, 40.0, 5.0, 3.5])  
KP_GRIPPER = 15.0  # Minimal force for contact only
KD_GRIPPER = 2.0

# Initial and target joint positions (6 DOF for arm)
initial_qpos = np.array([-1.5708, -2.0708, 1.2708, -2.0, 1.5708, 0.0])

# Cup starting positions in workspace (from XML)
CUP_POSITIONS = [
    np.array([1.00, 0.0, 0.11]),    # Cup 1
    np.array([1.00, 0.15, 0.11]),   # Cup 2
    np.array([1.00, 0.3, 0.11]),    # Cup 3
    np.array([1.00, 0.45, 0.11]),   # Cup 4
    np.array([1.00, 0.6, 0.11]),    # Cup 5
    np.array([1.15, 0.0, 0.11]),    # Cup 6
    np.array([1.15, 0.15, 0.11]),   # Cup 7
    np.array([1.15, 0.3, 0.11]),    # Cup 8
    np.array([1.15, 0.45, 0.11]),   # Cup 9
    np.array([1.15, 0.6, 0.11]),    # Cup 10
]

# Gripper finger offset - adjust to position cup between fingers
# The gripper's moving jaw is offset, so we need to compensate to center the cup
GRIPPER_FINGER_OFFSET = np.array([0.13, -0.2, 0])  # Y-offset to center cup between fingers

# Gripper positions (radians)
GRIPPER_OPEN = 1.5   # Open position (minimum)
GRIPPER_CLOSED = 0.9    # Closed position - barely closes, just touches cup


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


def calculate_pyramid_positions(base_pos, spacing=0.065, z_offset=0.07):
    """
    Calculate 10 positions arranged in a pyramid formation (4-3-2-1 stack).
    
    Args:
        base_pos: np.array([x, y, z]) - Base position for the pyramid
        spacing: float - Distance between cup centers (default 0.065m for small cups)
        z_offset: float - Height of each cup from ground (default 0.07m for block)
    
    Returns:
        list of np.array - 10 positions forming a pyramid
    """
    positions = []
    
    # Pyramid structure: 4-3-2-1 (bottom to top)
    # Each row is centered relative to the previous one
    
    row_sizes = [4, 3, 2, 1]  # Cups per row
    x_offset = base_pos[0]
    y_offset = base_pos[1]
    z_base = base_pos[2]
    
    z_current = z_base
    
    for row_idx, row_size in enumerate(row_sizes):
        # Center the row
        row_width = (row_size - 1) * spacing
        row_start_x = x_offset - row_width / 2
        
        for col_idx in range(row_size):
            x = row_start_x + col_idx * spacing
            y = y_offset
            z = z_current
            positions.append(np.array([x, y, z]))
        
        # Move up for next row (stack height based on block height ~0.07m)
        z_current += 0.075
    
    return positions


def main():
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    model.opt.timestep = TIME_STEP

    gripper_actuator_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "gripper")
    gripper_joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "gripper")

    data.qpos[:6] = initial_qpos
    mj.mj_forward(model, data)
    
    # Initialize IK solver (using 6 DOF for arm only)
    solver = IKSolver(model, data, n_dof=6, verbose=True)
    
    print("UR10e Cup Pickup Demo")
    print(f"Initial joint positions: {initial_qpos}")
    print(f"Number of cups to pick: {len(CUP_POSITIONS)}")
    
    # Define target orientation for grasping (gripper approaches cup vertically from above)
    # This rotation matrix makes the gripper's z-axis point downward for a vertical grip
    target_orient_pickup = np.array([
        [0.0,  0.0, 1.0],  # x-axis points forward
        [1.0,  0.0,  0.0],  # y-axis points left
        [0.0, 1.0,  0.0]   # z-axis points down
    ])
    
    # Pre-compute approach and pickup IK solutions for each cup
    print("\nComputing inverse kinematics for all cup positions...")
    side_approach_qpos_list = []
    target_qpos_list = []
    lift_qpos_list = []
    
    for cup_num, cup_pos in enumerate(CUP_POSITIONS):
        print(f"\n--- Cup {cup_num + 1} ---")
        cup_target = cup_pos + GRIPPER_FINGER_OFFSET
        print(f"Cup position: {cup_pos}, Gripper target: {cup_target}")
        
        # Side approach position (offset to the side of the cup)
        side_approach_offset = np.array([-0.15, 0.0, 0.0])
        side_approach_pos = cup_target + side_approach_offset
        
        # Compute IK for side approach
        data.qpos[:6] = initial_qpos
        result_side = solver.solve_ik(target_pos=side_approach_pos, target_rot_matrix=target_orient_pickup,
                                       body_name="gripper", rot_weight=0.2)
        side_approach_qpos_list.append(result_side["qpos"][:6].copy())
        if result_side["success"]:
            print(f"  Side approach: IK converged | Error: {result_side['err_norm']:.6e}")
        else:
            print(f"  Side approach: IK FAILED | Error: {result_side['err_norm']:.6e}")
        
        # Compute IK for pickup position
        data.qpos[:6] = initial_qpos
        result_pickup = solver.solve_ik(target_pos=cup_target, target_rot_matrix=target_orient_pickup,
                                        body_name="gripper", rot_weight=0.2)
        target_qpos_list.append(result_pickup["qpos"][:6].copy())
        if result_pickup["success"]:
            print(f"  Pickup: IK converged | Error: {result_pickup['err_norm']:.6e}")
        else:
            print(f"  Pickup: IK FAILED | Error: {result_pickup['err_norm']:.6e}")
        
        # Compute IK for lift position
        lift_target = cup_target + np.array([0, 0, 0.15])
        data.qpos[:6] = result_pickup["qpos"][:6]
        result_lift = solver.solve_ik(target_pos=lift_target, target_rot_matrix=target_orient_pickup,
                                      body_name="gripper", rot_weight=0.2)
        lift_qpos_list.append(result_lift["qpos"][:6].copy())
        if result_lift["success"]:
            print(f"  Lift: IK converged | Error: {result_lift['err_norm']:.6e}")
        else:
            print(f"  Lift: IK FAILED | Error: {result_lift['err_norm']:.6e}")
    
    # Generate pyramid positions for placing cups
    print("\nGenerating pyramid positions...")
    pyramid_base = np.array([0.5, -0.4, 0.17])  # Base of pyramid (z=0.02 for cups resting on floor)
    pyramid_positions = calculate_pyramid_positions(pyramid_base, spacing=0.055, z_offset=0.1)
    print(f"Pyramid positions: {len(pyramid_positions)} cups")
    
    # Compute IK solutions for all placement positions
    print("\nComputing IK solutions for placement positions...")
    placement_qpos_list = []
    placement_lower_qpos_list = []
    retreat_qpos_list = []
    
    for idx, pyramid_pos in enumerate(pyramid_positions):
        # Placement target with gripper offset applied
        placement_target = pyramid_pos + GRIPPER_FINGER_OFFSET
        
        # IK for placement position (at full height)
        data.qpos[:6] = lift_qpos_list[idx]  # Start from this cup's lifted position
        result_placement = solver.solve_ik(target_pos=placement_target, target_rot_matrix=target_orient_pickup,
                                           body_name="gripper", rot_weight=0.2)
        placement_qpos_list.append(result_placement["qpos"][:6].copy())
        
        # IK for lowered position (0.08m lower for contact with surface)
        lower_target = placement_target - np.array([0, 0, 0.08])
        data.qpos[:6] = lift_qpos_list[idx]
        result_lower = solver.solve_ik(target_pos=lower_target, target_rot_matrix=target_orient_pickup,
                                       body_name="gripper", rot_weight=0.2)
        placement_lower_qpos_list.append(result_lower["qpos"][:6].copy())
        
        # IK for retreat position (high above pyramid, away from placement area)
        retreat_target = placement_target + np.array([-0.2, 0, 0.2])  # Move back and up
        data.qpos[:6] = placement_lower_qpos_list[idx]
        result_retreat = solver.solve_ik(target_pos=retreat_target, target_rot_matrix=target_orient_pickup,
                                        body_name="gripper", rot_weight=0.2)
        retreat_qpos_list.append(result_retreat["qpos"][:6].copy())
        
        if result_placement["success"]:
            print(f"  Cup {idx+1}: Placement IK converged | Error: {result_placement['err_norm']:.6e}")
        else:
            print(f"  Cup {idx+1}: Placement IK FAILED | Error: {result_placement['err_norm']:.6e}")
    
    # Reset data for simulation
    data.qpos[:6] = initial_qpos
    mj.mj_forward(model, data)
    
    # Phase timing
    SIDE_APPROACH_PHASE = 2.8  # Move to side of cup (pre-grasp position)
    APPROACH_PHASE = 2.8      # Move forward toward cup to pick it up
    GRASP_PHASE = 2.0         # Close gripper
    LIFT_PHASE = 2.5          # Lift cup
    MOVE_TO_PLACE_PHASE = 3.0 # Move to placement location
    LOWER_PHASE = 2.0         # Lower cup to placement position
    RELEASE_PHASE = 1.5       # Open gripper to release
    RETREAT_PHASE = 2.0       # Retreat away from pyramid after releasing
    
    # Get gripper body ID (cup IDs will vary per iteration)
    gripper_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "gripper")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        cup_idx = 0  # Current cup being picked (1-10)
        start_time = 0.0  # Track time for each cup placement
        
        while viewer.is_running() and cup_idx < len(pyramid_positions):
            # Time since start of current cup placement
            phase_time = data.time - start_time
            
            # Define phase boundaries for this cup
            phase_1_end = SIDE_APPROACH_PHASE
            phase_2_end = phase_1_end + APPROACH_PHASE
            phase_3_end = phase_2_end + GRASP_PHASE
            phase_4_end = phase_3_end + LIFT_PHASE
            phase_5_end = phase_4_end + MOVE_TO_PLACE_PHASE
            phase_6_end = phase_5_end + LOWER_PHASE
            phase_7_end = phase_6_end + RELEASE_PHASE
            phase_8_end = phase_7_end + RETREAT_PHASE
            
            # Get positions for current cup
            placement_qpos = placement_qpos_list[cup_idx]
            placement_lower_qpos = placement_lower_qpos_list[cup_idx]
            retreat_qpos = retreat_qpos_list[cup_idx]
            side_approach_qpos = side_approach_qpos_list[cup_idx]
            target_qpos = target_qpos_list[cup_idx]
            lift_qpos = lift_qpos_list[cup_idx]
            
            # Phase 1: Move to side approach position
            if phase_time < phase_1_end:
                alpha = phase_time / SIDE_APPROACH_PHASE
                smooth_alpha = 3 * alpha**2 - 2 * alpha**3
                desired_qpos = interpolate_qpos(initial_qpos, side_approach_qpos, smooth_alpha)
                gripper_goal = GRIPPER_OPEN
            
            # Phase 2: Move from side approach toward cup (approach)
            elif phase_time < phase_2_end:
                alpha = (phase_time - phase_1_end) / APPROACH_PHASE
                smooth_alpha = 3 * alpha**2 - 2 * alpha**3
                desired_qpos = interpolate_qpos(side_approach_qpos, target_qpos, smooth_alpha)
                gripper_goal = GRIPPER_OPEN
            
            # Phase 3: Close gripper
            elif phase_time < phase_3_end:
                desired_qpos = target_qpos
                alpha = (phase_time - phase_2_end) / GRASP_PHASE
                smooth_alpha = 3 * alpha**2 - 2 * alpha**3
                gripper_goal = interpolate_qpos(GRIPPER_OPEN, GRIPPER_CLOSED, smooth_alpha)
            
            # Phase 4: Lift cup (gripper holds via friction)
            elif phase_time < phase_4_end:
                alpha = (phase_time - phase_3_end) / LIFT_PHASE
                smooth_alpha = 3 * alpha**2 - 2 * alpha**3
                desired_qpos = interpolate_qpos(target_qpos, lift_qpos, smooth_alpha)
                gripper_goal = GRIPPER_CLOSED
            
            # Phase 5: Move to placement location
            elif phase_time < phase_5_end:
                alpha = (phase_time - phase_4_end) / MOVE_TO_PLACE_PHASE
                smooth_alpha = 3 * alpha**2 - 2 * alpha**3
                desired_qpos = interpolate_qpos(lift_qpos, placement_qpos, smooth_alpha)
                gripper_goal = GRIPPER_CLOSED
            
            # Phase 6: Lower cup to placement surface
            elif phase_time < phase_6_end:
                alpha = (phase_time - phase_5_end) / LOWER_PHASE
                smooth_alpha = 3 * alpha**2 - 2 * alpha**3
                desired_qpos = interpolate_qpos(placement_qpos, placement_lower_qpos, smooth_alpha)
                gripper_goal = GRIPPER_CLOSED
            
            # Phase 7: Release gripper
            elif phase_time < phase_7_end:
                desired_qpos = placement_lower_qpos
                alpha = (phase_time - phase_6_end) / RELEASE_PHASE
                smooth_alpha = 3 * alpha**2 - 2 * alpha**3
                gripper_goal = interpolate_qpos(GRIPPER_CLOSED, GRIPPER_OPEN, smooth_alpha)
            
            # Phase 8: Retreat away from pyramid (up and back)
            elif phase_time < phase_8_end:
                alpha = (phase_time - phase_7_end) / RETREAT_PHASE
                smooth_alpha = 3 * alpha**2 - 2 * alpha**3
                desired_qpos = interpolate_qpos(placement_lower_qpos, retreat_qpos, smooth_alpha)
                gripper_goal = GRIPPER_OPEN
            
            # Move to next cup
            else:
                print(f"✓ Placed cup {cup_idx + 1}/10")
                cup_idx += 1
                start_time = data.time
                continue
            
            # Compute PD control torques for arm
            current_qpos = data.qpos[:6]
            current_qvel = data.qvel[:6]
            torques = pd_control(desired_qpos, current_qpos, current_qvel, KP_JOINTS, KD_JOINTS)

            data.ctrl[:6] = torques
            
            # Gripper PD control
            if gripper_actuator_id >= 0 and gripper_joint_id >= 0:
                qpos_adr = model.jnt_qposadr[gripper_joint_id]
                qvel_adr = model.jnt_dofadr[gripper_joint_id]
                current_gripper_pos = data.qpos[qpos_adr]
                current_gripper_vel = data.qvel[qvel_adr] if qvel_adr >= 0 else 0.0
                gripper_control = pd_control(np.array([gripper_goal]), np.array([current_gripper_pos]), np.array([current_gripper_vel]), KP_GRIPPER, KD_GRIPPER)
                data.ctrl[gripper_actuator_id] = gripper_control[0]
            
            # Step simulation
            mj.mj_step(model, data)
            viewer.sync()
    
    print(f"✓ Successfully placed all 10 cups in pyramid formation!")


if __name__ == "__main__":
    main()


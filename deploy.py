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
from scipy.optimize import minimize

# Path to XML model
current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "ur10e", "ur10e_custom_gripper_scene.xml")

# Control parameters
TIME_STEP = 0.002
KP_JOINTS = np.array([300, 250, 90, 100, 70, 100])   # Proportional gains per joint (reduced for smoother motion)
KD_JOINTS = np.array([5.0, 40.0, 35.0, 40.0, 5.0, 3.5])  
KP_GRIPPER = 17.8  # Proportional gain for gripper
KD_GRIPPER = 2.0   # Derivative gain for gripper

# Initial and target joint positions (6 DOF for arm)
initial_qpos = np.array([-1.5708, -2.0708, 1.2708, -2.0, 1.5708, 0.0])

# Cup position (where we want to pick it up)
CUP_POS = np.array([1.35, 0.05, 0.11])  # Center of cup wall at mid-height

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


def solve_ik(model, target_pos, initial_guess, prefer_upright=True):
    """
    Solve inverse kinematics to reach target position.
    
    Args:
        model: MuJoCo model
        target_pos: Target 3D position for end-effector
        initial_guess: Initial joint configuration
        prefer_upright: If True, penalizes configurations where arm hangs down
    
    Returns:
        Joint configuration that reaches target position
    """
    def objective(qpos):
        # Create a temporary data object for FK evaluation
        data_temp = mj.MjData(model)
        data_temp.qpos[:6] = qpos
        mj.mj_forward(model, data_temp)
        
        # Get end-effector position (gripper body)
        gripper_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "gripper")
        
        if gripper_body_id >= 0:
            ee_pos = data_temp.body(gripper_body_id).xpos
        else:
            # Fallback to wrist_3_link
            gripper_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "wrist_3_link")
            ee_pos = data_temp.body(gripper_body_id).xpos if gripper_body_id >= 0 else target_pos.copy()
        
        # Distance error
        pos_error = np.linalg.norm(ee_pos - target_pos)
        
        # Joint regularization (prefer initial position)
        joint_reg = 0.01 * np.linalg.norm(qpos - initial_guess)
        
        # Penalize large negative joint angles that cause arm to hang (especially joint 1)
        upright_penalty = 0.0
        if prefer_upright:
            # Penalize if shoulder or elbow joints are too negative (arm hanging down)
            if qpos[1] < -np.pi:  # Shoulder joint too negative
                upright_penalty += 100 * (abs(qpos[1]) - np.pi)**2
            if qpos[2] < -np.pi/2:  # Elbow joint too negative
                upright_penalty += 50 * (abs(qpos[2]) - np.pi/2)**2
        
        # Penalize if gripper gets too close to ground (below 0.08m)
        ground_penalty = 0.0
        if ee_pos[2] < 0.08:
            ground_penalty = 500 * (0.08 - ee_pos[2])**2
        
        # Also check wrist_3_link height
        wrist_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "wrist_3_link")
        if wrist_id >= 0:
            wrist_pos = data_temp.body(wrist_id).xpos
            if wrist_pos[2] < 0.06:
                ground_penalty += 300 * (0.06 - wrist_pos[2])**2
        
        return pos_error + joint_reg + upright_penalty + ground_penalty
    
    # Solve IK with tighter tolerances
    result = minimize(objective, initial_guess, method='Nelder-Mead', 
                     options={'xatol': 1e-5, 'fatol': 1e-6, 'maxiter': 2000})
    
    return result.x[:6]


def main():
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    model.opt.timestep = TIME_STEP

    gripper_actuator_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "gripper")
    gripper_joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "gripper")

    data.qpos[:6] = initial_qpos
    mj.mj_forward(model, data)
    
    print("UR10e Cup Pickup Demo")
    print(f"Initial joint positions: {initial_qpos}")
    print(f"Cup position: {CUP_POS}")
    
    # Compute IK for cup position
    print("Computing inverse kinematics...")
    target_qpos = solve_ik(model, CUP_POS, initial_qpos)
    print(f"Target joint positions (IK solution): {target_qpos}")
    
    # Verify IK solution reaches cup
    verify_data = mj.MjData(model)
    verify_data.qpos[:6] = target_qpos
    mj.mj_forward(model, verify_data)
    gripper_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "gripper")
    if gripper_id >= 0:
        achieved_pos = verify_data.body(gripper_id).xpos
        distance = np.linalg.norm(achieved_pos - CUP_POS)
        print(f"Gripper position: {achieved_pos}")
        print(f"Cup position: {CUP_POS}")
        print(f"Distance to cup: {distance:.4f}m")
    
    # Phase timing
    APPROACH_PHASE = 5.0      # Move to cup (slow approach)
    GRASP_PHASE = 2.0         # Close gripper
    LIFT_PHASE = 3.5          # Lift cup (slow and steady)
    
    # Compute lift position (cup raised by 0.15m from pickup point)
    lift_pos = CUP_POS + np.array([0, 0, 0.15])
    lift_qpos = solve_ik(model, lift_pos, target_qpos, prefer_upright=True)
    
    # Verify lift position
    verify_data.qpos[:6] = lift_qpos
    mj.mj_forward(model, verify_data)
    if gripper_id >= 0:
        achieved_lift = verify_data.body(gripper_id).xpos
        lift_distance = np.linalg.norm(achieved_lift - lift_pos)
        print(f"Lift position achieved: {achieved_lift}")
        print(f"Lift position target: {lift_pos}")
        print(f"Distance to lift target: {lift_distance:.4f}m")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            phase_time = data.time
            
            # Phase 1: Move to above cup (approach)
            if phase_time < APPROACH_PHASE:
                alpha = phase_time / APPROACH_PHASE
                smooth_alpha = 3 * alpha**2 - 2 * alpha**3
                desired_qpos = interpolate_qpos(initial_qpos, target_qpos, smooth_alpha)
                gripper_goal = GRIPPER_OPEN
            
            # Phase 2: Close gripper
            elif phase_time < APPROACH_PHASE + GRASP_PHASE:
                desired_qpos = target_qpos
                alpha = (phase_time - APPROACH_PHASE) / GRASP_PHASE
                smooth_alpha = 3 * alpha**2 - 2 * alpha**3
                gripper_goal = interpolate_qpos(GRIPPER_OPEN, GRIPPER_CLOSED, smooth_alpha)
            
            # Phase 3: Lift cup
            else:
                alpha = min(1.0, (phase_time - APPROACH_PHASE - GRASP_PHASE) / LIFT_PHASE)
                smooth_alpha = 3 * alpha**2 - 2 * alpha**3
                desired_qpos = interpolate_qpos(target_qpos, lift_qpos, smooth_alpha)
                gripper_goal = GRIPPER_CLOSED
            
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
                data.ctrl[gripper_actuator_id] = gripper_goal + gripper_control[0] * TIME_STEP
            
            # Step simulation
            mj.mj_step(model, data)
            viewer.sync()
    
    print("Pickup completed")


if __name__ == "__main__":
    main()


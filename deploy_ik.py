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
KP_JOINTS = np.array([450, 370, 135, 150, 100, 144])   # Proportional gains per joint
KD_JOINTS = np.array([3.5, 30.36, 26.4375, 35, 3.9, 2.86])  
DURATION = 5.0  # Simulation duration

# End-effector site name
SITE_NAME = "attachment_site"

# Initial joint positions (6 DOF for arm)
initial_qpos = np.array([-0.5, -2.5708, 1.2708, -2.0, 1.5708, 0.0])

# Target end-effector pose
target_pos = np.array([-1.0296, 0.3723, 0.6976])
target_rot_matrix = np.array([
    [-0.221451, 0.972087, 0.077503],
    [0.027669, -0.073181, 0.996935],
    [0.974779, 0.222917, -0.010691]
])


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


def solve_ik_for_target(model, data, target_pos, target_rot_matrix, site_name, n_dof=None):
    """Solve IK for target pose and return desired joint positions."""
    solver = IKSolver(model, data, n_dof=n_dof, verbose=False)
    ik_result = solver.solve_ik(
        target_pos=target_pos,
        target_rot_matrix=target_rot_matrix,
        site_name=site_name,
        max_steps=100000,
        tol=1e-8,
        regularization_strength=3e-2
    )
    return ik_result['qpos'][:6], ik_result['err_norm'], ik_result['steps'], ik_result['success']


def main():
    # Load model for simulation
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    model.opt.timestep = TIME_STEP
    
    # Load dummy model for IK solving
    model_dummy = mj.MjModel.from_xml_path(xml_path)
    data_dummy = mj.MjData(model_dummy)
    
    # Set initial joint positions
    data.qpos[:6] = initial_qpos
    mj.mj_forward(model, data)
    
    # Solve IK for target pose
    print("UR10e IK Control Demo")
    print(f"Target end-effector position: {target_pos}")
    print("Solving IK...")
    
    data_dummy.qpos[:6] = initial_qpos
    mj.mj_forward(model_dummy, data_dummy)
    target_qpos, err_norm, steps, success = solve_ik_for_target(model_dummy, data_dummy, target_pos, target_rot_matrix, SITE_NAME)
    
    print(f"Target joint positions: {target_qpos}")
    print(f"Error norm: {err_norm}")
    print(f"Steps: {steps}")
    print(f"Success: {success}")
    
    # Get site ID for visualization
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, SITE_NAME)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Compute interpolation factor for smooth motion
            if data.time < DURATION:
                alpha = data.time / DURATION
                # Smooth interpolation (cubic easing)
                smooth_alpha = 3 * alpha**2 - 2 * alpha**3
                desired_qpos = initial_qpos + smooth_alpha * (target_qpos - initial_qpos)
            else:
                desired_qpos = target_qpos
            
            # Compute PD control torques
            current_qpos = data.qpos[:6]
            current_qvel = data.qvel[:6]
            torques = pd_control(desired_qpos, current_qpos, current_qvel, KP_JOINTS, KD_JOINTS)
            
            # Apply control torques (only to arm joints)
            data.ctrl[:6] = torques
            
            # Step simulation
            mj.mj_step(model, data)
            viewer.sync()
    
    # Print final end-effector pose
    mj.mj_forward(model, data)
    final_pos = data.site_xpos[site_id]
    final_rot = data.site_xmat[site_id].reshape(3, 3)
    print(f"\nFinal end-effector position: {final_pos}")
    print(f"\nFinal end-effector rotation: {final_rot}")
    print("Simulation completed")


if __name__ == "__main__":
    main()


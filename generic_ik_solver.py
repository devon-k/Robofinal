"""
Generic Inverse Kinematics Solver for MuJoCo
============================================

This module provides a generic inverse kinematics solver that can be used with any MuJoCo model.
It supports both position-only and position+orientation IK, with convergence tracking and visualization.

Usage:
    from generic_ik_solver import IKSolver
    
    # Create solver
    solver = IKSolver(model, data)
    
    # Solve IK
    result = solver.solve_ik(target_pos, target_rot_matrix, site_name="end_effector")
    
    # Plot convergence
    solver.plot_convergence(result["convergence_data"], "my_convergence.png")
"""

import mujoco as mj
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from typing import Optional, Dict, List, Tuple, Union


class IKSolver:
    """
    Generic Inverse Kinematics Solver for MuJoCo models.
    
    This class provides a flexible IK solver that can work with any MuJoCo model
    by specifying the target site or body for the end effector.
    """
    
    def __init__(self, model: mj.MjModel, data: mj.MjData, n_dof: Optional[int] = None, verbose: bool = True):
        """
        Initialize the IK solver with a MuJoCo model and data.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data (will be modified during IK solving)
            n_dof: Number of degrees of freedom to use for IK computation.
                   If None, uses all available DOFs (model.nv).
            verbose: Whether to print logging and progress information
        """
        self.model = model
        self.data = data
        self.dtype = data.qpos.dtype
        self.n_dof = n_dof if n_dof is not None else model.nv
        self.verbose = verbose
        
        # Validate n_dof
        if self.n_dof <= 0 or self.n_dof > model.nv:
            raise ValueError(f"n_dof must be between 1 and {model.nv}, got {self.n_dof}")
        
        if self.verbose:
            print(f"IKSolver initialized: using {self.n_dof} out of {model.nv} DOFs for IK")
        
    def _nullspace_method(self, jac_joints: np.ndarray, delta: np.ndarray, 
                         regularization_strength: float = 0.0) -> np.ndarray:
        """
        Solve the inverse kinematics using the nullspace method.
        
        Args:
            jac_joints: Jacobian matrix for the joints
            delta: Error vector
            regularization_strength: L2 regularization strength
            
        Returns:
            Joint update vector
        """
        hess_approx = jac_joints.T.dot(jac_joints)
        joint_delta = jac_joints.T.dot(delta)
        
        if regularization_strength > 0:
            # L2 regularization
            hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
            return np.linalg.solve(hess_approx, joint_delta)
        else:
            return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]
    
    def _get_site_id(self, site_name: str) -> int:
        """Get site ID by name with error handling."""
        try:
            site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, site_name)
            if self.verbose:
                print(f"Found site '{site_name}' with ID: {site_id}")
            return site_id
        except:
            if self.verbose:
                print(f"Warning: Site '{site_name}' not found. Using site 0.")
            return 0
    
    def _get_body_id(self, body_name: str) -> int:
        """Get body ID by name with error handling."""
        try:
            body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body_name)
            if self.verbose:
                print(f"Found body '{body_name}' with ID: {body_id}")
            return body_id
        except:
            if self.verbose:
                print(f"Warning: Body '{body_name}' not found. Using last body.")
            return self.model.nbody - 1
    
    def solve_ik(self, 
                 target_pos: Optional[np.ndarray] = None,
                 target_rot_matrix: Optional[np.ndarray] = None,
                 site_name: Optional[str] = None,
                 body_name: Optional[str] = None,
                 rot_weight: float = 1.0,
                 max_steps: int = 1000,
                 tol: float = 1e-8,
                 regularization_strength: float = 3e-2,
                 regularization_threshold: float = 0.1,
                 progress_thresh: float = 20.0,
                 max_update_norm: float = 2.0,
                 progress_check_delay: int = 0) -> Dict:
        """
        Solve inverse kinematics for the specified target.
        
        Args:
            target_pos: Target position [x, y, z] (optional)
            target_rot_matrix: Target rotation matrix 3x3 or flattened 9x1 (optional)
            site_name: Name of the target site (use either site_name or body_name)
            body_name: Name of the target body (use either site_name or body_name)
            rot_weight: Weight for rotation error in total error
            max_steps: Maximum number of IK iterations
            tol: Convergence tolerance
            regularization_strength: L2 regularization strength
            regularization_threshold: Error threshold above which to apply regularization
            progress_thresh: Progress threshold for early termination
            max_update_norm: Maximum norm for joint updates
            progress_check_delay: Number of steps to wait before checking progress/update norms
            
        Returns:
            Dictionary containing:
                - qpos: Final joint positions
                - success: Whether IK converged
                - steps: Number of iterations taken
                - err_norm: Final error norm
                - convergence_data: Data for plotting convergence
        """
        # Validate inputs
        if target_pos is None and target_rot_matrix is None:
            raise ValueError("At least one of target_pos or target_rot_matrix must be specified.")
        
        if site_name is None and body_name is None:
            raise ValueError("Either site_name or body_name must be specified.")
        
        if site_name is not None and body_name is not None:
            raise ValueError("Specify either site_name or body_name, not both.")
        
        # Setup target rotation
        target_rot = None
        if target_rot_matrix is not None:
            target_rot = np.empty(4, dtype=self.dtype)
            # Ensure target_rot_matrix is flattened for mju_mat2Quat
            if target_rot_matrix.shape == (3, 3):
                target_rot_matrix_flat = target_rot_matrix.flatten()
            elif target_rot_matrix.shape == (9,):
                target_rot_matrix_flat = target_rot_matrix
            else:
                raise ValueError(f"target_rot_matrix must be 3x3 or 9x1, got shape {target_rot_matrix.shape}")
            mj.mju_mat2Quat(target_rot, target_rot_matrix_flat)
        
        # Setup Jacobian and error arrays
        if target_pos is not None and target_rot is not None:
            jac = np.empty((6, self.model.nv), dtype=self.dtype)
            err = np.empty(6, dtype=self.dtype)
            jac_pos, jac_rot = jac[:3], jac[3:]
            err_pos, err_rot = err[:3], err[3:]
        else:
            jac = np.empty((3, self.model.nv), dtype=self.dtype)
            err = np.empty(3, dtype=self.dtype)
            if target_pos is not None:
                jac_pos, jac_rot = jac, None
                err_pos, err_rot = err, None
            elif target_rot is not None:
                jac_pos, jac_rot = None, jac
                err_pos, err_rot = None, err
        
        update_nv = np.zeros(self.model.nv, dtype=self.dtype)
        
        # Setup rotation computation arrays
        if target_rot is not None:
            site_xquat = np.empty(4, dtype=self.dtype)
            neg_site_xquat = np.empty(4, dtype=self.dtype)
            err_rot_quat = np.empty(4, dtype=self.dtype)
        
        # Get reference ID and initial position/orientation
        use_site = site_name is not None
        if use_site:
            reference_id = self._get_site_id(site_name)
        else:
            reference_id = self._get_body_id(body_name)
        
        # Convergence tracking
        position_errors = []
        rotation_errors = []
        step_numbers = []
        
        steps = 0
        success = False
        
        for steps in range(max_steps):
            # Forward kinematics to update positions
            mj.mj_forward(self.model, self.data)
            
            # Get current position and orientation
            if use_site:
                current_pos = self.data.site_xpos[reference_id]
                current_mat = self.data.site_xmat[reference_id]
            else:
                current_pos = self.data.xpos[reference_id]
                current_mat = self.data.xmat[reference_id]
            
            # Compute errors
            err_norm = 0.0
            pos_error_norm = 0.0
            rot_error_norm = 0.0
            
            if target_pos is not None:
                err_pos[:] = target_pos - current_pos
                pos_error_norm = np.linalg.norm(err_pos)
                err_norm += pos_error_norm
            
            if target_rot is not None:
                mj.mju_mat2Quat(site_xquat, current_mat)
                mj.mju_negQuat(neg_site_xquat, site_xquat)
                mj.mju_mulQuat(err_rot_quat, target_rot, neg_site_xquat)
                mj.mju_quat2Vel(err_rot, err_rot_quat, 1)
                rot_error_norm = np.linalg.norm(err_rot)
                err_norm += rot_error_norm * rot_weight
            
            # Track errors for plotting
            step_numbers.append(steps)
            position_errors.append(pos_error_norm)
            rotation_errors.append(rot_error_norm)
            
            # Check convergence
            if err_norm < tol:
                if self.verbose:
                    print(f'Converged after {steps} steps: err_norm={err_norm:.6g}')
                success = True
                break
            
            # Compute Jacobian
            if use_site:
                mj.mj_jacSite(self.model, self.data, jac_pos, jac_rot, reference_id)
            else:
                mj.mj_jacBody(self.model, self.data, jac_pos, jac_rot, reference_id)
            
            # Extract joint Jacobian for the specified number of DOFs
            jac_joints = jac[:, :self.n_dof]
            
            # Apply regularization
            reg_strength = (regularization_strength if err_norm > regularization_threshold else 0.0)
            update_joints = self._nullspace_method(jac_joints, err, regularization_strength=reg_strength)
            
            update_norm = np.linalg.norm(update_joints)
            
            # Check progress and update norm (only after specified delay)
            if steps >= progress_check_delay:
                # Check progress
                if update_norm > 0:  # Avoid division by zero
                    progress_criterion = err_norm / update_norm
                    if progress_criterion > progress_thresh:
                        if self.verbose:
                            print(f'Step {steps}: err_norm / update_norm ({progress_criterion:.3g}) > '
                                  f'tolerance ({progress_thresh:.3g}). Halting due to insufficient progress')
                        break
                
                # Limit update norm
                if update_norm > max_update_norm:
                    update_joints *= max_update_norm / update_norm
            
            # Update joint positions (only for the specified DOFs)
            update_nv[:] = 0  # Reset all updates to zero
            update_nv[:self.n_dof] = update_joints  # Apply updates only to specified DOFs
            mj.mj_integratePos(self.model, self.data.qpos, update_nv, 1)
            
            # Print progress
            if self.verbose and (steps % 50 == 0 or steps < 10):
                print(f'Step {steps}: pos_err={pos_error_norm:.6g}, rot_err={rot_error_norm:.6g}, '
                      f'total_err={err_norm:.6g}, update_norm={update_norm:.6g}')
        
        # Final status
        if not success and self.verbose:
            print(f'Failed to converge after {steps} steps: err_norm={err_norm:.6g}')
        
        # Prepare return data
        qpos = self.data.qpos.copy()
        convergence_data = {
            "step_numbers": step_numbers,
            "position_errors": position_errors,
            "rotation_errors": rotation_errors
        }
        
        return {
            "qpos": qpos,
            "success": success,
            "steps": steps,
            "err_norm": err_norm,
            "convergence_data": convergence_data
        }
    
    @staticmethod
    def plot_convergence(convergence_data: Dict, save_path: str = "ik_convergence.png",
                        title: str = "IK Convergence", show_plot: bool = False) -> None:
        """
        Plot position and rotation errors over IK iterations.
        
        Args:
            convergence_data: Dictionary with step_numbers, position_errors, rotation_errors
            save_path: Path to save the plot
            title: Title for the plot
            show_plot: Whether to display the plot (in addition to saving)
        """
        step_numbers = convergence_data["step_numbers"]
        position_errors = convergence_data["position_errors"]
        rotation_errors = convergence_data["rotation_errors"]
        
        if not step_numbers:
            if show_plot:
                print("No convergence data to plot")
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot position error
        ax1.plot(step_numbers, position_errors, 'b-', linewidth=2, label='Position Error')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Position Error (m)')
        ax1.set_title(f'{title} - Position Error')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        if max(position_errors) > 0:
            ax1.set_yscale('log')
        
        # Plot rotation error
        ax2.plot(step_numbers, rotation_errors, 'r-', linewidth=2, label='Rotation Error')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Rotation Error (rad)')
        ax2.set_title(f'{title} - Rotation Error')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        if max(rotation_errors) > 0:
            ax2.set_yscale('log')
        
        # Add convergence statistics as text
        final_pos_err = position_errors[-1] if position_errors else 0
        final_rot_err = rotation_errors[-1] if rotation_errors else 0
        iterations = len(step_numbers)
        
        stats_text = f'Final Position Error: {final_pos_err:.2e} m\n'
        stats_text += f'Final Rotation Error: {final_rot_err:.2e} rad\n'
        stats_text += f'Total Iterations: {iterations}'
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # Adjust layout and save
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        if show_plot:
            print(f"Convergence plot saved to: {save_path}")


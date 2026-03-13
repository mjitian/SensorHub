#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tiangong 2.0 Pro (天工2.0 Pro) deployment adapter for the fine-tuned XR-1 model.

This module provides a thin adapter layer between the XR-1 ACT policy inference
output and the Tiangong 2.0 Pro robot hardware, ensuring that the 26-dim action
vector produced by the fine-tuned model is correctly mapped to the robot's joints.

Usage:
    Configure TIANGONG_JOINT_MAPPING below to match your hardware setup,
    then import TiangongPolicyAdapter in place of PolicyAgent from action_policy.py.

References:
    - x-humanoid-training-toolchain deployment:
      https://github.com/Open-X-Humanoid/x-humanoid-training-toolchain/tree/main/deployment
"""

import numpy as np

# ---------------------------------------------------------------------------
# Default XR-1 action-vector layout (26-dim)
# Adjust these constants if your fine-tuning data used a different convention.
# ---------------------------------------------------------------------------
XR1_LEFT_ARM_SLICE = slice(0, 7)    # indices 0–6:  left arm  (7 joints)
XR1_LEFT_HAND_SLICE = slice(7, 13)  # indices 7–12: left hand  (6 joints)
XR1_RIGHT_ARM_SLICE = slice(13, 20) # indices 13–19: right arm (7 joints)
XR1_RIGHT_HAND_SLICE = slice(20, 26)# indices 20–25: right hand (6 joints)
XR1_ACTION_DIM = 26

# ---------------------------------------------------------------------------
# Tiangong 2.0 Pro joint mapping
#
# Set each entry to the corresponding XR-1 action-vector index, or to None
# if that joint is not present / should be held at zero.
#
# Example: if Tiangong's joint 0 corresponds to XR-1 action index 0, set
#          TIANGONG_JOINT_MAPPING[0] = 0.
#
# This identity mapping assumes the fine-tuning data and the robot hardware
# share the same 26-dim convention.  Modify as needed.
# ---------------------------------------------------------------------------
TIANGONG_JOINT_MAPPING = list(range(XR1_ACTION_DIM))


class TiangongPolicyAdapter:
    """Wraps a PolicyAgent and re-maps actions to Tiangong 2.0 Pro joint order.

    If the fine-tuning data was collected on the same robot with the same joint
    convention as the deployment target, this adapter is a transparent pass-through
    (TIANGONG_JOINT_MAPPING stays as the identity).  If the joint order differs
    between the training data and the hardware, edit TIANGONG_JOINT_MAPPING above.

    Args:
        policy_agent: A PolicyAgent instance (from action_policy.py).
        joint_mapping (list[int | None]): Per-joint mapping from Tiangong index to
            XR-1 action-vector index.  Defaults to TIANGONG_JOINT_MAPPING.
    """

    def __init__(self, policy_agent, joint_mapping=None):
        self.agent = policy_agent
        self.joint_mapping = joint_mapping if joint_mapping is not None else TIANGONG_JOINT_MAPPING

    def inference(self, obs):
        """Run policy inference and remap the output to Tiangong joint order.

        Args:
            obs (dict | None): Observation dict passed directly to the wrapped agent.

        Returns:
            np.ndarray: Remapped action vector sized len(joint_mapping).
        """
        raw_action = self.agent.inference(obs)

        # Convert to numpy if necessary (e.g. torch.Tensor)
        if hasattr(raw_action, "cpu"):
            raw_action = raw_action.cpu().numpy()
        raw_action = np.asarray(raw_action).flatten()

        return self._remap(raw_action)

    def _remap(self, action):
        """Apply the joint mapping to a raw XR-1 action vector.

        Args:
            action (np.ndarray): Raw action vector from the policy model.

        Returns:
            np.ndarray: Remapped action vector.
        """
        remapped = np.zeros(len(self.joint_mapping))
        for tg_idx, xr1_idx in enumerate(self.joint_mapping):
            if xr1_idx is not None and xr1_idx < len(action):
                remapped[tg_idx] = action[xr1_idx]
        return remapped

    def reset(self):
        """Delegate reset to the wrapped policy agent."""
        self.agent.reset()

    def publish_action(self, action):
        """Split an already-remapped action vector into the four limb groups.

        This method operates on an action vector that has **already been remapped**
        by :meth:`inference` (or :meth:`_remap`).  It assumes the remapped vector
        still follows the XR-1 26-dim convention after mapping; if your
        ``joint_mapping`` changes the ordering, adjust the slices accordingly.

        Args:
            action (np.ndarray): Remapped action vector (output of :meth:`inference`).

        Returns:
            tuple: (target_joint, left_hand_pos, right_hand_pos)
                - target_joint (np.ndarray): 14-dim array (left arm 7 + right arm 7)
                - left_hand_pos (np.ndarray): 6-dim left hand positions
                - right_hand_pos (np.ndarray): 6-dim right hand positions
        """
        target_joint = np.concatenate(
            [action[XR1_LEFT_ARM_SLICE], action[XR1_RIGHT_ARM_SLICE]]
        )
        left_hand_pos = action[XR1_LEFT_HAND_SLICE]
        right_hand_pos = action[XR1_RIGHT_HAND_SLICE]
        return target_joint, left_hand_pos, right_hand_pos


def verify_action_space(model_path):
    """Check whether the saved model's action/observation dimensions are compatible
    with the Tiangong 2.0 Pro hardware configuration.

    Args:
        model_path (str): Path to the fine-tuned model checkpoint.

    Returns:
        bool: True if compatible, False otherwise.
    """
    try:
        from lerobot.common.policies.act.modeling_act import ACTPolicy as Policy  # noqa: PLC0415
    except ImportError:
        print("[ERROR] lerobot is not installed. Run: pip install lerobot")
        return False

    try:
        policy = Policy.from_pretrained(model_path)
    except FileNotFoundError:
        print(f"[ERROR] Model checkpoint not found at '{model_path}'. Check the path and try again.")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to load model from '{model_path}': {exc}")
        return False

    try:
        output_shapes = policy.config.output_shapes
        input_shapes = policy.config.input_shapes
        action_dim = output_shapes["action"][0] if "action" in output_shapes else None
        obs_dim = input_shapes["observation.state"][0] if "observation.state" in input_shapes else None
    except (AttributeError, KeyError, IndexError) as exc:
        print(f"[ERROR] Could not read model config shapes: {exc}")
        return False

    print(f"Model action dim : {action_dim}")
    print(f"Model obs dim    : {obs_dim}")
    print(f"Expected action dim (XR-1 default): {XR1_ACTION_DIM}")

    if action_dim != XR1_ACTION_DIM:
        print(
            f"[WARNING] Action dimension mismatch: model outputs {action_dim}-dim "
            f"but the default mapping expects {XR1_ACTION_DIM}-dim. "
            "Update TIANGONG_JOINT_MAPPING to match your hardware."
        )
        return False

    print("[OK] Action dimension matches XR-1 default convention.")
    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tiangong_adapter.py <model_path>")
        sys.exit(1)

    verify_action_space(sys.argv[1])

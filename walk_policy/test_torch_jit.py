import os
import torch


class NaturalWalkPolicy:
    def __init__(self, model_path: str, device: str):
        # load natural walk policy
        try:
            self.policy = torch.jit.load(model_path, map_location=device)
            print(f"Load policy from {model_path} on device {device}")
        except Exception as e:
            print(f"error: {e}")

        # set to evaluation mode
        self.policy.eval()

    def inference(self, observation: torch.Tensor) -> torch.Tensor:
        """Inference the policy with a control frequency of 100Hz.

        Args:
            observation: policy input. Shape: (num_envs, 15*72+3)
            The policy input is a history of proprioceptive observation [o_t-N+1, o_t-N+2, ..., o_t] and user commands,
            where N=15 is the history length and the single frame proprioceptive observation is the concatenation of the following:
            - 22-dimensional joint position (q-q_default) scaled by obs_scales.dof_pos 1.0
            - 22-dimensional joint velocity (dq) scaled by obs_scales.dof_vel 0.05
            - 22-dimensional last action (a)
            - 3-dimensional base angular velocity (wx, wy, wz) scaled by obs_scales.base_ang_vel 1.0
            - 3-dimensional gravity vector projected to the local base frame (gx, gy, gz)
            The user commands are the concatenation of the following:
            - 3-dimensional command velocity (vx, vy, w) scaled by obs_scales.commands (2.0, 2.0, 1.0)
            The default joint position q_default=[
                -0.06, 0.0, 0.0, 0.12, -0.06, 0.0,
                -0.06, 0.0, 0.0, 0.12, -0.06, 0.0,
                0.0, 0.15, 0.0, -0.25, 0.0,
                0.0, -0.15, 0.0, -0.25, 0.0].

        Returns:
            action: policy output. Shape: (num_envs, 22)
            The policy output is a 22-dimensional action used to calculate the target joint position q_t = alpha * a + q_default,
            where alpha is the action scale = [
                0.5, 0.2, 0.2, 0.5, 0.5, 0.2,
                0.5, 0.2, 0.2, 0.5, 0.5, 0.2,
                0.2, 0.2, 0.05, 0.2, 0.05,
                0.2, 0.2, 0.05, 0.2, 0.05].
            Then the target joint torque can be calculated as tau_t = Kp * (q_t - q) + Kd * (0 - dq),
            where Kp and Kd are the position and velocity gains, respectively.
            Kp = [
                110, 70, 70, 110, 30, 30,
                110, 70, 70, 110, 30, 30,
                30, 30, 30, 30, 30,
                30, 30, 30, 30, 30].
            Kd = [
                5.0, 3.0, 3.0, 5.0, 0.3, 0.3,
                5.0, 3.0, 3.0, 5.0, 0.3, 0.3,
                0.3, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.3, 0.3, 0.3, 0.3].
        """
        with torch.no_grad():
            action = self.policy(observation)
        return action


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "policy.pt")
    print(f"model_path: {model_path}")
    walk_policy = NaturalWalkPolicy(model_path, device="cuda:0")

    # create test input data
    num_envs = 16
    observation = torch.randn(num_envs, 1083, device="cuda:0")
    print(f"observation shape: {observation.shape}")

    # inference
    action = walk_policy.inference(observation)
    print(f"action shape: {action.shape}")
    print(f"action: {action}")


if __name__ == "__main__":
    main()

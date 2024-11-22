from stable_baselines3.ppo import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

class DebugPPO(PPO):
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        result = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        # Debug: Check the rollout buffer observations
        print(f"Rollout buffer observations shape: {rollout_buffer.observations.shape}")
        print(f"Rollout buffer first observation stats: Min={rollout_buffer.observations[0].min()}, Max={rollout_buffer.observations[0].max()}, Shape={rollout_buffer.observations[0].shape}")
        return result

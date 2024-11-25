from stable_baselines3.common.vec_env import DummyVecEnv


class ModVecEnv(DummyVecEnv):
    def update_values(self, values):
        self.set_attr("current_value", values)

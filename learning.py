from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from chess_env import SelfPlayChessEnv  # твой класс среды


# Callback для обновления оппонента
class OpponentUpdateCallback(BaseCallback):
    def __init__(self, env, update_freq=100_000, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.update_freq = update_freq

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % self.update_freq == 0:
            if self.verbose:
                print(f"\n🔁 Updating opponent at timestep {self.num_timesteps}")
            self.model.save("latest_model")
            new_opponent = MaskablePPO.load("latest_model")
            self.env.set_attr("opponent_model", new_opponent)
        return True


def make_env():
    return Monitor(SelfPlayChessEnv(engine_depth=6))


vec_env = DummyVecEnv([make_env])

# Инициализация модели с включенным tensorboard логированием
model = MaskablePPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log="./tensorboard_logs/"
)

# Изначально копируем модель в оппонента
model.save("latest_model")
initial_opponent = MaskablePPO.load("latest_model")
vec_env.set_attr("opponent_model", initial_opponent)

# Callback для обновления оппонента
opponent_callback = OpponentUpdateCallback(vec_env, update_freq=100_000)

# Запуск обучения с callback
model.learn(total_timesteps=500_000, callback=opponent_callback)

model.save("selfplay_chess_final")
vec_env.close()

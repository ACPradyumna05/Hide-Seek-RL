from config import MODEL_PATH
from environment import HideSeekEnv

def train(steps, seed):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
    env = HideSeekEnv()
    check_env(env, warn=True)
    env = Monitor(env)
    MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
    model = PPO("MlpPolicy", env, verbose=1, seed=seed, n_steps=1024,
                batch_size=256, learning_rate=3e-4, gamma=0.99,
                ent_coef=0.015, policy_kwargs={"net_arch": [128, 128]})
    model.learn(total_timesteps=steps, progress_bar=False)
    model.save(str(MODEL_PATH))
    print(f"Saved policy to {MODEL_PATH}.zip")
    env.close()

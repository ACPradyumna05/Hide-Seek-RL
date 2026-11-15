from pathlib import Path
from config import WIDTH, HEIGHT, FPS, MODEL_PATH
from environment import HideSeekEnv
from policies import heuristic_action
from rendering import draw

def demo(policy, seed):
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Hide & Seek RL: Vision Cone")
    font = pygame.font.SysFont("consolas", 21, bold=True)
    smallfont = pygame.font.SysFont("consolas", 14)
    clock = pygame.time.Clock()
    env = HideSeekEnv()
    obs, _ = env.reset(seed=seed)
    model = None
    if policy == "ppo":
        from stable_baselines3 import PPO
        path = str(MODEL_PATH) + ".zip"
        if not Path(path).exists():
            raise SystemExit(f"No model at {path}. First run --mode train")
        model = PPO.load(path)
    paused = False
    end_at = None
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    end_at = None
        if not paused:
            if end_at is not None:
                if pygame.time.get_ticks() >= end_at:
                    obs, _ = env.reset()
                    end_at = None
            else:
                if policy == "random":
                    action = env.action_space.sample()
                elif policy == "heuristic":
                    action = heuristic_action(env)
                elif policy == "ppo":
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    keys = pygame.key.get_pressed()
                    action = (1 if keys[pygame.K_w] or keys[pygame.K_UP] else
                              2 if keys[pygame.K_s] or keys[pygame.K_DOWN] else
                              3 if keys[pygame.K_a] or keys[pygame.K_LEFT] else
                              4 if keys[pygame.K_d] or keys[pygame.K_RIGHT] else 0)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    print(f"Episode ended: survived {info['survival_seconds']:.1f}s; "
                          f"{'SURVIVED' if info['won'] else 'CAUGHT'}")
                    end_at = pygame.time.get_ticks() + 1500
        draw(env, screen, font, smallfont, policy, paused, end_at)
        clock.tick(FPS)
    pygame.quit()

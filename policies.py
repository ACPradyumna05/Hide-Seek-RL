import numpy as np
from config import HIDER_SPEED, R_HIDER

def heuristic_action(env):
    if env.hider_senses_seeker():
        away = env.hider - env.seeker
        scores = []
        for i, d in enumerate([(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]):
            nxt = env.move(env.hider, np.array(d) * HIDER_SPEED, R_HIDER)
            score = float(np.linalg.norm(nxt - env.seeker))
            if not env.clear_los(nxt, env.seeker):
                score += 65
            if i != 0 and np.linalg.norm(nxt - env.hider) < 1:
                score -= 50
            scores.append(score)
        return int(np.argmax(scores))
    return int(env.rng.integers(0, 5))

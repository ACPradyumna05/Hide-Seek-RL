import math
from pathlib import Path

WIDTH, HEIGHT = 800, 520
FPS = 10
MAX_STEPS = 300
HIDER_SPEED = 8.0
SEEKER_SPEED = 7.0
R_HIDER, R_SEEKER = 12.0, 14.0
FOV_HALF = math.radians(50)
FOV_RANGE = 240.0
HEARING_RANGE = 105.0
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hider_ppo"
WALLS = [
    (175, 70, 48, 175),
    (345, 0, 47, 158),
    (525, 102, 47, 197),
    (260, 340, 178, 38),
    (80, 340, 72, 100),
    (637, 356, 60, 104),
]

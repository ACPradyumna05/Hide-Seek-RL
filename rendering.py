import math
import numpy as np
from config import WIDTH, HEIGHT, FPS, FOV_HALF, FOV_RANGE, WALLS, R_SEEKER, R_HIDER

def draw(env, screen, font, smallfont, policy_name, paused, end_at):
    import pygame
    screen.fill((232, 237, 230))
    points = [tuple(map(float, env.seeker))]

    for a in np.linspace(
        env.seeker_angle - FOV_HALF,
        env.seeker_angle + FOV_HALF,
        55
    ):
        unit = np.array(
            [math.cos(a), math.sin(a)],
            dtype=np.float32
        )

        far = env.seeker + unit * FOV_RANGE
        endpoint = far

        for d in np.arange(5, FOV_RANGE + 5, 5):
            p = env.seeker + unit * d

            if (
                not (0 <= p[0] <= WIDTH and 0 <= p[1] <= HEIGHT)
                or any(
                    rx <= p[0] <= rx + rw
                    and ry <= p[1] <= ry + rh
                    for rx, ry, rw, rh in WALLS
                )
            ):
                endpoint = env.seeker + unit * max(0, d - 5)
                break

        # IMPORTANT: Inside the outer loop,
        # but outside the inner ray-stepping loop.
        points.append(tuple(map(float, endpoint)))

    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.polygon(overlay, (236, 69, 65, 65), points)
    screen.blit(overlay, (0, 0))
    for x, y, w, h in WALLS:
        pygame.draw.rect(screen, (79, 91, 103), (x, y, w, h), border_radius=4)
        pygame.draw.rect(screen, (53, 64, 76), (x, y, w, h), 2, border_radius=4)
    if env.last_seen is not None and not env.seeker_spots_hider():
        pygame.draw.circle(screen, (211, 87, 82), env.last_seen.astype(int), 8, 2)
    pygame.draw.circle(screen, (235, 73, 65), env.seeker.astype(int), int(R_SEEKER))
    pygame.draw.circle(screen, (255, 255, 255), env.seeker.astype(int), int(R_SEEKER), 2)
    nose = env.seeker + 22 * np.array([math.cos(env.seeker_angle), math.sin(env.seeker_angle)])
    pygame.draw.line(screen, (115, 25, 28), env.seeker.astype(int), nose.astype(int), 3)
    pygame.draw.circle(screen, (55, 140, 238), env.hider.astype(int), int(R_HIDER))
    pygame.draw.circle(screen, (255, 255, 255), env.hider.astype(int), int(R_HIDER), 2)
    hud = pygame.Surface((WIDTH, 56), pygame.SRCALPHA)
    hud.fill((18, 28, 36, 218))
    screen.blit(hud, (0, 0))
    text = f"HIDE & SEEK  |  {policy_name.upper()}  |  {env.step_count / FPS:04.1f} / 30.0 sec"
    screen.blit(font.render(text, True, (255, 255, 255)), (16, 7))
    status = "SPOTTED!" if env.seeker_spots_hider() else "HIDDEN"
    screen.blit(smallfont.render(f"{status}     ESC quit | R reset | SPACE pause | WASD move in HUMAN mode", True,
                                 (255, 180, 175) if status == "SPOTTED!" else (169, 242, 192)), (16, 34))
    if env.caught or env.won or paused:
        msg = "BLUE SURVIVED!" if env.won else ("RED CAUGHT BLUE!" if env.caught else "PAUSED")
        s = font.render(msg, True, (255, 255, 255))
        bg = pygame.Surface((s.get_width() + 36, 54), pygame.SRCALPHA)
        bg.fill((15, 24, 35, 210))
        rect = bg.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(bg, rect)
        screen.blit(s, s.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
    pygame.display.flip()

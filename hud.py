"""HUD and game-over screen."""
import pygame

from .config import WIDTH, HEIGHT, ENEMY_EDGE


def draw_hud(screen, font, score, wave, enemies, ship):
    hud = font.render("WASD thrusters  Q/E rotate  SPACE fire  B auto-stop  R restart  ESC quit",
                      True, (110, 120, 140))
    screen.blit(hud, (8, HEIGHT - 22))
    screen.blit(font.render(f"SCORE {score}", True, (200, 210, 225)), (8, 8))
    screen.blit(font.render(f"WAVE {wave}", True, (120, 200, 255)), (8, 28))
    screen.blit(font.render(f"ENEMIES {len(enemies)}", True, ENEMY_EDGE), (8, 48))
    screen.blit(font.render(f"v={ship.vel.length():.0f}", True, (110, 120, 140)),
                (WIDTH - 70, HEIGHT - 22))
    _bar(screen, font, 8, 68, "PWR", ship.power_used, ship.power_supply,
         warn=ship.brownout)
    _bar(screen, font, 8, 88, "CPU", ship.compute_used, ship.compute_supply)
    if ship.shield_comp is not None:
        _bar(screen, font, 8, 108, "SHD", ship.shield_charge,
            ship.shield_comp.shield_max_charge, warn=not ship.shield_on)


def _bar(screen, font, x, y, label, used, supply, warn=False):
    screen.blit(font.render(f"{label} {used:.0f}/{supply:.0f}",
                            True, (200, 210, 225)), (x, y))
    bx, by, bw, bh = x + 54, y + 3, 100, 8
    pygame.draw.rect(screen, (40, 46, 58), (bx, by, bw, bh))
    frac = min(1.0, used / supply) if supply > 0 else 0.0
    if frac > 0:
        pygame.draw.rect(screen, (255, 120, 90) if warn else (120, 200, 255),
                         (bx, by, int(bw * frac), bh))


def draw_game_over(screen, big_font, font, score):
    go = big_font.render("GAME OVER", True, (255, 120, 90))
    screen.blit(go, (WIDTH / 2 - go.get_width() / 2, HEIGHT / 2 - 40))
    fs = font.render(f"final score {score}   press R to restart",
                     True, (200, 210, 225))
    screen.blit(fs, (WIDTH / 2 - fs.get_width() / 2, HEIGHT / 2 + 10))

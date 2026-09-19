"""Audition all synthesized SFX without playing the game.

Run:  python -m ship5.test_sound
Plays each one-shot in sequence, then the thruster loop for 2 s.
Audio only — no display needed.
"""
import time

import pygame

from .sound import SoundBank

# (name, seconds to wait before the next one)
ORDER = [
    ("laser", 0.8),
    ("enemy_laser", 0.8),
    ("explosion", 1.0),
    ("small_explosion", 0.8),
    ("shield_hit", 0.8),
    ("missile_launch", 1.0),
    ("game_over", 1.6),
]


def main():
    pygame.init()
    bank = SoundBank()
    bank.init()
    if not bank.ready:
        print("no audio device -- nothing to play")
        return
    for name, gap in ORDER:
        print(">", name)
        bank.play(name)
        time.sleep(gap)
    print("> thruster (2 s)")
    bank.set_thruster(True)
    time.sleep(2.0)
    bank.set_thruster(False)
    pygame.quit()


if __name__ == "__main__":
    main()
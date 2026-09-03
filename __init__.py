"""Asteroids-like space fighter.

W/S  forward / reverse thrusters
A/D  lateral thrusters (A = starboard fires -> ship goes left)
Q/E  rotate (aim the gun)
SPACE fire
B    retro-thrust: fires the right thrusters to kill your velocity
R    restart (after game over)
ESC  quit

Enemy ships chase, circle, and shoot back. They take 3 hits and also
die if they crash into an asteroid. Each kill instantly spawns a
replacement far from the player.

Run with:  python -m ship5
"""

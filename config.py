"""Tuning knobs and colors. Tweak game feel here, not in the classes."""

# --- screen ---
WIDTH, HEIGHT = 1920, 1280
FPS = 60

# --- camera / infinite world ---
LOOK_AHEAD = 0.6        # seconds of velocity the camera leads by
CAMERA_SMOOTHING = 6.0  # higher = snappier
CAMERA_LEAD_SMOOTHING = 4.0
MAX_DIST = 1200         # tether radius from group centroid (multiplayer)
TETHER_STRENGTH = 0.5   # restoring accel per px of tether overshoot

# --- infinite field (sector spawning) ---
SECTOR_SIZE = 1000          # px per sector cell
ACTIVE_SECTORS = 2          # Chebyshev radius of sectors kept populated
ROCKS_PER_SECTOR = 3        # target rocks per active sector
DESPAWN_RADIUS = 2600       # rocks farther than this from the group are culled
SPAWN_CLEAR_RADIUS = 500    # never spawn closer than this to a player
MAX_SPAWNS_PER_TICK = 4     # fill the field gradually, not all at once
WAVE_INTERVAL = 40          # seconds between difficulty waves

# --- power / compute (Phase 2) ---
POWER_HYSTERESIS = 0.10   # brownout latch deadband, fraction of supply
AUTO_STOP_COMPUTE = 10.0  # compute drawn by auto-stop guidance (B)

# --- ship feel knobs ---
SHIP_ACCEL = 320.0
ROT_SPEED = 3.6
MAX_SPEED = 520.0
BULLET_SPEED = 860.0
BULLET_LIFE = 1
FIRE_COOLDOWN = 0.15
MAX_BULLETS = 3
STOP_GAIN = 3.0
MAX_STOP_ACCEL = 600.0
STOP_DEADBAND = 1.5
SHIP_RADIUS = 12.0
SPAWN_PROTECT = 3.5      # seconds of shield after (re)start

SHIELD_DUMP_DECAY = 50.0  # power/s, rate the hit-dump decays

# --- asteroid knobs ---
ROCK_SIZES = {
    'large':  {'radius': 50, 'score': 20,  'speed': (20, 60),  'spin': (0.2, 0.8)},
    'medium': {'radius': 28, 'score': 50,  'speed': (40, 90),  'spin': (0.4, 1.2)},
    'small':  {'radius': 15, 'score': 100, 'speed': (70, 130), 'spin': (0.8, 2.0)},
}
ROCK_SPLIT = {'large': 'medium', 'medium': 'small', 'small': None}
STARTING_WAVE = 2        # large rocks in wave 1; +1 per wave

# --- enemy ship knobs ---
ENEMY_ACCEL = 240.0
ENEMY_MAX_SPEED = 500.0
ENEMY_ROT_SPEED = 3.4
ENEMY_RADIUS = 11.0
ENEMY_HP = 1
ENEMY_SCORE = 250
ENEMY_BULLET_SPEED = 420.0
ENEMY_BULLET_LIFE = 1
ENEMY_FIRE_COOLDOWN = .25
ENEMY_FIRE_SPREAD = 0.16
ENEMY_ENGAGE_RANGE = 700.0  # only chase/fire within this range
ENEMY_ORBIT_OFFSET = 140.0  # aim offset so it circles instead of ramming
ENEMY_AVOID_RADIUS = 200.0  # how far ahead the enemy "sees" rocks
ENEMY_AVOID_WEIGHT = 3.2    # strength of the avoidance steering

# --- fog of war knobs ---
FOG_ALPHA = 100          # how dark the fog is (0-255)
FOG_BASE_RADIUS = 160.0  # all-around visibility around the ship
FOG_FRONT_RADIUS = 430.0 # reach of the forward lobe
FOG_SIDE_RADIUS = 200.0  # width of the forward lobe
FOG_FRONT_OFFSET = 120.0 # how far ahead of the ship the lobe is centered

# --- colors ---
BG = (10, 12, 18)
SHIP_COLOR = (135, 145, 160)
SHIP_EDGE = (70, 130, 170)
FLAME_OUT = (255, 150, 50)
FLAME_IN = (255, 225, 160)
BULLET_COLOR = (255, 230, 120)
ENEMY_FILL = (110, 30, 45)
ENEMY_EDGE = (255, 90, 110)
ENEMY_FLAME = (255, 120, 80)
ENEMY_BULLET_COLOR = (255, 90, 120)
STAR_COLOR = (70, 78, 95)
ROCK_FILL = (55, 62, 75)
ROCK_EDGE = (190, 200, 215)
PARTICLE_COLORS = [(200, 210, 225), (150, 160, 175), (255, 150, 50)]

# --- timing ---
TICK = 1 / 60        # fixed sim step, seconds
MAX_FRAME_DT = 0.25  # clamp frame dt so a hiccup can't trigger a catch-up spiral

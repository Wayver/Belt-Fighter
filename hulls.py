"""Hull and component type definitions — pure data, no rendering.

A HullType is a polygon plus a set of named mount points (slots). A
ComponentType declares which slot types it fits and its stats. A Ship is
built from a HullType plus a loadout of components.

Orientation semantics (important):
- Slot.orientation is the direction of the FORCE on the ship (the way the
  part pushes it), in hull-local coords. The exhaust is the opposite.
- Slot.flame_dir is the visual exhaust direction, defaulting to
  -orientation. It exists so a thruster can push one way but show its
  flame another (e.g. RCS pods: the force is across the ship, but the
  flame is drawn outward so it stays visible past the hull edge).

Thrust values are expressed off config.SHIP_ACCEL so config.py remains
the tuning surface.
"""
from dataclasses import dataclass

from .config import (SHIP_ACCEL, FIRE_COOLDOWN, BULLET_SPEED,
                     ENEMY_FIRE_COOLDOWN, ENEMY_BULLET_SPEED)


@dataclass(frozen=True)
class Slot:
    """A mount point on a hull.

    name:        stable id, used to key flames and the loadout
    slot_type:   what kind of component fits ('thruster', 'weapon', ...)
    position:    (lx, ly) in hull-local coords (+x = nose, +y = starboard)
    orientation: (dx, dy) local direction of the FORCE on the ship
    flame_dir:   (dx, dy) visual exhaust direction; None = -orientation
    flame_key:   which flame_mags bucket this slot renders under ('' = none)
    flame_scale / flame_width: visual tuning for the flame
    """
    name: str
    slot_type: str
    position: tuple
    orientation: tuple = (1, 0)
    flame_dir: tuple = None
    flame_key: str = ''
    flame_scale: float = 0.6
    flame_width: int = 3

    def __post_init__(self):
        if self.flame_dir is None:
            object.__setattr__(self, 'flame_dir',
                               (-self.orientation[0], -self.orientation[1]))


@dataclass(frozen=True)
class HullType:
    """Static geometry + slots for one hull design."""
    id: str
    polygon: tuple                 # hull outline in local coords
    slots: tuple                   # tuple of Slot
    base_mass: float = 1.0
    collision_radius: float = 12.0
    nose: tuple = (18, 0)          # fallback muzzle / ram anchor
    cockpit: tuple = (8, 0)        # small cockpit dot
    max_speed_factor: float = 1.0  # per-hull top-speed multiplier
    turn_rate_factor: float = 1.0  # per-hull turn-rate multiplier
    fill: tuple = None     # body color; None = config SHIP_COLOR
    edge: tuple = None     # edge color; None = config SHIP_EDGE

@dataclass(frozen=True)
class ComponentType:
    """A mountable part: fit + stats.

    `thrust` is the force this part applies at full demand. Power/compute:
    `power_idle` is drawn whenever fitted, `power_active` while active
    (scaled by demand), `compute_demand` while active. Generators provide
    `power_supply` / `compute_supply`. `priority` (lower = first) decides
    who keeps power when supply is short.
    """
    id: str
    name: str
    slot_types: tuple              # slot types this part fits
    mass: float = 0.0
    thrust: float = 0.0
    power_idle: float = 0.0
    power_active: float = 0.0
    compute_demand: float = 0.0
    power_supply: float = 0.0
    compute_supply: float = 0.0
    priority: int = 0
    power_hit: float = 0.0
    shield_max_charge: float = 0.0
    shield_recharge_rate: float = 0.0
    fire_cooldown: float = 0.0   # seconds between shots
    bullet_speed: float = 0.0    # px/s, expressed off config below
    fire_cooldown: float = 0.0
    bullet_speed: float = 0.0
    # --- laser (charge weapon) ---
    laser_arc_start_deg: float = 0.0   # wedge start, deg rel. to nose (+ = starboard)
    laser_arc_end_deg: float = 0.0     # wedge end
    laser_range: float = 0.0           # 0 = not a laser; else max firing range
    laser_charge_time: float = 0.0     # seconds to full charge
    laser_damage: int = 0
    laser_discharge_dump: float = 0.0  # power spike added on fire

# --- The current ship, as data ---
# orientation = force direction on the ship (exhaust is the opposite).

FORWARD_S = Slot('forward_s', 'thruster', (-13, 8), (1, 0), flame_key='forward')
FORWARD_P = Slot('forward_p', 'thruster', (-13, -8), (1, 0), flame_key='forward')
REVERSE   = Slot('reverse', 'thruster', (16, 0), (-1, 0), flame_key='reverse')
# RCS: force is across the ship (port/starboard), but the flame is drawn
# outward (flame_dir) so it stays visible instead of ending under the hull.
RCS_LEFT  = Slot('to_left', 'thruster', (-10, -9.5), (0, -1),
                 flame_dir=(0, -1), flame_key='to_left',
                 flame_scale=0.5, flame_width=2)
RCS_RIGHT = Slot('to_right', 'thruster', (-10, 9.5), (0, 1),
                 flame_dir=(0, 1), flame_key='to_right',
                 flame_scale=0.5, flame_width=2)
GUN       = Slot('gun', 'weapon', (18, 0), (1, 0))
# Generators: plain mount points, no orientation/flame semantics.
REACTOR   = Slot('reactor', 'reactor', (-4, 0))
COMPUTER  = Slot('computer', 'computer', (2, 0))
SHIELD    = Slot('shield', 'shield', (0, 0))

DEFAULT_HULL = HullType(
    id='scout',
    polygon=(
        (18, 0),     # nose tip
        (14, 3.5),
        (8, 6.5),
        (0, 8),
        (-8, 8),
        (-12, 11),   # starboard pod, outer rear (sticks out)
        (-12, 5),    # starboard pod, inner rear
        (-9, 3),     # notch between engines
        (-9, -3),
        (-12, -5),   # port pod, inner rear
        (-12, -11),  # port pod, outer rear (sticks out)
        (-8, -8),
        (0, -8),
        (8, -6.5),
        (14, -3.5),
    ),
    slots=(FORWARD_S, FORWARD_P, REVERSE, RCS_LEFT, RCS_RIGHT, GUN,
           REACTOR, COMPUTER, SHIELD),
    base_mass=1.0,
    collision_radius=12.0,
    nose=(18, 0),
    cockpit=(8, 0),
)

# Component types for the default loadout.
# Two main engines sum to SHIP_ACCEL straight ahead; nose + RCS match the
# old single-vector magnitudes.
#
## Stock power/compute budget (deliberately generous — feel must not change):
#   power idle:   2*1 + 1 + 2*1 + 2 + 5 + 2 = 14
#   power steady: + 10 (shield) = 24 at rest
#   power max:    + 2*20 + 15 + 2*5 + 5 = 94  (of 100)  -> never brownouts
#   hit dump:     + 25 (transient, decays 50/s) -> brief brownout at full thrust
#
# Priority: RCS (1) is fine control near rocks, mains (2) propulsion,
# reverse (3) sheds first.
MAIN_ENGINE   = ComponentType('main_engine', 'Main Engine', ('thruster',),
                              mass=2.0, thrust=SHIP_ACCEL / 2,
                              power_idle=1.0, power_active=20.0, priority=2)
NOSE_THRUSTER = ComponentType('nose_thruster', 'Nose Thruster', ('thruster',),
                              mass=1.0, thrust=SHIP_ACCEL,
                              power_idle=1.0, power_active=15.0, priority=3)
RCS           = ComponentType('rcs', 'RCS', ('thruster',),
                              mass=0.5, thrust=SHIP_ACCEL,
                              power_idle=1.0, power_active=5.0,
                              compute_demand=5.0, priority=1)
RCS_HEAVY = ComponentType('rcs_heavy', 'Heavy RCS', ('thruster',),
                          mass=0.75, thrust=SHIP_ACCEL * 1.5,
                          power_idle=1.0, power_active=8.0,
                          compute_demand=5.0, priority=1)
GUN_TYPE      = ComponentType('gun', 'Pulse Gun', ('weapon',), mass=1.0,
                              power_idle=2.0, power_active=5.0,
                              fire_cooldown=FIRE_COOLDOWN,
                              bullet_speed=BULLET_SPEED, priority=2)
REACTOR_TYPE  = ComponentType('reactor', 'Reactor', ('reactor',),
                              mass=3.0, power_supply=100.0)
COMPUTER_TYPE = ComponentType('computer', 'Computer', ('computer',),
                              mass=2.0, compute_supply=50.0)
SHIELD_TYPE = ComponentType('shield', 'Shield', ('shield',),
                            mass=2.0,
                            power_idle=2.0, power_active=10.0,
                            power_hit=25.0,
                            shield_max_charge=3.0,
                            shield_recharge_rate=0.5)

E_SHIELD_TYPE = ComponentType('shield', 'Shield', ('shield',),
                             mass=2.0,
                             power_idle=2.0, power_active=10.0,
                             power_hit=45.0,
                             shield_max_charge=5.0,
                             shield_recharge_rate=0.5)


LASER_TYPE = ComponentType('laser', 'Laser', ('weapon',),
    mass=3.0, power_idle=2.0, power_active=12.0,   # power_active = charge draw
    laser_arc_start_deg=-20.0, laser_arc_end_deg=20.0,
    laser_range=600.0,          # <-- the arbitrary range; turn this
    laser_charge_time=0.5,
    laser_damage=2,
    laser_discharge_dump=30.0,
    priority=2)

LASER_S = ComponentType('laser_s', 'Laser (Starboard)', ('weapon',),
    mass=3.0, power_idle=2.0, power_active=12.0,
    laser_arc_start_deg=0.0,  laser_arc_end_deg=110.0,   # nose to starboard
    laser_range=900.0,
    laser_charge_time=1.,
    laser_damage=2,
    laser_discharge_dump=30.0,
    priority=2)

LASER_P = ComponentType('laser_p', 'Laser (Port)', ('weapon',),
    mass=3.0, power_idle=2.0, power_active=12.0,
    laser_arc_start_deg=-110.0, laser_arc_end_deg=0.0,   # port to nose
    laser_range=900.0,
    laser_charge_time=0.5,
    laser_damage=2,
    laser_discharge_dump=30.0,
    priority=2)




# Silas's teeth: 5 x SHIP_ACCEL/5 = SHIP_ACCEL total (scout-mains parity).
TOOTH_THRUSTER = ComponentType('tooth_thruster', 'Tooth Thruster', ('thruster',),
                               mass=0.5, thrust=SHIP_ACCEL / 5,
                               power_idle=0.5, power_active=8.0, priority=2)

def default_loadout(hull=None):
    """slot_name -> ComponentType, the stock fit for a hull."""
    hull = hull or DEFAULT_HULL
    is_bb = hull.id == 'blackbird'
    is_silas = hull.id == 'silas'
    # Blackbird: strong forward drive, weak reverse (swapped vs the scout).
    fwd, rev = (NOSE_THRUSTER, MAIN_ENGINE) if is_bb else (MAIN_ENGINE, NOSE_THRUSTER)
    out = {}
    for s in hull.slots:
        if s.slot_type == 'thruster':
            if s.name in ('forward_s', 'forward_p'):
                out[s.name] = fwd
            elif s.name.startswith('tooth'):
                out[s.name] = TOOTH_THRUSTER
            elif s.name.startswith('reverse'):
                out[s.name] = rev
            else:
                out[s.name] = RCS_HEAVY if hull.id == 'blackbird' else RCS
        
        elif s.slot_type == 'weapon':
            if s.name == 'gun_s':
                out[s.name] = LASER_S
            elif s.name == 'gun_p':
                out[s.name] = LASER_P
            else:
                out[s.name] = GUN_TYPE


        elif s.slot_type == 'reactor':
            out[s.name] = REACTOR_TYPE
        elif s.slot_type == 'computer':
            out[s.name] = COMPUTER_TYPE
        elif s.slot_type == 'shield':
            out[s.name] = SHIELD_TYPE
    return out


# --- Player-selectable catalog ---
#PLAYER_HULLS = (DEFAULT_HULL, BLACKBIRD_HULL)   # future hulls append here

# What the menu offers per slot type.
COMPONENT_CATALOG = {
    'thruster': (MAIN_ENGINE, NOSE_THRUSTER, TOOTH_THRUSTER, RCS, RCS_HEAVY),
    'weapon':   (GUN_TYPE, LASER_TYPE, LASER_S, LASER_P),
    'reactor':  (REACTOR_TYPE,),
    'computer': (COMPUTER_TYPE,),
    'shield':   (SHIELD_TYPE,),
}

def validate_loadout(hull, loadout):
    """Every slot filled, every part compatible with its slot type."""
    slots = {s.name: s for s in hull.slots}
    if set(loadout) != set(slots):
        return False
    return all(s.slot_type in loadout[n].slot_types
               for n, s in slots.items())

def loadout_stats(hull, loadout):
    """Derived numbers for menu display — pure data, no sim."""
    parts = list(loadout.values())
    idle = sum(c.power_idle for c in parts)
    return {
        'mass': hull.base_mass + sum(c.mass for c in parts),
        'power_supply': sum(c.power_supply for c in parts),
        'power_idle': idle,
        'power_max': idle + sum(c.power_active for c in parts),
        'compute_supply': sum(c.compute_supply for c in parts),
        'shield_charge': max((c.shield_max_charge for c in parts), default=0.0),
    }


# --- Blackbird: long slender fuselage, swept delta wings, twin tails ---
# Twin reverse thrusters flank the nose; RCS pods sit on the wingtips.

BB_FORWARD_S = Slot('forward_s', 'thruster', (-15, 1.5), (1, 0), flame_key='forward')
BB_FORWARD_P = Slot('forward_p', 'thruster', (-15, -1.5), (1, 0), flame_key='forward')
# Twin reverse thrusters flanking the nose (shared 'reverse' flame bucket).
BB_REVERSE_S = Slot('reverse_s', 'thruster', (6, 5.5), (-1, 0), flame_key='reverse')
BB_REVERSE_P = Slot('reverse_p', 'thruster', (6, -5.5), (-1, 0), flame_key='reverse')
# RCS pods on the wingtips: force across the ship, flame drawn outward.
BB_RCS_L     = Slot('to_left', 'thruster', (-8, -13.5), (0, -1),
                    flame_dir=(0, -1), flame_key='to_left',
                    flame_scale=0.5, flame_width=2)
BB_RCS_R     = Slot('to_right', 'thruster', (-8, 13.5), (0, 1),
                    flame_dir=(0, 1), flame_key='to_right',
                    flame_scale=0.5, flame_width=2)
BB_GUN       = Slot('gun', 'weapon', (24, 0), (1, 0))
BB_REACTOR   = Slot('reactor', 'reactor', (-2, 0))
BB_COMPUTER  = Slot('computer', 'computer', (4, 0))
BB_SHIELD    = Slot('shield', 'shield', (0, 0))

BLACKBIRD_HULL = HullType(
    id='blackbird',
    polygon=(
        (26, 0),      # nose tip
        (14, 1.5),    # nose side (widened for the twin reverse thrusters)
        (10, 2.5),    # wing root leading edge
        (-6, 14),     # starboard wingtip, front
        (-10, 14),    # starboard wingtip, rear (flat tip)
        (-14, 6),     # wing trailing edge
        (-16, 5),     # starboard tail fin, front
        (-18, 5),     # starboard tail fin, rear
        (-18, 2.5),   # starboard tail fin, inner
        (-15, 2),     # rear fuselage corner
        (-15, 0),     # rear center (between exhausts)
        (-15, -2),
        (-18, -2.5),
        (-18, -5),
        (-16, -5),
        (-14, -6),
        (-10, -14),
        (-6, -14),
        (10, -2.5),
        (14, -1.5),
    ),
    slots=(BB_FORWARD_S, BB_FORWARD_P, BB_REVERSE_S, BB_REVERSE_P,
           BB_RCS_L, BB_RCS_R, BB_GUN, BB_REACTOR, BB_COMPUTER, BB_SHIELD),
    base_mass=1.5,
    collision_radius=14.0,
    nose=(26, 0),
    cockpit=(16, 0),
)

# --- Blackbird WG: Blackbird with wing guns on the leading edges ---
# Gun slots sit on the wing leading edge, midway between wingtip and
# wing root: midpoint of (10, 2.5) -> (-6, 14) is (2, 8.25).

BBWG_GUN_S = Slot('gun_s', 'weapon', (2, 8.25), (1, 0))
BBWG_GUN_P = Slot('gun_p', 'weapon', (2, -8.25), (1, 0))

BLACKBIRD_WG_HULL = HullType(
    id='blackbird_wg',
    polygon=BLACKBIRD_HULL.polygon,
    slots=(BB_FORWARD_S, BB_FORWARD_P, BB_REVERSE_S, BB_REVERSE_P,
           BB_RCS_L, BB_RCS_R, BB_GUN, BBWG_GUN_S, BBWG_GUN_P,
           BB_REACTOR, BB_COMPUTER, BB_SHIELD),
    base_mass=1.5,
    collision_radius=14.0,
    nose=(26, 0),
    cockpit=(16, 0),
)


# --- Silas: cartoon cat head. Orange and white, ears pointing forward,
# laser eyes, and a row of five tooth-thrusters along the chin.
# The teeth are the main drive (5 x SHIP_ACCEL/5 = SHIP_ACCEL total).

SILAS_TOOTH_S2 = Slot('tooth_s2', 'thruster', (-11.5, 5), (1, 0), flame_key='forward')
SILAS_TOOTH_S1 = Slot('tooth_s1', 'thruster', (-13, 2.5), (1, 0), flame_key='forward')
SILAS_TOOTH_C  = Slot('tooth_c',  'thruster', (-13.5, 0), (1, 0), flame_key='forward')
SILAS_TOOTH_P1 = Slot('tooth_p1', 'thruster', (-13, -2.5), (1, 0), flame_key='forward')
SILAS_TOOTH_P2 = Slot('tooth_p2', 'thruster', (-11.5, -5), (1, 0), flame_key='forward')
# The cat's nose: reverse thruster on the forehead (retro-thrust, flame fwd).
SILAS_REVERSE  = Slot('reverse', 'thruster', (13, 0), (-1, 0), flame_key='reverse')
# RCS on the cheeks.
SILAS_RCS_L = Slot('to_left', 'thruster', (-2, -13.5), (0, -1),
                   flame_dir=(0, -1), flame_key='to_left',
                   flame_scale=0.5, flame_width=2)
SILAS_RCS_R = Slot('to_right', 'thruster', (-2, 13.5), (0, 1),
                   flame_dir=(0, 1), flame_key='to_right',
                   flame_scale=0.5, flame_width=2)
# Laser eyes.
SILAS_EYE_S = Slot('eye_s', 'weapon', (8, 5.5), (1, 0))
SILAS_EYE_P = Slot('eye_p', 'weapon', (8, -5.5), (1, 0))
SILAS_REACTOR  = Slot('reactor', 'reactor', (-6, 0))
SILAS_COMPUTER = Slot('computer', 'computer', (-2, 0))
SILAS_SHIELD   = Slot('shield', 'shield', (0, 0))

SILAS_HULL = HullType(
    id='silas',
    polygon=(
        (19, 12),     # starboard ear tip (points forward)
        (6, 13),      # starboard ear outer base
        (-1, 15),     # starboard cheek (widest)
        (-9, 11),     # starboard lower cheek
        (-13, 6),     # starboard chin corner
        (-15, 3),     # chin
        (-15.5, 0),   # chin center (stern)
        (-15, -3),    # chin
        (-13, -6),    # port chin corner
        (-9, -11),    # port lower cheek
        (-1, -15),    # port cheek
        (6, -13),     # port ear outer base
        (19, -12),    # port ear tip (points forward)
        (10, -5.5),   # port ear inner base
        (13, 0),      # forehead (dip between the ears)
        (10, 5.5),    # starboard ear inner base
    ),
    slots=(SILAS_TOOTH_S2, SILAS_TOOTH_S1, SILAS_TOOTH_C,
           SILAS_TOOTH_P1, SILAS_TOOTH_P2, SILAS_REVERSE,
           SILAS_RCS_L, SILAS_RCS_R, SILAS_EYE_S, SILAS_EYE_P,
           SILAS_REACTOR, SILAS_COMPUTER, SILAS_SHIELD),
    base_mass=1.0,
    collision_radius=15.0,
    nose=(13, 0),
    cockpit=(11, 0),      # renders as the cat's nose dot
    max_speed_factor=1.0,
    turn_rate_factor=1.15,
    fill=(240, 150, 60),  # orange
    edge=(255, 245, 230), # white
)



# --- Enemy hull: slender dart body + forward wing gun pods ---
# Standard thruster slot names (forward_s/forward_p) so Ship._set_demands
# works unchanged. The wings carry the guns (gun_s/gun_p), not RCS.

E_FORWARD_S = Slot('forward_s', 'thruster', (-14, 2.5), (1, 0), flame_key='forward')
E_FORWARD_P = Slot('forward_p', 'thruster', (-14, -2.5), (1, 0), flame_key='forward')
# braking/strafing: same slot names the player uses, so Ship._set_demands
# and auto-stop work unchanged
E_REVERSE   = Slot('reverse', 'thruster', (24, 0), (-1, 0), flame_key='reverse')
E_RCS_L     = Slot('to_left', 'thruster', (-7, -8), (0, -1),
                   flame_dir=(0, -1), flame_key='to_left',
                   flame_scale=0.5, flame_width=2)
E_RCS_R     = Slot('to_right', 'thruster', (-7, 8), (0, 1),
                   flame_dir=(0, 1), flame_key='to_right',
                   flame_scale=0.5, flame_width=2)
E_GUN_S     = Slot('gun_s', 'weapon', (21, 11), (1, 0))
E_GUN_P     = Slot('gun_p', 'weapon', (21, -11), (1, 0))
E_REACTOR   = Slot('reactor', 'reactor', (-4, 0))
E_REACTOR_2 = Slot('reactor2', 'reactor', (-8, 0))
E_COMPUTER  = Slot('computer', 'computer', (2, 0))
E_SHIELD    = Slot('shield', 'shield', (0, 0))


ENEMY_HULL = HullType(
    id='interceptor',
    polygon=(
        (28, 0),     # nose tip (extended)
        (12, 3),     # fuselage shoulder (slender body)
        (10, 6),     # wing root leading edge (notch before the pod)
        (22, 9),     # starboard gun pod leading edge (extended)
        (22, 13),    # starboard gun pod front outer (extended)
        (4, 13),     # starboard wingtip (pod outer rear)
        (-0, 9),     # starboard wing trailing edge
        (-7, 7),    # starboard wingtip rear
        (-10, 3),    # fuselage rear corner (starboard)
        (-10, -3),   # fuselage rear corner (port)
        (-7, -7),   # port wingtip rear
        (-0, -9),    # port wing trailing edge
        (4, -13),    # port wingtip (pod outer rear)
        (22, -13),   # port gun pod front outer (extended)
        (22, -9),    # port gun pod leading edge (extended)
        (10, -6),    # port wing root leading edge
        (12, -3),    # fuselage shoulder (port)
    ),
    slots=(E_FORWARD_S, E_FORWARD_P, E_GUN_S, E_GUN_P, E_REACTOR, E_COMPUTER, E_SHIELD),
    base_mass=3.0,
    collision_radius=16.0,
    nose=(28, 0),
    cockpit=(10, 0),
)

# Enemy guns use the enemy's own config values (not the player's) — see notes.
ENEMY_GUN = ComponentType('enemy_gun', 'Enemy Gun', ('weapon',), mass=1.0,
                          power_idle=2.0, power_active=5.0,
                          fire_cooldown=ENEMY_FIRE_COOLDOWN,
                          bullet_speed=ENEMY_BULLET_SPEED, priority=2)


def enemy_loadout():
    return {
        'forward_s': MAIN_ENGINE,
        'forward_p': MAIN_ENGINE,
        'reverse': NOSE_THRUSTER,
        'to_left': RCS,
        'to_right': RCS,
        'gun_s': ENEMY_GUN,
        'gun_p': ENEMY_GUN,
        'reactor': REACTOR_TYPE,
        'reactor2': REACTOR_TYPE,
        'computer': COMPUTER_TYPE,
        'shield'  : E_SHIELD_TYPE,
    }


PLAYER_HULLS = (DEFAULT_HULL, BLACKBIRD_HULL, BLACKBIRD_WG_HULL, SILAS_HULL)   # future hulls append here

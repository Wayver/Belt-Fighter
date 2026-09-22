"""Ship selection menu: pick a mode, a hull, then fit components per slot.

Four screens, matching the game's monospace style:
  1. 'mode'    — choose single / host / join (UP/DOWN, ENTER)
  2. 'join'    — join only: type the host's IP + port (letters/digits/. and
                 ':' for the port, ENTER connect, ESC back)
  3. 'hull'    — choose from PLAYER_HULLS (UP/DOWN, ENTER)
  4. 'loadout' — cycle a component per slot (UP/DOWN slot, LEFT/RIGHT part,
                 ENTER launch, ESC/BACKSPACE back)

Produces (self.hull, self.loadout) for Game/Ship, plus the connection mode:
self.mode ('single'/'host'/'join'), self.host_ip, self.host_port (host: the
port to listen on; join: the host to connect to). Pure data + drawing; never
touches sim code.
"""
import math

import pygame

from .config import WIDTH, HEIGHT, BG, SHIP_COLOR, SHIP_EDGE
from .hulls import PLAYER_HULLS, COMPONENT_CATALOG, loadout_stats, default_loadout

DIM = (110, 120, 140)
BRIGHT = (200, 210, 225)
ACCENT = (120, 200, 255)
WARN = (255, 160, 80)
SEL_BG = (40, 48, 64)

MODES = [
    ("single", "SINGLE PLAYER", "solo run against the AI"),
    ("host",   "HOST GAME",     "wait for one player to join"),
    ("join",   "JOIN GAME",     "connect to a host by IP"),
]


class Menu:
    def __init__(self, font, big_font):
        self.font = font
        self.big_font = big_font
        self.state = 'mode'        # 'mode' -> 'join' -> 'hull' -> 'loadout' -> done
        self.mode_index = 0
        self.mode = 'single'       # 'single' | 'host' | 'join'
        self.join_text = ""        # join: raw 'IP:port' being typed
        self.host_ip = ""          # join: parsed host to connect to (on ENTER)
        self.host_port = 0         # join: parsed host port (on ENTER)
        self.hull_index = 0
        self.slot_index = 0
        self.slot_choice = self._stock_choices(self._hull()) 
        self.done = False
        self.hull = None
        self.loadout = None

    # --- loadout construction (pure data) ---

    def _hull(self):
        return PLAYER_HULLS[self.hull_index]

    def _stock_choices(self, hull):
        """slot_name -> catalog index, matching the hull's stock loadout."""
        stock = default_loadout(hull)
        out = {}
        for s in hull.slots:
            opts = COMPONENT_CATALOG[s.slot_type]
            comp = stock.get(s.name)
            out[s.name] = opts.index(comp) if comp in opts else 0
        return out

    def _current_loadout(self):
        return {s.name: COMPONENT_CATALOG[s.slot_type][
                    self.slot_choice.get(s.name, 0)]
                for s in self._hull().slots}

    def _confirm(self):
        """Launch: record the chosen hull/loadout (and mode) and finish."""
        self.hull = self._hull()
        self.loadout = self._current_loadout()
        self.done = True

    def _join_target(self):
        """Parse the typed 'IP:port' into (self.host_ip, self.host_port).

        Returns True when the entry is usable (a non-empty IP and a port in
        1..65535); the caller keeps the join screen up otherwise.
        """
        text = self.join_text.strip()
        if ":" in text:
            ip, _, port_s = text.rpartition(":")
        else:
            ip, port_s = text, ""
        self.host_ip = ip.strip()
        try:
            self.host_port = int(port_s)
        except ValueError:
            self.host_port = 0
        return bool(self.host_ip) and 1 <= self.host_port <= 65535

    # --- events ---

    def handle_events(self):
        """Returns False when the window should close."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if self.state == 'mode':
                    if event.key == pygame.K_ESCAPE:
                        return False
                    elif event.key in (pygame.K_UP, pygame.K_DOWN):
                        d = 1 if event.key == pygame.K_DOWN else -1
                        self.mode_index = (self.mode_index + d) % len(MODES)
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        self.mode = MODES[self.mode_index][0]
                        self.state = 'join' if self.mode == 'join' else 'hull'
                elif self.state == 'join':
                    if event.key == pygame.K_ESCAPE:
                        self.state = 'mode'
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        if self._join_target():
                            self.state = 'hull'
                    elif event.key == pygame.K_BACKSPACE:
                        self.join_text = self.join_text[:-1]
                    elif event.key == pygame.K_PERIOD:
                        self.join_text += "."
                    elif event.key == pygame.K_COLON:
                        self.join_text += ":"
                    elif event.unicode and event.unicode.isprintable() \
                            and event.unicode.isalnum():
                        self.join_text += event.unicode
                elif self.state == 'hull':
                    if event.key == pygame.K_ESCAPE:
                        self.state = 'mode'
                    elif event.key in (pygame.K_UP, pygame.K_DOWN):
                        n = len(PLAYER_HULLS)
                        d = 1 if event.key == pygame.K_DOWN else -1
                        self.hull_index = (self.hull_index + d) % n
                        self.slot_choice = self._stock_choices(self._hull())
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        self.state = 'loadout'
                elif self.state == 'loadout':
                    n = len(self._hull().slots)
                    if event.key in (pygame.K_UP, pygame.K_DOWN):
                        d = 1 if event.key == pygame.K_DOWN else -1
                        self.slot_index = (self.slot_index + d) % n
                    elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                        s = self._hull().slots[self.slot_index]
                        opts = COMPONENT_CATALOG[s.slot_type]
                        d = 1 if event.key == pygame.K_RIGHT else -1
                        i = self.slot_choice.get(s.name, 0)
                        self.slot_choice[s.name] = (i + d) % len(opts)
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        self._confirm()
                    elif event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
                        self.state = 'hull'
        return True

    # --- drawing ---

    def draw(self, screen):
        screen.fill(BG)
        if self.state == 'mode':
            self._draw_mode_screen(screen)
        elif self.state == 'join':
            self._draw_join_screen(screen)
        elif self.state == 'hull':
            self._draw_hull_screen(screen)
        else:
            self._draw_loadout_screen(screen)

    def _draw_mode_screen(self, screen):
        title = self.big_font.render("BELT FIGHTER", True, BRIGHT)
        screen.blit(title, (WIDTH / 2 - title.get_width() / 2, 60))

        x = int(WIDTH * 0.5)
        y = int(HEIGHT * 0.38)
        for i, (mode, label, desc) in enumerate(MODES):
            sel = i == self.mode_index
            if sel:
                pygame.draw.rect(screen, SEL_BG, (x - 16, y - 22, 460, 30))
            screen.blit(self.font.render(label, True,
                                         BRIGHT if sel else DIM), (x, y - 18))
            if sel:
                screen.blit(self.font.render(desc, True, ACCENT), (x, y + 12))
            y += 64

        hint = self.font.render(
            "UP/DOWN select   ENTER continue   ESC quit", True, DIM)
        screen.blit(hint, (8, HEIGHT - 24))

    def _draw_join_screen(self, screen):
        title = self.big_font.render("JOIN GAME", True, BRIGHT)
        screen.blit(title, (WIDTH / 2 - title.get_width() / 2, 40))

        x = int(WIDTH * 0.35)
        y = int(HEIGHT * 0.42)
        screen.blit(self.font.render("HOST ADDRESS  (IP:port)", True, DIM),
                    (x, y))
        box = (x - 10, y + 26, 420, 34)
        pygame.draw.rect(screen, SEL_BG, box)
        pygame.draw.rect(screen, ACCENT, box, 2)
        caret = "_" if (pygame.time.get_ticks() // 400) % 2 == 0 else ""
        screen.blit(self.font.render(self.join_text + caret, True, BRIGHT),
                    (x, y + 30))

        # live validation (pure — the parse that mutates host_ip/host_port
        # only happens on ENTER, in _join_target)
        text = self.join_text.strip()
        if text:
            ip, _, port_s = text.rpartition(":")
            try:
                p = int(port_s)
            except ValueError:
                p = 0
            if not ip.strip() or not 1 <= p <= 65535:
                screen.blit(self.font.render(
                    "need IP:port  (port 1-65535)", True, WARN), (x, y + 70))

        hint = self.font.render(
            "type IP:port   ENTER continue   ESC back", True, DIM)
        screen.blit(hint, (8, HEIGHT - 24))

    def _draw_hull_preview(self, screen, hull, cx, cy, scale, highlight=None):
        """Hull polygon + slot dots, oriented like the in-game ship (nose up)."""
        angle = -math.pi / 2
        fx, fy = math.cos(angle), math.sin(angle)
        rx, ry = -fy, fx

        def w(lx, ly):
            return (cx + (fx * lx + rx * ly) * scale,
                    cy + (fy * lx + ry * ly) * scale)


        pts = [w(*p) for p in hull.polygon]
        pygame.draw.polygon(screen, hull.fill or SHIP_COLOR, pts)
        
        #
        for shape, panel_fill in hull.panels:
            if isinstance(shape[0], str):
                c = w(shape[1], shape[2])
                pygame.draw.circle(screen, panel_fill,
                                   (int(c[0]), int(c[1])),
                                   max(1, int(shape[3] * scale)))
            else:
                pygame.draw.polygon(screen, panel_fill, [w(*p) for p in shape])

        
        for s in hull.slots:
            x, y = w(*s.position)
            if s.name == highlight:
                pygame.draw.circle(screen, ACCENT, (int(x), int(y)), 6)
                pygame.draw.circle(screen, BRIGHT, (int(x), int(y)), 2)
            else:
                pygame.draw.circle(screen, DIM, (int(x), int(y)), 3)

    def _draw_hull_screen(self, screen):
        self._draw_hull_preview(screen, self._hull(),
                                int(WIDTH * 0.28), int(HEIGHT * 0.52), 8)

        title = self.big_font.render("SELECT HULL", True, BRIGHT)
        screen.blit(title, (WIDTH / 2 - title.get_width() / 2, 40))

        x = int(WIDTH * 0.55)
        y = int(HEIGHT * 0.35)
        for i, h in enumerate(PLAYER_HULLS):
            sel = i == self.hull_index
            if sel:
                pygame.draw.rect(screen, SEL_BG, (x - 10, y - 20, 220, 26))
            screen.blit(self.font.render(h.id.upper(), True,
                                         BRIGHT if sel else DIM), (x, y - 16))
            y += 30

        hint = self.font.render(
            "UP/DOWN select   ENTER fit components   ESC back to mode", True, DIM)
        screen.blit(hint, (8, HEIGHT - 24))

    def _draw_loadout_screen(self, screen):
        hull = self._hull()
        slot = hull.slots[self.slot_index]

        self._draw_hull_preview(screen, hull,
                                int(WIDTH * 0.22), int(HEIGHT * 0.52), 7,
                                highlight=slot.name)

        title = self.big_font.render("FIT: " + hull.id.upper(), True, BRIGHT)
        screen.blit(title, (WIDTH / 2 - title.get_width() / 2, 40))

        # slot rows
        x = int(WIDTH * 0.42)
        y = int(HEIGHT * 0.28)
        for i, s in enumerate(hull.slots):
            opts = COMPONENT_CATALOG[s.slot_type]
            idx = self.slot_choice.get(s.name, 0)
            sel = i == self.slot_index
            if sel:
                pygame.draw.rect(screen, SEL_BG, (x - 10, y - 18, 300, 24))
            name = opts[idx].name
            shown = "<" + name + ">" if len(opts) > 1 else name
            screen.blit(self.font.render(f"{s.name:<12} {shown}", True,
                                         BRIGHT if sel else DIM), (x, y - 14))
            y += 26

        # detail line for the selected slot's part
        comp = COMPONENT_CATALOG[slot.slot_type][self.slot_choice.get(slot.name, 0)]
        parts = [f"mass {comp.mass:.1f}"]
        if comp.thrust:
            parts.append(f"thrust {comp.thrust:.0f}")
        if comp.power_idle or comp.power_active:
            parts.append(f"pwr {comp.power_idle:.0f}+{comp.power_active:.0f}")
        if comp.power_supply:
            parts.append(f"pwr+ {comp.power_supply:.0f}")
        if comp.compute_supply:
            parts.append(f"cpu+ {comp.compute_supply:.0f}")
        if comp.shield_max_charge:
            parts.append(f"shld {comp.shield_max_charge:.0f}")
        if comp.fire_cooldown:
            parts.append(f"rof {comp.fire_cooldown:.2f}s")
        if comp.laser_range:
            parts.append(f"range {comp.laser_range:.0f}")
        if comp.laser_arc_start_deg or comp.laser_arc_end_deg:
            span = comp.laser_arc_end_deg - comp.laser_arc_start_deg
            if span >= 360:
                parts.append("arc 360")
            else:
                parts.append(f"arc {comp.laser_arc_start_deg:.0f}..{comp.laser_arc_end_deg:.0f}")
        if comp.sensor_range:
            parts.append(f"sns {comp.sensor_range:.0f}")
        if comp.scan_cooldown:
            parts.append(f"scan {comp.scan_range:.0f} @ {comp.scan_cooldown:.0f}s")
        screen.blit(self.font.render("  ".join(parts), True, ACCENT), (x, y + 8))

        # stats panel
        stats = loadout_stats(hull, self._current_loadout())
        sx = int(WIDTH * 0.75)
        sy = int(HEIGHT * 0.28)
        rows = [
            ("MASS", f"{stats['mass']:.1f}"),
            ("PWR SUPPLY", f"{stats['power_supply']:.0f}"),
            ("PWR IDLE", f"{stats['power_idle']:.0f}"),
            ("PWR MAX", f"{stats['power_max']:.0f}"),
            ("CPU SUPPLY", f"{stats['compute_supply']:.0f}"),
            ("SHLD CHARGE", f"{stats['shield_charge']:.0f}"),
        ]
        for label, val in rows:
            warn = (label == "PWR MAX"
                    and stats['power_max'] > stats['power_supply'])
            screen.blit(self.font.render(f"{label:<12} {val}", True,
                                         WARN if warn else DIM), (sx, sy))
            sy += 24
        if stats['power_max'] > stats['power_supply']:
            screen.blit(self.font.render("BROWNOUT RISK", True, WARN), (sx, sy + 8))

        hint = self.font.render(
            "UP/DOWN slot   LEFT/RIGHT part   ENTER launch   ESC back", True, DIM)
        screen.blit(hint, (8, HEIGHT - 24))
